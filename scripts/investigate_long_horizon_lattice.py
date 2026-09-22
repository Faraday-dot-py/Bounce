"""Diagnostic script for docs/debugging/findings-long-horizon-lattice.md
(Problem 3, flownet-open-issues-v2.md): what is the periodic diagonal
ripple/checkerboard that grows to dominate the v7 100-step rollout by
steps 90-100?

Reuses the exact FFT methodology (fft_top_peaks, interior 40x40 crop,
Hanning window) and padding_mode counterfactual method from
scripts/investigate_gridding_artifact.py / findings-gridding-artifact.md
so results are directly comparable.

Usage:
    PYTHONPATH=. python3 scripts/investigate_long_horizon_lattice.py \
        --out-dir /tmp/long_horizon_lattice
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bounce
from model.dataset import make_scenario_uniform
from model.net import BounceNextFrameModel


def fft_top_peaks(img2d, min_freq=0.03, n_peaks=5):
    img2d = img2d.astype(np.float64)
    img2d = img2d - img2d.mean()
    win = np.outer(np.hanning(img2d.shape[0]), np.hanning(img2d.shape[1]))
    spec = np.abs(np.fft.fftshift(np.fft.fft2(img2d * win)))
    spec_n = spec / (spec.max() + 1e-12)
    freqs0 = np.fft.fftshift(np.fft.fftfreq(img2d.shape[0]))
    freqs1 = np.fft.fftshift(np.fft.fftfreq(img2d.shape[1]))
    flat = []
    for i in range(img2d.shape[0]):
        for j in range(img2d.shape[1]):
            fr = np.hypot(freqs0[i], freqs1[j])
            if fr > min_freq:
                flat.append((spec_n[i, j], freqs0[i], freqs1[j]))
    flat.sort(reverse=True)
    return flat[:n_peaks]


def describe_peaks(peaks):
    return ", ".join(
        f"mag={m:.3f} period=({(1/fi if fi else float('inf')):+.1f},{(1/fj if fj else float('inf')):+.1f})px"
        for m, fi, fj in peaks
    )


def load_model(checkpoint_path):
    model = BounceNextFrameModel(channels=64, depth=7)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def build_initial(n, num_balls, seed, radius=0.75, vy=2.3):
    rng = random.Random(seed)
    balls = make_scenario_uniform(num_balls, n, vy, rng)
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    return np.array(G, dtype=np.float32)


def real_rollout(model, g0, num_steps):
    """Full model.forward() rollout -- real production path (renorm,
    threshold, VX/VY recenter, padding_mode='border', bicubic all
    included, exactly as trained/shipped)."""
    x = torch.from_numpy(g0.transpose(2, 0, 1)).unsqueeze(0)
    frames = [x[0].numpy().copy()]
    with torch.no_grad():
        for _ in range(num_steps):
            x = model(x)
            frames.append(x[0].numpy().copy())
    return np.stack(frames)


def instrumented_forward(model, g_t, padding_mode, grid_sample_mode="bicubic"):
    """Mirrors BounceNextFrameModel.forward exactly (renorm, threshold,
    VX/VY recenter all included), except grid_sample's padding_mode (and
    optionally its interpolation mode) is swappable at call time -- for
    the inference-only counterfactual, no retraining, same method as
    investigate_gridding_artifact.py's instrumented_forward."""
    B, C, H, W = g_t.shape
    x = model.stem(g_t)
    for block in model.blocks:
        x = block(x)
    flow = torch.tanh(model.flow_head(x)) * model.max_flow
    correction = torch.tanh(model.correction_head(x)) * model.max_correction
    correction = correction - correction.mean(dim=(2, 3), keepdim=True)

    ys, xs = torch.meshgrid(
        torch.arange(H, device=g_t.device, dtype=g_t.dtype),
        torch.arange(W, device=g_t.device, dtype=g_t.dtype),
        indexing="ij",
    )
    sample_x = xs.unsqueeze(0) + flow[:, 0]
    sample_y = ys.unsqueeze(0) + flow[:, 1]
    norm_x = sample_x / max(W - 1, 1) * 2 - 1
    norm_y = sample_y / max(H - 1, 1) * 2 - 1
    sample_grid = torch.stack([norm_x, norm_y], dim=-1)

    warped = F.grid_sample(g_t, sample_grid, mode=grid_sample_mode, padding_mode=padding_mode, align_corners=True)
    out = warped + correction

    prob_thresholded = F.relu(out[:, 0:1] - model.prob_threshold)
    prob_in = g_t[:, 0:1].sum(dim=(2, 3), keepdim=True)
    prob_out = prob_thresholded.sum(dim=(2, 3), keepdim=True)
    scale = prob_in / (prob_out + 1e-6)
    prob_final = prob_thresholded * scale

    vxvy_in_mean = g_t[:, 1:].mean(dim=(2, 3), keepdim=True)
    vxvy_out_mean = out[:, 1:].mean(dim=(2, 3), keepdim=True)
    vxvy_final = out[:, 1:] - vxvy_out_mean + vxvy_in_mean

    return torch.cat([prob_final, vxvy_final], dim=1)


def instrumented_rollout(model, g0, num_steps, padding_mode, grid_sample_mode="bicubic"):
    x = torch.from_numpy(g0.transpose(2, 0, 1)).unsqueeze(0)
    frames = [x[0].numpy().copy()]
    with torch.no_grad():
        for _ in range(num_steps):
            x = instrumented_forward(model, x, padding_mode, grid_sample_mode)
            frames.append(x[0].numpy().copy())
    return np.stack(frames)


def crop_prob(frames, step):
    # same interior 40x40 crop of the 50x50 grid used in
    # investigate_gridding_artifact.py / findings-gridding-artifact.md
    return frames[step, 0, 5:45, 5:45]


def amplitude(crop):
    return float(crop.std())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--num-balls", type=int, default=150)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--out-dir", default="/tmp/long_horizon_lattice")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    g0 = build_initial(args.n, args.num_balls, args.seed)
    check_steps = [5, 10, 20, 40, 60, 80, 100]

    print("=" * 70)
    print("1+2. Real (production forward()) rollouts -- v7 and v6, FFT of interior crop")
    print("=" * 70)
    all_frames = {}
    for tag, ckpt in [("v7", "checkpoints/stage2_flownet_h12_v7.pt"),
                       ("v6", "checkpoints/stage2_flownet_h12_v6.pt")]:
        model = load_model(ckpt)
        frames = real_rollout(model, g0, args.num_steps)
        all_frames[tag] = frames
        np.save(os.path.join(args.out_dir, f"{tag}_prob_frames.npy"), frames[:, 0])
        print(f"\n-- {tag} ({ckpt}) --")
        for s in check_steps:
            crop = crop_prob(frames, s)
            peaks = fft_top_peaks(crop, min_freq=0.05, n_peaks=3)
            amp = amplitude(crop)
            print(f"  step {s:3d}: amp(std)={amp:.5f} max={crop.max():.4f}  {describe_peaks(peaks)}")

    print()
    print("=" * 70)
    print("3. Is the lattice present (low-amplitude) at early steps 0-20 in v7, and in v6 at 80-100?")
    print("=" * 70)
    for tag in ("v7", "v6"):
        frames = all_frames[tag]
        print(f"\n-- {tag} early steps --")
        for s in (0, 2, 5, 10, 15, 20):
            crop = crop_prob(frames, s)
            peaks = fft_top_peaks(crop, min_freq=0.05, n_peaks=2)
            print(f"  step {s:3d}: amp(std)={amplitude(crop):.5f}  {describe_peaks(peaks)}")

    print()
    print("=" * 70)
    print("4. padding_mode counterfactual on v7 (same trained weights, inference-only swap)")
    print("   NOTE: current shipped default is padding_mode='border' (already the fix from")
    print("   findings-gridding-artifact.md); 'zeros' here is the OLD counterfactual, replayed")
    print("   at long horizon to see if reverting it changes the long-horizon lattice.")
    print("=" * 70)
    model_v7 = load_model("checkpoints/stage2_flownet_h12_v7.pt")
    pm_frames = {}
    for pm in ("border", "zeros"):
        fr = instrumented_rollout(model_v7, g0, args.num_steps, padding_mode=pm)
        pm_frames[pm] = fr
        print(f"\n-- padding_mode={pm} --")
        for s in check_steps:
            crop = crop_prob(fr, s)
            peaks = fft_top_peaks(crop, min_freq=0.05, n_peaks=3)
            print(f"  step {s:3d}: amp(std)={amplitude(crop):.5f} max={crop.max():.4f}  {describe_peaks(peaks)}")

    # sanity check: instrumented "border" rollout should closely match real forward() v7 rollout
    diff = np.abs(pm_frames["border"][:, 0] - all_frames["v7"][:, 0]).max()
    print(f"\nsanity: max abs diff between instrumented(border) and real forward() PROB frames: {diff:.6f}")

    steps_to_show = [1, 20, 40, 60, 80, 100]
    fig, axes = plt.subplots(4, len(steps_to_show), figsize=(3 * len(steps_to_show), 12))
    rows = [("v7", all_frames["v7"][:, 0]), ("v6", all_frames["v6"][:, 0]),
            ("v7 border(cf)", pm_frames["border"][:, 0]), ("v7 zeros(cf)", pm_frames["zeros"][:, 0])]
    for row, (label, fr) in enumerate(rows):
        for col, s in enumerate(steps_to_show):
            axes[row, col].imshow(fr[s], cmap="inferno")
            axes[row, col].set_title(f"{label} step {s}")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "long_horizon_lattice_grid.png")
    plt.savefig(out_path, dpi=110)
    print(f"\nwrote {out_path}")
