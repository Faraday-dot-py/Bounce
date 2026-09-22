"""Diagnostic script for docs/debugging/findings-gridding-artifact.md (Problem
1: grid/lattice artifact + edge brightening in stage2_flownet_h12 rollout).

Runs four checks against the trained checkpoint's DILATIONS=(1,2,4,8,4,2,1)
dilated-conv stack and grid_sample warp head:

1. Theoretical path-count coverage of the dilation stack (with residual skip
   connections, matching ResidualConvBlock) -- tests the classic "gridding
   hole" hypothesis (Wang et al., HDC) directly: are any pixels combined
   through zero paths, and is there high-frequency (2/4/8px-scale) periodic
   structure in path multiplicity?
2. FFT of actual rollout frames at steps 16/20/25/30 -- what spatial period
   does the observed lattice pattern actually have?
3. grid_sample out-of-bounds fraction/location, and |flow|/|correction|
   magnitude as a function of distance from the frame edge -- tests the
   zero-padding-discontinuity hypothesis.
4. Counterfactual: replay the SAME trained weights with grid_sample's
   padding_mode swapped from "zeros" to "border" (no retraining) to test
   whether the padding discontinuity is a causal driver of the lattice
   texture.

Usage:
    PYTHONPATH=. python3 scripts/investigate_gridding_artifact.py \
        --checkpoint checkpoints/stage2_flownet_h12.pt --out-dir /tmp/gridding_probe
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
from model.net import BounceNextFrameModel, DILATIONS


def path_count_coverage(size=65, with_residual=True):
    x = torch.zeros(1, 1, size, size)
    c = size // 2
    x[0, 0, c, c] = 1.0
    for d in DILATIONS:
        k = torch.ones(1, 1, 3, 3)
        h = F.conv2d(x, k, padding=d, dilation=d)
        h = F.conv2d(h, k, padding=d, dilation=d)
        x = x + h if with_residual else h
    return x[0, 0].numpy()


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


def instrumented_forward(model, g_t, padding_mode="zeros"):
    B, C, H, W = g_t.shape
    xf = model.stem(g_t)
    for block in model.blocks:
        xf = block(xf)
    flow = torch.tanh(model.flow_head(xf)) * model.max_flow
    correction = model.correction_head(xf)
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
    oob_mask = (norm_x.abs() > 1) | (norm_y.abs() > 1)
    warped = F.grid_sample(g_t, sample_grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)
    return warped + correction, flow, correction, oob_mask


def build_initial(n, num_balls, seed, radius=0.75, vy=2.3):
    rng = random.Random(seed)
    balls = make_scenario_uniform(num_balls, n, vy, rng)
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    return np.array(G, dtype=np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/stage2_flownet_h12.pt")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--num-balls", type=int, default=150)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--out-dir", default="/tmp/gridding_probe")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 70)
    print("1. Theoretical path-count coverage, DILATIONS =", DILATIONS)
    print("=" * 70)
    cov = path_count_coverage(with_residual=True)
    size = cov.shape[0]
    c = size // 2
    zeros = (cov[c - 20:c + 21, c - 20:c + 21] == 0).sum()
    print(f"zero-coverage cells in 41x41 window around impulse: {zeros}/1681")
    sub = cov[c - 16:c + 17, c - 16:c + 17]
    print(f"mean coverage even/odd row parity: {sub[0::2].mean():.1f} / {sub[1::2].mean():.1f}"
          f"  (ratio {sub[0::2].mean() / sub[1::2].mean():.4f})")
    peaks = fft_top_peaks(cov[c - 24:c + 25, c - 24:c + 25], n_peaks=4)
    print("top non-DC FFT peaks of coverage map (mag, freq, period_px):")
    for mag, fi, fj in peaks:
        pi = 1 / fi if fi else float("inf")
        pj = 1 / fj if fj else float("inf")
        print(f"  mag={mag:.3f} freq=({fi:+.3f},{fj:+.3f}) period=({pi:+.1f}px,{pj:+.1f}px)")

    model = BounceNextFrameModel(channels=64, depth=len(DILATIONS))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    g0 = build_initial(args.n, args.num_balls, args.seed)
    x0 = torch.from_numpy(g0.transpose(2, 0, 1)).unsqueeze(0)

    print()
    print("=" * 70)
    print("2+3. Rollout: OOB fraction, edge flow/correction magnitude, per-step stats")
    print("=" * 70)
    x = x0.clone()
    frames = [x[0].numpy().copy()]
    oob_masks = []
    with torch.no_grad():
        for step in range(30):
            out, flow, correction, oob_mask = instrumented_forward(model, x, "zeros")
            oob_masks.append(oob_mask[0].numpy().copy())
            if step == 0:
                flow_mag = flow[0].abs().mean(dim=0).numpy()
                d0 = np.concatenate([flow_mag[0, :], flow_mag[-1, :], flow_mag[:, 0], flow_mag[:, -1]]).mean()
                d1 = flow_mag[1:-1, 1:-1][0, :].mean()  # rough interior sample
                interior_mag = flow_mag[10:-10, 10:-10].mean()
                print(f"step1 |flow| at border ring: {d0:.4f}  vs interior (>=10px in): {interior_mag:.4f}"
                      f"  ratio {d0 / interior_mag:.2f}x")
            x = out
            frames.append(x[0].numpy().copy())
    frames = np.stack(frames)

    perim = 4 * args.n - 4
    oob_frac_step1 = oob_masks[0].mean()
    print(f"step1 OOB fraction: {oob_frac_step1:.4f}  (perimeter ring = {perim}/{args.n * args.n} = {perim / (args.n * args.n):.4f})")
    row_profile = oob_masks[0].sum(axis=1)
    print("OOB per-row counts (step1):", row_profile.astype(int).tolist())

    print()
    print("Per-step edge vs interior PROB mean (channel 0):")
    for s in (1, 8, 16, 20, 25, 30):
        p = frames[s, 0]
        edge = np.concatenate([p[0, :], p[-1, :], p[:, 0], p[:, -1]]).mean()
        interior = p[5:-5, 5:-5].mean()
        print(f"  step {s:2d}: edge={edge:+.4f} interior={interior:+.4f}")

    print()
    print("FFT peaks of interior 40x40 crop at each step (period in px):")
    for s in (16, 20, 25, 30):
        crop = frames[s, 0, 5:45, 5:45]
        peaks = fft_top_peaks(crop, min_freq=0.05, n_peaks=3)
        desc = ", ".join(f"period=({(1/fi if fi else float('inf')):.1f},{(1/fj if fj else float('inf')):.1f})px" for _, fi, fj in peaks)
        print(f"  step {s}: {desc}")

    print()
    print("=" * 70)
    print("4. Counterfactual: padding_mode zeros vs border (same weights, no retrain)")
    print("=" * 70)
    results = {}
    for pm in ("zeros", "border"):
        x = x0.clone()
        fr = [x[0, 0].numpy().copy()]
        with torch.no_grad():
            for _ in range(30):
                out, _, _, _ = instrumented_forward(model, x, pm)
                x = out
                fr.append(x[0, 0].numpy().copy())
        results[pm] = fr
        print(f"-- padding_mode={pm} --")
        for s in (16, 20, 25, 30):
            p = fr[s]
            edge = np.concatenate([p[0, :], p[-1, :], p[:, 0], p[:, -1]]).mean()
            interior = p[5:-5, 5:-5].mean()
            print(f"  step {s}: edge={edge:.4f} interior={interior:.4f} max={p.max():.4f}")

    steps_to_show = [1, 8, 16, 20, 25, 30]
    fig, axes = plt.subplots(2, len(steps_to_show), figsize=(3 * len(steps_to_show), 6))
    for row, pm in enumerate(("zeros", "border")):
        for col, s in enumerate(steps_to_show):
            axes[row, col].imshow(results[pm][s], cmap="inferno")
            axes[row, col].set_title(f"{pm} step {s}")
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    plt.tight_layout()
    out_path = os.path.join(args.out_dir, "counterfactual_padding_mode.png")
    plt.savefig(out_path, dpi=110)
    print(f"\nwrote {out_path}")
