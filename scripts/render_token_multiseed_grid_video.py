"""Renders a 4x4 grid of ground-truth-vs-token-model rollout comparisons,
one per seed, animated over rollout steps. Each of the 16 cells shows its
own ground-truth frame and model-predicted frame side by side (GT | model),
so this is 16 independent side-by-side comparisons in one video, useful for
seeing whether a failure mode (e.g. give-up dissolution) is consistent
across different starting configurations or seed-dependent.

Usage:
    PYTHONPATH=. python3 scripts/render_token_multiseed_grid_video.py \
        --checkpoint checkpoints/token_model_h12_v2.pt \
        --out videos/token_model_v2_multiseed.mp4 \
        --num-steps 30
"""
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model.token_model import TokenModel
from render_token_rollout_video import simulate_ground_truth, rollout  # noqa: E402


def load_model(checkpoint_path, n, hidden_dim, neighbor_radius, velocity_weight=0.0,
                territory_masking=False):
    model = TokenModel(n=n, radius=0.75, dt=0.15, hidden_dim=hidden_dim, neighbor_radius=neighbor_radius,
                        velocity_weight=velocity_weight, territory_masking=territory_masking)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def combined_frame(gt_frame, pred_frame, gap_width=1):
    """Concatenates ground-truth and predicted PROB channels horizontally
    with a thin constant-value gap column, so each grid cell is itself a
    GT | model side-by-side comparison."""
    gt = gt_frame[:, :, 0]
    pred = pred_frame[:, :, 0]
    gap = np.full((gt.shape[0], gap_width), np.nan, dtype=np.float32)
    return np.concatenate([gt, gap, pred], axis=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--num-balls", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--neighbor-radius", type=float, default=4.0)
    ap.add_argument("--velocity-weight", type=float, default=0.0)
    ap.add_argument("--territory-masking", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(4738, 4738 + 16)))
    args = ap.parse_args()

    if len(args.seeds) != 16:
        raise ValueError(f"expected exactly 16 seeds, got {len(args.seeds)}")

    model = load_model(args.checkpoint, args.n, args.hidden_dim, args.neighbor_radius, args.velocity_weight,
                        territory_masking=args.territory_masking)

    all_gt_frames = []
    all_pred_frames = []
    for seed in args.seeds:
        gt_frames = simulate_ground_truth(args.n, args.num_balls, seed, args.num_steps)
        pred_frames = rollout(model, gt_frames[0], gt_frames[1], args.num_steps)
        all_gt_frames.append(gt_frames)
        all_pred_frames.append(pred_frames)

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    images = []
    for idx, ax in enumerate(axes.flat):
        seed = args.seeds[idx]
        combined0 = combined_frame(all_gt_frames[idx][0], all_pred_frames[idx][0])
        im = ax.imshow(combined0, vmin=0, vmax=1, cmap=cmap)
        ax.set_title(f"seed {seed} (GT | model)", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        images.append(im)

    def update(i):
        for idx, im in enumerate(images):
            im.set_data(combined_frame(all_gt_frames[idx][i], all_pred_frames[idx][i]))
        return images

    num_frames = len(all_gt_frames[0])
    plt.tight_layout()
    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1000 / 12)
    ani.save(args.out, writer="ffmpeg", fps=12)
    print(f"saved {args.out}", flush=True)
