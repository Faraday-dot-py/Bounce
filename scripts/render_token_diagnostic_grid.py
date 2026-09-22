"""Diagnostic tool: renders a grid of (ground-truth vs token-model) frames at
selected rollout steps, each frame auto-scaled to its own min/max, so
structural issues (frozen/near-static content, drift, artifacts) are visible
even once absolute magnitude has decayed. Mirrors
scripts/render_diagnostic_grid.py's convention for the flow-warp models.

Usage:
    PYTHONPATH=. python3 scripts/render_token_diagnostic_grid.py \
        --checkpoint checkpoints/token_model_h12_v1.pt \
        --steps 0 1 2 3 5 8 12 20 \
        --out /tmp/token_artifact_grid.png
"""
import argparse
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bounce
from model.dataset import make_scenario_uniform
from model.token_model import TokenModel


def simulate_ground_truth(n, num_balls, seed, num_steps, dt=0.15, gravity=9.0,
                           radius=0.75, stiffness=400.0, substeps=8, vy=2.3):
    rng = random.Random(seed)
    balls = make_scenario_uniform(num_balls, n, vy, rng)
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    frames = [np.array(G, dtype=np.float32)]
    for _ in range(num_steps):
        bounce.step(G, n, balls, dt, gravity, radius, stiffness, substeps)
        frames.append(np.array(G, dtype=np.float32))
    return frames


def rollout(model, frame0, frame1, num_steps):
    g0 = torch.from_numpy(frame0.transpose(2, 0, 1))
    g1 = torch.from_numpy(frame1.transpose(2, 0, 1))
    frames = [frame0, frame1]
    with torch.no_grad():
        positions, velocities, hidden = model.init_tokens(g0, g1)
        observed = g1
        for _ in range(num_steps - 1):
            positions, velocities, hidden, pred_grid = model.step(positions, velocities, hidden, observed)
            frames.append(pred_grid.numpy().transpose(1, 2, 0))
            observed = pred_grid
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 1, 2, 3, 5, 8, 12, 20])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--num-balls", type=int, default=4)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--neighbor-radius", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--out", type=str, default="/tmp/token_artifact_grid.png")
    args = ap.parse_args()

    num_steps = max(args.steps)
    gt_frames = simulate_ground_truth(args.n, args.num_balls, args.seed, num_steps)

    model = TokenModel(n=args.n, radius=0.75, dt=0.15, hidden_dim=args.hidden_dim,
                        neighbor_radius=args.neighbor_radius)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    pred_frames = rollout(model, gt_frames[0], gt_frames[1], num_steps)

    rows = {"ground_truth": gt_frames, "token_model": pred_frames}
    fig, axes = plt.subplots(
        len(rows), len(args.steps),
        figsize=(2 * len(args.steps), 2 * len(rows)),
        squeeze=False,
    )
    for r, (name, frames) in enumerate(rows.items()):
        for c, step in enumerate(args.steps):
            ax = axes[r][c]
            frame = frames[step][:, :, bounce.PROB]
            ax.imshow(frame, cmap="viridis")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"step {step}")
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center")
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"saved {args.out}")
