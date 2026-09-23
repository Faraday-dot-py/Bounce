# scripts/render_token_rollout_video.py
"""Renders a ground-truth vs. token-model-rollout comparison video for a
TokenModel checkpoint.

Usage:
    PYTHONPATH=. python3 scripts/render_token_rollout_video.py \
        --checkpoint checkpoint_token.pt \
        --out videos/token_model_rollout_comparison.mp4 \
        --num-steps 30
"""
import argparse
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import bounce
from model.dataset import make_scenario_uniform
from model.token_model import TokenModel


def load_model(checkpoint_path, n, hidden_dim, neighbor_radius, velocity_weight=0.0,
                territory_masking=False):
    model = TokenModel(n=n, radius=0.75, dt=0.15, hidden_dim=hidden_dim, neighbor_radius=neighbor_radius,
                        velocity_weight=velocity_weight, territory_masking=territory_masking)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


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
        prev_obs_pos = positions
        for _ in range(num_steps - 1):
            positions, velocities, hidden, pred_grid, prev_obs_pos = model.step(
                positions, velocities, hidden, observed, prev_obs_pos
            )
            frames.append(pred_grid.numpy().transpose(1, 2, 0))
            observed = pred_grid
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--num-balls", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--neighbor-radius", type=float, default=3.0)
    ap.add_argument("--velocity-weight", type=float, default=0.0)
    ap.add_argument("--territory-masking", action="store_true")
    ap.add_argument("--gravity", type=float, default=9.0)
    ap.add_argument("--seed", type=int, default=4738)
    args = ap.parse_args()

    model = load_model(args.checkpoint, args.n, args.hidden_dim, args.neighbor_radius, args.velocity_weight,
                        territory_masking=args.territory_masking)
    gt_frames = simulate_ground_truth(args.n, args.num_balls, args.seed, args.num_steps, gravity=args.gravity)
    pred_frames = rollout(model, gt_frames[0], gt_frames[1], args.num_steps)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].set_title("ground truth")
    axes[1].set_title("token model")
    im0 = axes[0].imshow(gt_frames[0][:, :, 0], vmin=0, vmax=1, cmap="viridis")
    im1 = axes[1].imshow(pred_frames[0][:, :, 0], vmin=0, vmax=1, cmap="viridis")

    def update(i):
        im0.set_data(gt_frames[i][:, :, 0])
        im1.set_data(pred_frames[i][:, :, 0])
        return im0, im1

    ani = animation.FuncAnimation(fig, update, frames=len(gt_frames), interval=1000 / 12)
    ani.save(args.out, writer="ffmpeg", fps=12)
    print(f"saved {args.out}", flush=True)
