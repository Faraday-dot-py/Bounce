"""Renders a ground-truth vs. model-rollout comparison video for any
BounceNextFrameModel checkpoint (generalizes render_rollout_video.py,
which is hardcoded to the old stage1/stage2 checkpoints).

Usage:
    PYTHONPATH=. python3 scripts/render_flownet_rollout_video.py \
        --checkpoint checkpoints/stage2_flownet_h12_v6.pt \
        --out videos/stage2_flownet_v6_rollout_comparison.mp4 \
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
from model.net import BounceNextFrameModel


def load_model(checkpoint_path):
    model = BounceNextFrameModel(channels=64, depth=7)
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


def rollout(model, g0, num_steps):
    g_pred = torch.from_numpy(g0.transpose(2, 0, 1)).unsqueeze(0)
    frames = [g0]
    with torch.no_grad():
        for _ in range(num_steps):
            g_pred = model(g_pred)
            frames.append(g_pred[0].numpy().transpose(1, 2, 0))
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--num-balls", type=int, default=150)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--num-steps", type=int, default=30)
    args = ap.parse_args()

    model = load_model(args.checkpoint)
    truth = simulate_ground_truth(args.n, args.num_balls, args.seed, args.num_steps)
    pred = rollout(model, truth[0], args.num_steps)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.3))
    for ax, title in zip(axes, ["ground truth", "model (rollout)"]):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    ims = [
        axes[0].imshow(truth[0][:, :, bounce.PROB], vmin=0, vmax=1, cmap="inferno"),
        axes[1].imshow(pred[0][:, :, bounce.PROB], vmin=0, vmax=1, cmap="inferno"),
    ]
    step_text = fig.suptitle("step 0")

    def update(i):
        ims[0].set_data(truth[i][:, :, bounce.PROB])
        ims[1].set_data(pred[i][:, :, bounce.PROB])
        step_text.set_text(f"step {i}")
        return ims + [step_text]

    anim = animation.FuncAnimation(fig, update, frames=args.num_steps + 1, interval=300, blit=False)
    anim.save(args.out, writer="ffmpeg", fps=3, dpi=120)
    print(f"wrote {args.out}")
