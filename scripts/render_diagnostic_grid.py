"""Diagnostic tool: renders a grid of (checkpoint x rollout step) frames with
per-frame auto-scaled color limits, so structural artifacts (checkerboards,
stripes, hallucinated patterns) are visible even once absolute magnitude has
decayed below a fixed display range. Use this instead of fixed vmin/vmax
comparison videos when debugging rollout degradation - fixed scaling can hide
a growing artifact by making it look like it "faded to black".

Usage:
    PYTHONPATH=. python3 scripts/render_diagnostic_grid.py \
        --checkpoints stage1=checkpoints/stage1.pt stage2=checkpoints/stage2.pt \
        --steps 0 1 2 3 5 8 12 20 \
        --out /tmp/artifact_grid.png
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
from model.net import BounceNextFrameModel


def load_model(checkpoint_path):
    model = BounceNextFrameModel(channels=64, depth=7)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def build_initial_grid(n, num_balls, seed, radius=0.75, vy=2.3):
    rng = random.Random(seed)
    balls = make_scenario_uniform(num_balls, n, vy, rng)
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    return np.array(G, dtype=np.float32)


def rollout_frames(model, g0, num_steps):
    g_pred = torch.from_numpy(g0.transpose(2, 0, 1)).unsqueeze(0)
    frames = {0: g0[:, :, bounce.PROB]}
    with torch.no_grad():
        for i in range(1, num_steps + 1):
            g_pred = model(g_pred)
            frames[i] = g_pred[0, bounce.PROB].numpy()
    return frames


def parse_checkpoints(pairs):
    result = {}
    for pair in pairs:
        name, path = pair.split("=", 1)
        result[name] = path
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                     help="name=path pairs, e.g. stage1=checkpoints/stage1.pt")
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 1, 2, 3, 5, 8, 12, 20])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--num-balls", type=int, default=150)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--out", type=str, default="/tmp/artifact_grid.png")
    args = ap.parse_args()

    checkpoints = parse_checkpoints(args.checkpoints)
    num_steps = max(args.steps)
    g0 = build_initial_grid(args.n, args.num_balls, args.seed)

    fig, axes = plt.subplots(
        len(checkpoints), len(args.steps),
        figsize=(2 * len(args.steps), 2 * len(checkpoints)),
        squeeze=False,
    )
    for row, (name, path) in enumerate(checkpoints.items()):
        model = load_model(path)
        frames = rollout_frames(model, g0, num_steps)
        for col, step in enumerate(args.steps):
            ax = axes[row, col]
            ax.imshow(frames[step], cmap="inferno")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"step {step}")
            if col == 0:
                ax.set_ylabel(name, fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")
