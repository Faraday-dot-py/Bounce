"""Diagnostic tool: compares a checkpoint's single-step (step-1) rollout
prediction MSE against a trivial "copy the input frame forward" baseline,
averaged over several seeds. A model that can't beat this baseline is
producing a blurred/diffuse prediction rather than confidently tracking
ball positions, regardless of what its training loss curve shows.

Usage:
    PYTHONPATH=. python3 scripts/eval_step1_baseline.py \
        --checkpoints stage2=checkpoints/stage2.pt stage2_200ep=checkpoints/stage2_200ep.pt \
        --seeds 4738 4739 4740 4741 4742 4743 4744 4745 4746 4747
"""
import argparse
import random

import numpy as np
import torch

import bounce
from model.dataset import make_scenario_uniform
from model.net import BounceNextFrameModel


def build_initial(n, num_balls, seed, radius=0.75, vy=2.3):
    rng = random.Random(seed)
    balls = make_scenario_uniform(num_balls, n, vy, rng)
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    return balls, np.array(G, dtype=np.float32)


def parse_checkpoints(pairs):
    result = {}
    for pair in pairs:
        name, path = pair.split("=", 1)
        result[name] = path
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                     help="name=path pairs, e.g. stage2=checkpoints/stage2.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(4738, 4748)))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--num-balls", type=int, default=150)
    args = ap.parse_args()

    checkpoints = parse_checkpoints(args.checkpoints)
    models = {}
    for name, path in checkpoints.items():
        model = BounceNextFrameModel(channels=64, depth=7)
        model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        model.eval()
        models[name] = model

    copy_mses = []
    model_mses = {name: [] for name in checkpoints}
    for seed in args.seeds:
        balls, g0 = build_initial(args.n, args.num_balls, seed)
        G = bounce.make_grid(args.n)
        bounce.step(G, args.n, balls, dt=0.15, gravity=9.0, radius=0.75, stiffness=400.0, substeps=8)
        g1_true = np.array(G, dtype=np.float32)
        prob_true = g1_true[:, :, 0]
        copy_mses.append(((g0[:, :, 0] - prob_true) ** 2).mean())

        x = torch.from_numpy(g0.transpose(2, 0, 1)).unsqueeze(0)
        for name, model in models.items():
            with torch.no_grad():
                out = model(x)
            prob_pred = out[0, 0].numpy()
            model_mses[name].append(((prob_pred - prob_true) ** 2).mean())

    copy_mean = np.mean(copy_mses)
    print(f"copy-input baseline MSE (mean over {len(args.seeds)} seeds): {copy_mean:.6f}")
    for name in checkpoints:
        m = np.mean(model_mses[name])
        print(f"{name}: MSE={m:.6f}  ratio={m / copy_mean:.2f}x")
