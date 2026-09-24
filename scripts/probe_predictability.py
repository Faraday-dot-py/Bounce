"""Spike: how long does the simulator stay predictable from a slightly wrong
initial state? Runs twin simulations (same scenario generator as
scripts/diagnose_token_dropout.py) with frame-0 positions and velocities
perturbed by Gaussian noise of scale eps, and reports mean per-ball position
divergence vs step. Throwaway.

Usage:
    PYTHONPATH=. python3 scripts/probe_predictability.py --out results/probe_predictability.json
"""
import argparse
import json
import random

import numpy as np

import bounce
from scripts.diagnose_token_dropout import make_scenario_uniform


def run(balls, n, num_steps, dt, gravity, radius, stiffness, substeps):
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    traj = []
    for _ in range(num_steps):
        bounce.step(G, n, balls, dt, gravity, radius, stiffness, substeps)
        traj.append(np.array([[b["x"], b["y"]] for b in balls]))
    return np.stack(traj)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--num-balls", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--num-seeds", type=int, default=48)
    ap.add_argument("--base-seed", type=int, default=4738)
    ap.add_argument("--eps", type=str, default="0.001,0.01,0.1")
    ap.add_argument("--report-steps", type=str, default="5,10,20,50,100")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    eps_list = [float(e) for e in args.eps.split(",")]
    report_steps = [int(s) for s in args.report_steps.split(",")]
    dt, gravity, radius, stiffness, substeps, vy = 0.15, 9.0, 0.75, 400.0, 8, 2.3

    result = {"n": args.n, "num_balls": args.num_balls, "num_seeds": args.num_seeds, "eps": {}}
    for eps in eps_list:
        div = []
        for seed in range(args.base_seed, args.base_seed + args.num_seeds):
            rng = random.Random(seed)
            balls = make_scenario_uniform(args.num_balls, args.n, vy, rng)
            noise = np.random.RandomState(seed)
            twin = [dict(b) for b in balls]
            for b in twin:
                b["x"] += eps * noise.randn()
                b["y"] += eps * noise.randn()
                b["vx"] += eps * noise.randn()
                b["vy"] += eps * noise.randn()
            a = run(balls, args.n, args.num_steps, dt, gravity, radius, stiffness, substeps)
            b = run(twin, args.n, args.num_steps, dt, gravity, radius, stiffness, substeps)
            div.append(np.linalg.norm(a - b, axis=-1).mean(axis=1))
        div = np.stack(div)
        mean = div.mean(axis=0)
        result["eps"][str(eps)] = {
            "mean_divergence": {str(k): float(mean[min(k, len(mean)) - 1]) for k in report_steps},
            "median_steps_to_1_cell": float(np.median([(d > 1.0).argmax() + 1 if (d > 1.0).any() else args.num_steps + 1 for d in div])),
            "median_steps_to_3_cells": float(np.median([(d > 3.0).argmax() + 1 if (d > 3.0).any() else args.num_steps + 1 for d in div])),
        }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
