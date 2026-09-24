"""Trivial position-error baselines on the same seeds/scenarios as
scripts/eval_free_rollout.py, for scaling its numbers.

stay: every token frozen at its frame-1 ground-truth ball position.
centroid: every token at the true ball centroid at each step (oracle mean position).
velocity: frame-1 to frame-2 constant velocity with no wall reflection (clamped to the grid).

Usage:
    PYTHONPATH=. python3 scripts/eval_trivial_baselines.py \
        --num-seeds 48 --num-steps 300 --report-steps 20,100,300 --out results/eval_baselines.json
"""
import argparse
import json

import torch

from scripts.diagnose_token_dropout import simulate


def baseline_errors(gt_x, gt_y, n):
    gt = torch.stack([gt_x, gt_y], dim=-1)
    T = gt.shape[0]
    stay = torch.norm(gt - gt[0], dim=-1).mean(dim=1)
    centroid = torch.norm(gt - gt.mean(dim=1, keepdim=True), dim=-1).mean(dim=1)
    vel = gt[1] - gt[0] if T > 1 else torch.zeros_like(gt[0])
    steps = torch.arange(T, dtype=torch.float32).view(T, 1, 1)
    extrap = (gt[0] + vel * steps).clamp(0, n)
    velocity = torch.norm(gt - extrap, dim=-1).mean(dim=1)
    return stay, centroid, velocity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--num-balls", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--num-seeds", type=int, default=48)
    ap.add_argument("--base-seed", type=int, default=4738)
    ap.add_argument("--report-steps", type=str, default="5,10,20")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    report_steps = [int(s) for s in args.report_steps.split(",")]

    acc = {"stay": [], "centroid": [], "velocity": []}
    for seed in range(args.base_seed, args.base_seed + args.num_seeds):
        _, states = simulate(args.n, args.num_balls, seed, args.num_steps)
        gt_x = torch.tensor([s["x"] for s in states[1:args.num_steps + 1]], dtype=torch.float32)
        gt_y = torch.tensor([s["y"] for s in states[1:args.num_steps + 1]], dtype=torch.float32)
        for name, e in zip(acc, baseline_errors(gt_x, gt_y, args.n)):
            acc[name].append(e)

    result = {"n": args.n, "num_balls": args.num_balls, "num_seeds": args.num_seeds, "num_steps": args.num_steps}
    for name, es in acc.items():
        m = torch.stack(es).mean(dim=0)
        result[name] = {str(k): float(m[min(k, len(m) - 1)]) for k in report_steps}
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
