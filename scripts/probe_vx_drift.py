"""Spike: horizontal velocity statistics vs step, ground truth vs model
free rollout (mean vx, mean |vx|, mean |vy|, mean speed). Throwaway.

Usage:
    PYTHONPATH=. python3 scripts/probe_vx_drift.py --checkpoint checkpoints/token_model_h24_v18.pt
"""
import argparse

import torch

from scripts.diagnose_token_dropout import simulate
from scripts.eval_free_rollout import load_model


def stats(vel):
    return torch.stack([vel[..., 0].mean(), vel[..., 0].abs().mean(), vel[..., 1].mean(), vel[..., 1].abs().mean(), vel.norm(dim=-1).mean()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mirror-sym", action="store_true")
    ap.add_argument("--velocity-readout", action="store_true")
    ap.add_argument("--wall-lookahead", action="store_true")
    ap.add_argument("--wall-head", action="store_true")
    ap.add_argument("--pair-impulse", action="store_true")
    ap.add_argument("--position-refine", action="store_true")
    ap.add_argument("--num-seeds", type=int, default=48)
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--base-seed", type=int, default=4738)
    args = ap.parse_args()
    report = [1, 5, 10, 20, 50, 100]
    model = load_model(args.checkpoint, "free", 20, 32, 4.0, args.mirror_sym, args.velocity_readout, args.wall_lookahead, args.wall_head, args.pair_impulse, args.position_refine)
    rows = {"model": [], "truth": []}
    for seed in range(args.base_seed, args.base_seed + args.num_seeds):
        frames, states = simulate(20, 4, seed, args.num_steps + 1)
        gt = torch.stack([torch.tensor([[x, y] for x, y in zip(s["x"], s["y"])], dtype=torch.float32) for s in states])
        gt_v = (gt[1:] - gt[:-1]) / 0.15
        g0 = torch.from_numpy(frames[0].transpose(2, 0, 1))
        g1 = torch.from_numpy(frames[1].transpose(2, 0, 1))
        with torch.no_grad():
            p, v, h = model.init_tokens(g0, g1)
            if p.shape[0] != 4:
                continue
            vs = [v]
            for _ in range(args.num_steps - 1):
                p, v, h, _ = model.step_free(p, v, h)
                vs.append(v)
        rows["model"].append(torch.stack([stats(x) for x in vs]))
        rows["truth"].append(torch.stack([stats(x) for x in gt_v[:args.num_steps]]))
    print("step: mean vx | mean|vx| | mean vy | mean|vy| | speed")
    for name, r in rows.items():
        m = torch.stack(r).mean(dim=0)
        print(name)
        for k in report:
            print(f"  {k:3d}: " + " ".join(f"{float(x):7.3f}" for x in m[k - 1]))


if __name__ == "__main__":
    main()
