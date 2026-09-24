"""Spike: teacher-forced one-step vy residual (model minus truth) on real
scenarios, grouped by token context, to localize the y-momentum bias. At each
step the token positions/velocities are reset to ground truth; the GRU hidden
state is carried. Throwaway.

Usage:
    PYTHONPATH=. python3 scripts/probe_y_bias_context.py --checkpoint checkpoints/token_model_h24_v18.pt
"""
import argparse
from collections import defaultdict

import torch

from scripts.diagnose_token_dropout import simulate
from scripts.eval_free_rollout import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-seeds", type=int, default=48)
    ap.add_argument("--num-steps", type=int, default=25)
    ap.add_argument("--base-seed", type=int, default=4738)
    args = ap.parse_args()
    model = load_model(args.checkpoint, "free", 20, 32, 4.0)
    res = defaultdict(list)
    for seed in range(args.base_seed, args.base_seed + args.num_seeds):
        frames, states = simulate(20, 4, seed, args.num_steps + 2)
        gt = torch.stack([torch.tensor([[x, y] for x, y in zip(s["x"], s["y"])], dtype=torch.float32) for s in states])
        gt_v = (gt[1:] - gt[:-1]) / 0.15
        g0 = torch.from_numpy(frames[0].transpose(2, 0, 1))
        g1 = torch.from_numpy(frames[1].transpose(2, 0, 1))
        with torch.no_grad():
            p, v, h = model.init_tokens(g0, g1)
            if p.shape[0] != 4:
                continue
            for t in range(1, args.num_steps):
                p = gt[t]; v = gt_v[t - 1]
                np_, nv, h, _ = model.step_free(p, v, h)
                dv = nv - gt_v[t]
                dist = torch.cdist(p, p) + torch.eye(4) * 99
                near_nb = dist.min(dim=1).values < 3.0
                for i in range(4):
                    y, x = float(p[i, 1]), float(p[i, 0])
                    ctx = []
                    if y < 3: ctx.append("y-wall(y<3)")
                    elif y > 16: ctx.append("y-wall(y>16)")
                    if x > 16: ctx.append("x-floor")
                    if x < 3: ctx.append("x-top")
                    if bool(near_nb[i]): ctx.append("neighbor<3")
                    if not ctx: ctx.append("free")
                    for c in ctx:
                        res[c].append((float(dv[i, 1]), float(dv[i, 0])))
    print("context: n | mean d_vy | mean |d_vy| | mean d_vx | mean |d_vx|  (model - truth, one step)")
    for c, r in sorted(res.items()):
        r = torch.tensor(r)
        print(f"  {c:14s} {len(r):5d} | {r[:,0].mean():7.3f} | {r[:,0].abs().mean():6.3f} | {r[:,1].mean():7.3f} | {r[:,1].abs().mean():6.3f}")


if __name__ == "__main__":
    main()
