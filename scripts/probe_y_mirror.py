"""Spike: y-reflection equivariance of the free-rollout model. The simulator
is exactly symmetric under y -> 19 - y, vy -> -vy (gravity is along x). For
teacher-forced states, the model's predicted vy on the mirrored state should
be the negative of its prediction on the original. Reports the violation
(pred_vy(s) + pred_vy(mirror(s))) / 2, which is the model's own y-bias, by
context. Throwaway.

Usage:
    PYTHONPATH=. python3 scripts/probe_y_mirror.py --checkpoint checkpoints/token_model_h24_v18.pt
"""
import argparse
from collections import defaultdict

import torch

from scripts.diagnose_token_dropout import simulate
from scripts.eval_free_rollout import load_model


def mirror(p, v):
    return torch.stack([p[:, 0], 19.0 - p[:, 1]], dim=1), torch.stack([v[:, 0], -v[:, 1]], dim=1)


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
            h_m = h.clone()
            for t in range(1, args.num_steps):
                p = gt[t]; v = gt_v[t - 1]
                pm, vm = mirror(p, v)
                _, nv, h, _ = model.step_free(p, v, h)
                _, nvm, h_m, _ = model.step_free(pm, vm, h_m)
                viol = (nv[:, 1] + nvm[:, 1]) / 2
                dist = torch.cdist(p, p) + torch.eye(4) * 99
                near = dist.min(dim=1).values < 3.0
                for i in range(4):
                    y, x = float(p[i, 1]), float(p[i, 0])
                    ctx = "y-wall" if (y < 3 or y > 16) else ("neighbor<3" if bool(near[i]) else "free")
                    res[ctx].append(float(viol[i]))
                    res["all"].append(float(viol[i]))
    print("context: n | mean y-bias (pred vy(s)+vy(mirror))/2 | mean |.|")
    for c, r in sorted(res.items()):
        r = torch.tensor(r)
        print(f"  {c:12s} {len(r):5d} | {r.mean():7.3f} | {r.abs().mean():6.3f}")


if __name__ == "__main__":
    main()
