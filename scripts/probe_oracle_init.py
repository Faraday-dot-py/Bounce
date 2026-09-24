"""Spike: v18 free rollout started from ground-truth frame-1 positions and
backward-difference velocities instead of detected ones, to split init-state
error from dynamics error. Throwaway.

Usage:
    PYTHONPATH=. python3 scripts/probe_oracle_init.py --checkpoint checkpoints/token_model_h24_v18.pt
"""
import argparse
import json

import torch

from scripts.diagnose_token_dropout import simulate
from scripts.eval_free_rollout import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-seeds", type=int, default=48)
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--base-seed", type=int, default=4738)
    ap.add_argument("--report-steps", type=str, default="5,10,20,50,100")
    args = ap.parse_args()
    report_steps = [int(s) for s in args.report_steps.split(",")]
    model = load_model(args.checkpoint, "free", 20, 32, 4.0)
    errs = {"oracle": [], "detected": []}
    for seed in range(args.base_seed, args.base_seed + args.num_seeds):
        frames, states = simulate(20, 4, seed, args.num_steps)
        gt = torch.stack([torch.tensor([[x, y] for x, y in zip(s["x"], s["y"])], dtype=torch.float32) for s in states[1:args.num_steps + 1]])
        g0 = torch.from_numpy(frames[0].transpose(2, 0, 1))
        g1 = torch.from_numpy(frames[1].transpose(2, 0, 1))
        with torch.no_grad():
            dpos, dvel, dh = model.init_tokens(g0, g1)
            if dpos.shape[0] != 4:
                continue
            p0 = torch.tensor([[x, y] for x, y in zip(states[0]["x"], states[0]["y"])], dtype=torch.float32)
            opos = gt[0]
            ovel = (opos - p0) / 0.15
            for name, (p, v, h) in {"oracle": (opos, ovel, torch.zeros_like(dh)), "detected": (dpos, dvel, dh)}.items():
                traj = [p]
                for _ in range(args.num_steps - 1):
                    p, v, h, _ = model.step_free(p, v, h)
                    traj.append(p)
                traj = torch.stack(traj)
                d = torch.cdist(traj[0], gt[0]).argmin(dim=1)
                errs[name].append(torch.norm(traj - gt[:, d], dim=-1).mean(dim=1))
    out = {name: {str(k): float(torch.stack(e).mean(dim=0)[k - 1]) for k in report_steps} for name, e in errs.items()}
    out["seeds_used"] = len(errs["oracle"])
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
