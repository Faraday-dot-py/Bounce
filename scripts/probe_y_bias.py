"""Spike: does the free-rollout model have a directional bias in y velocity
(the non-gravity axis)? (a) per-seed mean vy of tokens vs step for model and
truth; (b) an isolated interior token at rest / with vy=+-2: delta_vel and
delta_pos per step from step_free. Throwaway.

Usage:
    PYTHONPATH=. python3 scripts/probe_y_bias.py --checkpoint checkpoints/token_model_h24_v18.pt
"""
import argparse

import torch

from scripts.diagnose_token_dropout import simulate
from scripts.eval_free_rollout import load_model


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
    ap.add_argument("--num-steps", type=int, default=30)
    ap.add_argument("--base-seed", type=int, default=4738)
    args = ap.parse_args()
    model = load_model(args.checkpoint, "free", 20, 32, 4.0, args.mirror_sym, args.velocity_readout, args.wall_lookahead, args.wall_head, args.pair_impulse, args.position_refine)

    mv, tv = [], []
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
        mv.append(torch.stack(vs)[..., 1].mean(dim=1))
        tv.append(gt_v[:args.num_steps, :, 1].mean(dim=1))
    mv, tv = torch.stack(mv), torch.stack(tv)
    print("mean vy (y is non-gravity axis); model, truth, model frac seeds <0 ; seeds", len(mv))
    for k in [1, 2, 3, 5, 8, 10, 15, 20, 30]:
        print(f"  step {k:2d}: model {mv[:, k-1].mean():7.3f}  truth {tv[:, k-1].mean():7.3f}  frac<0 {float((mv[:, k-1] < 0).float().mean()):.2f}")
    print("model mean vy over steps 3-20 per seed: mean", float(mv[:, 2:20].mean()), "truth", float(tv[:, 2:20].mean()))

    print("isolated interior token, rest of world empty:")
    for x0, y0, vx0, vy0 in [(10., 10., 0., 0.), (10., 10., 0., 2.), (10., 10., 0., -2.), (10., 10., 2., 0.)]:
        p = torch.tensor([[x0, y0]]); v = torch.tensor([[vx0, vy0]]); h = torch.zeros(1, 32)
        with torch.no_grad():
            out = []
            for i in range(6):
                p, v, h, _ = model.step_free(p, v, h)
                out.append((float(p[0, 0]), float(p[0, 1]), float(v[0, 0]), float(v[0, 1])))
        print(f"  start x{x0} y{y0} vx{vx0} vy{vy0}: after 1/3/6 steps (x,y,vx,vy) =", [tuple(round(a, 2) for a in out[i]) for i in (0, 2, 5)])


if __name__ == "__main__":
    main()
