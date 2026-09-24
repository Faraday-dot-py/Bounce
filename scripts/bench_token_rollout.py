"""Times TokenModel init and free-rollout steps on CPU.

Usage:
    PYTHONPATH=. python3 scripts/bench_token_rollout.py \
        --checkpoint checkpoints/token_model_soup_b.pt --n 100 --num-balls 25
"""
import argparse
import time

import torch

from model.token_rasterize import rasterize_tokens
from scripts.diagnose_token_dropout import simulate
from scripts.eval_free_rollout import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--num-balls", type=int, default=25)
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=4738)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    model = load_model(args.checkpoint, "free", args.n, 32, 4.0, False, True, True, True, True, True, True)
    frames, _ = simulate(args.n, args.num_balls, args.seed, 4)
    g0 = torch.from_numpy(frames[0].transpose(2, 0, 1))
    g1 = torch.from_numpy(frames[1].transpose(2, 0, 1))

    with torch.no_grad():
        t = time.perf_counter()
        positions, velocities, hidden = model.init_tokens(g0, g1)
        init_ms = (time.perf_counter() - t) * 1000
        tokens = positions.shape[0]

        for _ in range(3):
            model.step_free(positions, velocities, hidden)

        dyn, ras = [], []
        for _ in range(args.num_steps):
            t0 = time.perf_counter()
            dp, dv, nh = model.dynamics(positions, velocities, hidden)
            t1 = time.perf_counter()
            fp = positions + velocities * model.dt + dp
            fv = velocities + dv
            rasterize_tokens(fp, fv, model.n, model.radius)
            t2 = time.perf_counter()
            dyn.append((t1 - t0) * 1000)
            ras.append((t2 - t1) * 1000)
            positions, velocities, hidden, _ = model.step_free(positions, velocities, hidden)

    def med(x):
        return sorted(x)[len(x) // 2]

    print(f"n={args.n} balls={args.num_balls} tokens={tokens} threads={torch.get_num_threads()}")
    print(f"init_tokens {init_ms:.1f} ms (once)")
    print(f"dynamics {med(dyn):.2f} ms, rasterize {med(ras):.2f} ms per step (median of {args.num_steps})")


if __name__ == "__main__":
    main()
