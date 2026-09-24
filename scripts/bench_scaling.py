"""Times the real-time sim step (Sim.step, no spawning) for N balls, one
row per N: total wall time for --ticks steps, median ms per tick, peak
memory. Grid side is 10000 up to 1M balls, then 10*sqrt(N) (density 0.01).
Above --tile-balls the step runs through scripts/tiled_sim.py (host-resident
state streamed through the GPU strip by strip).

Usage:
    PYTHONPATH=. python3 scripts/bench_scaling.py --device cuda --out results/bench_scaling_cuda.json
"""
import argparse
import json
import math
import resource
import time

import torch

from scripts.eval_free_rollout import load_model
from scripts.realtime_sim import Sim
from scripts.tiled_sim import TiledSim

SIZES = [10 ** e * m for e in range(1, 7) for m in (1, 5)][:-1]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def make_sim(model, count, grid, device, seed):
    sim = Sim(model, count, 1, 2.3, seed)
    gen = torch.Generator().manual_seed(seed)
    sim.positions = (torch.rand(count, 2, generator=gen) * (grid - 3) + 1.5).to(device)
    sim.velocities = ((torch.rand(count, 2, generator=gen) - 0.5) * 4.6).to(device)
    sim.hidden = torch.zeros(count, model.dynamics.hidden_dim, device=device)
    return sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/token_model_soup_b.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--grid", type=int, default=0, help="0 = max(10000, 10*sqrt(N))")
    ap.add_argument("--tile-balls", type=int, default=10_000_000)
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--sizes", type=str, default=None)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else SIZES
    rows = []
    for count in sizes:
        grid = args.grid or max(10000, math.ceil(10 * math.sqrt(count)))
        model = load_model(args.checkpoint, "free", grid, 32, 4.0, False, True, True, True, True, True, True).to(device)
        tiled = count > args.tile_balls
        if tiled:
            strips = math.ceil(count / args.tile_balls)
            sim = TiledSim(model, strips, device, slack=1.3)
            warm = make_sim(model, 1000, grid, device, args.seed)
            for _ in range(3):
                warm.step(None)
            sim.init_random(count, args.seed)
        else:
            strips = 1
            warm = make_sim(model, min(count, 1000), args.grid or grid, device, args.seed)
            for _ in range(3):
                warm.step(None)
            sim = make_sim(model, count, grid, device, args.seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        ticks = []
        sync(device)
        start = time.perf_counter()
        for i in range(args.ticks):
            t = time.perf_counter()
            sim.step() if tiled else sim.step(None)
            sync(device)
            ticks.append((time.perf_counter() - t) * 1000)
            if tiled and i % 10 == 9:
                print(f"  N={count} tick {i + 1} {ticks[-1] / 1000:.1f} s, elapsed {time.perf_counter() - start:.0f} s", flush=True)
        total = time.perf_counter() - start
        alive = sim.alive() if tiled else int(sim.positions.shape[0])
        ticks.sort()
        row = {"balls": count, "grid": grid, "ticks": args.ticks, "device": args.device, "tiled": tiled, "strips": strips,
               "total_s": total, "median_ms": ticks[len(ticks) // 2], "p95_ms": ticks[int(len(ticks) * 0.95)],
               "alive": alive,
               "peak_mem_mb": (torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda"
                               else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
               "host_state_gb": count * 4 * (4 + 32) / 2 ** 30 if tiled else 0.0}
        rows.append(row)
        print(json.dumps(row), flush=True)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(rows, f, indent=2)
        del sim, warm, model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
