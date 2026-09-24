"""Plots results/bench_scaling_{cuda,cpu}.json: total time for the tick run
and per-ball cost vs ball count, with an O(N) reference line.

Usage:
    PYTHONPATH=. python3 scripts/plot_bench_scaling.py --out results/bench_scaling.png
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=["results/bench_scaling_cuda.json", "results/bench_scaling_cpu.json"])
    ap.add_argument("--out", default="results/bench_scaling.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for path in args.inputs:
        if not os.path.exists(path):
            continue
        rows = json.load(open(path))
        label = rows[0]["device"]
        n = [r["balls"] for r in rows]
        axes[0].plot(n, [r["total_s"] for r in rows], "o-", label=label)
        axes[1].plot(n, [r["median_ms"] * 1000 / r["balls"] for r in rows], "o-", label=label)
        axes[2].plot(n, [r["peak_mem_mb"] for r in rows], "o-", label=label)
        for x, r in zip(n, rows):
            axes[0].annotate(f"{r['total_s']:.2g}s", (x, r["total_s"]), textcoords="offset points", xytext=(0, 6),
                             ha="center", fontsize=7)
    ref = [10, 1_000_000]
    first = json.load(open([p for p in args.inputs if os.path.exists(p)][0]))
    anchor = first[-1]["total_s"] / first[-1]["balls"]
    axes[0].plot(ref, [anchor * x for x in ref], "k--", lw=0.8, label="O(N) through 1M point")
    axes[0].set_title(f"Total time for {first[0]['ticks']} ticks (grid {first[0]['grid']}^2)")
    axes[0].set_ylabel("seconds")
    axes[1].set_title("Median cost per ball per tick")
    axes[1].set_ylabel("microseconds / ball")
    axes[2].set_title("Peak memory (GPU alloc / CPU RSS)")
    axes[2].set_ylabel("MB")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("balls")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
