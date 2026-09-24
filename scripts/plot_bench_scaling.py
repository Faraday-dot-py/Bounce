"""Plots results/bench_scaling_{cuda,big}.json merged into one curve: total
time for the tick run, per-ball cost and peak GPU memory vs ball count, with
an O(N) reference line. Tiled points (host-of-strips path) are drawn as
squares.

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
    ap.add_argument("--inputs", nargs="+", default=["results/bench_scaling_cuda.json", "results/bench_scaling_big.json"])
    ap.add_argument("--out", default="results/bench_scaling.png")
    args = ap.parse_args()

    rows = []
    for path in args.inputs:
        if os.path.exists(path):
            rows += json.load(open(path))
    rows.sort(key=lambda r: r["balls"])
    n = [r["balls"] for r in rows]
    tiled = [r.get("tiled", False) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    series = [
        ([r["total_s"] for r in rows], "seconds", f"Total time for {rows[0]['ticks']} ticks"),
        ([r["median_ms"] * 1000 / r["balls"] for r in rows], "microseconds / ball", "Median cost per ball per tick"),
        ([r["peak_mem_mb"] / 1024 for r in rows], "GB", "Peak GPU memory allocated"),
    ]
    for ax, (y, label, title) in zip(axes, series):
        ax.plot(n, y, "-", color="C0")
        ax.plot([a for a, t in zip(n, tiled) if not t], [b for b, t in zip(y, tiled) if not t], "o", color="C0",
                label="single pass")
        ax.plot([a for a, t in zip(n, tiled) if t], [b for b, t in zip(y, tiled) if t], "s", color="C1", label="tiled")
        ax.set_title(title)
        ax.set_ylabel(label)
        ax.set_xlabel("balls")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
    for x, r in zip(n, rows):
        axes[0].annotate(f"{r['total_s']:.3g}s", (x, r["total_s"]), textcoords="offset points", xytext=(0, 6),
                         ha="center", fontsize=7)
    anchor = rows[-1]["total_s"] / rows[-1]["balls"]
    axes[0].plot([n[0], n[-1]], [anchor * n[0], anchor * n[-1]], "k--", lw=0.8, label="O(N) through last point")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle("Real-time sim step scaling on one H200 (grid 10000^2 up to 1M balls, then 10*sqrt(N)^2 = density 0.01)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
