"""Averages the parameters of several same-architecture state_dict checkpoints
(a "model soup") and saves the result.

Usage:
    python3 scripts/make_checkpoint_soup.py --out checkpoints/soup.pt ckpt_a.pt ckpt_b.pt ...
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("checkpoints", nargs="+")
    args = ap.parse_args()
    states = [torch.load(p, map_location="cpu") for p in args.checkpoints]
    states = [s["model"] if "model" in s else s for s in states]
    soup = {k: sum(s[k].float() for s in states) / len(states) for k in states[0]}
    torch.save(soup, args.out)
    print(f"averaged {len(states)} checkpoints -> {args.out}")


if __name__ == "__main__":
    main()
