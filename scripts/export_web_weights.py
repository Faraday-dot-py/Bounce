"""Export a free-rollout token-model checkpoint to web/weights.bin (fp32
little-endian, tensors concatenated) + web/weights.json (manifest + config).

Usage:
    PYTHONPATH=. python3 scripts/export_web_weights.py --checkpoint checkpoints/token_model_soup_b.pt
"""
import argparse
import json

import numpy as np
import torch

CONFIG = {"hidden": 32, "neighbor_radius": 4.0, "wall_range": 3.0, "radius": 0.75, "dt": 0.15,
          "wall_lookahead": True, "wall_head": True, "pair_impulse": True, "gravity": 9.0,
          "n": 100, "max_speed": 30.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/token_model_soup_b.pt")
    ap.add_argument("--out", default="web/weights")
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    state = state.get("model", state)
    tensors, chunks, offset = {}, [], 0
    for name, t in state.items():
        a = t.detach().numpy().astype("<f4")
        tensors[name.removeprefix("dynamics.")] = {"shape": list(a.shape), "offset": offset}
        chunks.append(a.reshape(-1))
        offset += a.size
    np.concatenate(chunks).tofile(args.out + ".bin")
    with open(args.out + ".json", "w") as f:
        json.dump({"checkpoint": args.checkpoint.split("/")[-1], "config": CONFIG, "tensors": tensors}, f, indent=1)
    print(f"{len(tensors)} tensors, {offset} floats -> {args.out}.bin")


if __name__ == "__main__":
    main()
