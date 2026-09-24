"""Throwaway: print the free-rollout delta head's output bias."""
import argparse

from scripts.eval_free_rollout import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
args = ap.parse_args()
model = load_model(args.checkpoint, "free", 20, 32, 4.0)
for name, p in model.dynamics.named_parameters():
    if "delta" in name or "head" in name:
        print(name, tuple(p.shape), p.detach().flatten()[:8].tolist() if p.numel() <= 8 else "")
