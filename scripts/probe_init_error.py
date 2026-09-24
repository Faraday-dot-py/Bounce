"""Spike: error of TokenModel.init_tokens at frame 1 against ground truth,
to compare with the perturbation scales in probe_predictability.py. Throwaway.

Usage:
    PYTHONPATH=. python3 scripts/probe_init_error.py --checkpoint checkpoints/token_model_h24_v18.pt
"""
import argparse

import torch

from model.token_match import match_tokens_to_state
from scripts.diagnose_token_dropout import simulate
from scripts.eval_free_rollout import load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num-seeds", type=int, default=48)
    ap.add_argument("--base-seed", type=int, default=4738)
    args = ap.parse_args()
    model = load_model(args.checkpoint, "free", 20, 32, 4.0)
    pos_err, vel_err, speed = [], [], []
    for seed in range(args.base_seed, args.base_seed + args.num_seeds):
        frames, states = simulate(20, 4, seed, 3)
        g0 = torch.from_numpy(frames[0].transpose(2, 0, 1))
        g1 = torch.from_numpy(frames[1].transpose(2, 0, 1))
        with torch.no_grad():
            pos, vel, _ = model.init_tokens(g0, g1)
        if pos.shape[0] != 4:
            continue
        s = [{k: torch.tensor(v, dtype=torch.float32) for k, v in st.items()} for st in states]
        m = match_tokens_to_state(pos, s[1])
        gt = torch.stack([s[1]["x"], s[1]["y"]], dim=-1)[m]
        gt_v = (torch.stack([s[1]["x"], s[1]["y"]], dim=-1) - torch.stack([s[0]["x"], s[0]["y"]], dim=-1))[m] / 0.15
        pos_err.append(torch.norm(pos - gt, dim=-1).mean())
        speed.append(torch.norm(gt_v, dim=-1).mean())
        vel_err.append(torch.norm(vel - gt_v, dim=-1).mean())
    print(f"seeds used {len(pos_err)}; mean pos err {torch.stack(pos_err).mean():.4f}; mean vel err (vs backward diff) {torch.stack(vel_err).mean():.4f}; mean gt speed {torch.stack(speed).mean():.4f}")


if __name__ == "__main__":
    main()
