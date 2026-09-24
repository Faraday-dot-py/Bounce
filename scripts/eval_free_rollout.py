"""Scores a token-model checkpoint's open-loop rollout against the
simulator: mean position error vs step, identity swaps, token-count
mismatches, and the legacy own-PROB-peak dropout fraction. Same seeds and
scenario generator as scripts/diagnose_token_dropout.py, so v9 (with its
normal observation loop), v9 with observation off, v17 and the
free-rollout model are directly comparable -- see
docs/superpowers/specs/2026-09-23-token-free-rollout-design.md.

Usage:
    PYTHONPATH=. python3 scripts/eval_free_rollout.py \
        --checkpoint checkpoints/token_model_h24_v18.pt --mode free \
        --num-seeds 48 --num-steps 300 --report-steps 20,100,300 --out results/eval_v18.json
"""
import argparse
import json

import torch

from model.token_match import match_tokens_to_state
from model.token_model import TokenModel
from scripts.diagnose_token_dropout import peak_prob_at, simulate


def position_errors(pred_positions, token_to_ball, gt_x, gt_y):
    """(T,) mean L2 error per step between each token and the ground-truth
    ball it was matched to at frame 1. pred_positions is (T, N, 2); gt_x
    and gt_y are (T, B)."""
    T = pred_positions.shape[0]
    if pred_positions.shape[1] == 0:
        return torch.zeros(T)
    gt = torch.stack([gt_x, gt_y], dim=-1)[:, token_to_ball]
    return torch.norm(pred_positions - gt, dim=-1).mean(dim=1)


def count_identity_swaps(pred_positions, gt_x, gt_y, token_to_ball):
    """Tokens whose nearest ground-truth ball at the final step is not the
    ball they were matched to at frame 1."""
    if pred_positions.shape[1] == 0:
        return 0
    final_gt = torch.stack([gt_x[-1], gt_y[-1]], dim=-1)
    nearest = torch.cdist(pred_positions[-1], final_gt).argmin(dim=1)
    return int((nearest != token_to_ball).sum())


def load_model(checkpoint, mode, n, hidden_dim, neighbor_radius, mirror_sym=False, velocity_readout=False,
               wall_lookahead=False, wall_head=False, pair_impulse=False,
               position_refine=False, ball_split=False):
    model = TokenModel(n=n, radius=0.75, dt=0.15, hidden_dim=hidden_dim, neighbor_radius=neighbor_radius,
                        free_rollout=mode == "free", track_query=mode == "v17", mirror_sym=mirror_sym, velocity_readout=velocity_readout,
                        wall_lookahead=wall_lookahead, wall_head=wall_head, pair_impulse=pair_impulse,
                        position_refine=position_refine, ball_split=ball_split,
                        observation_weight=0.0 if mode == "v9-noobs" else 0.5)
    state = torch.load(checkpoint, map_location="cpu")
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    return model


def rollout(model, mode, frames, num_steps):
    """Returns (positions (T, N, 2) for frames 1..num_steps, final rendered grid)."""
    g0 = torch.from_numpy(frames[0].transpose(2, 0, 1))
    g1 = torch.from_numpy(frames[1].transpose(2, 0, 1))
    with torch.no_grad():
        positions, velocities, hidden = model.init_tokens(g0, g1)
        trajectory = [positions]
        observed = g1
        prev_obs_pos = positions
        for _ in range(num_steps - 1):
            if mode == "free":
                positions, velocities, hidden, observed = model.step_free(positions, velocities, hidden)
            else:
                positions, velocities, hidden, observed, prev_obs_pos = model.step(
                    positions, velocities, hidden, observed, prev_obs_pos
                )
            trajectory.append(positions)
    return torch.stack(trajectory), observed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", choices=["free", "v9", "v9-noobs", "v17"], required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--num-balls", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--num-seeds", type=int, default=48)
    ap.add_argument("--base-seed", type=int, default=4738)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--mirror-sym", action="store_true")
    ap.add_argument("--velocity-readout", action="store_true")
    ap.add_argument("--wall-lookahead", action="store_true")
    ap.add_argument("--wall-head", action="store_true")
    ap.add_argument("--pair-impulse", action="store_true")
    ap.add_argument("--position-refine", action="store_true")
    ap.add_argument("--ball-split", action="store_true")
    ap.add_argument("--neighbor-radius", type=float, default=4.0)
    ap.add_argument("--report-steps", type=str, default="5,10,20")
    ap.add_argument("--dropout-threshold", type=float, default=0.05)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    model = load_model(args.checkpoint, args.mode, args.n, args.hidden_dim, args.neighbor_radius, args.mirror_sym, args.velocity_readout,
                       args.wall_lookahead, args.wall_head, args.pair_impulse,
                       args.position_refine, args.ball_split)
    report_steps = [int(s) for s in args.report_steps.split(",")]

    errors = []
    swaps = 0
    mismatch_seeds = []
    tokens_total = 0
    tokens_faded = 0
    for seed in range(args.base_seed, args.base_seed + args.num_seeds):
        frames, states = simulate(args.n, args.num_balls, seed, args.num_steps)
        trajectory, final_grid = rollout(model, args.mode, frames, args.num_steps)
        gt_x = torch.tensor([s["x"] for s in states[1:args.num_steps + 1]], dtype=torch.float32)
        gt_y = torch.tensor([s["y"] for s in states[1:args.num_steps + 1]], dtype=torch.float32)
        state1 = {"x": gt_x[0], "y": gt_y[0]}
        token_to_ball = match_tokens_to_state(trajectory[0], state1)
        if trajectory.shape[1] != args.num_balls:
            mismatch_seeds.append(seed)
        errors.append(position_errors(trajectory, token_to_ball, gt_x, gt_y))
        swaps += count_identity_swaps(trajectory, gt_x, gt_y, token_to_ball)
        for t in range(trajectory.shape[1]):
            tokens_total += 1
            if peak_prob_at(final_grid[0], trajectory[-1][t]) < args.dropout_threshold:
                tokens_faded += 1

    mean_err = torch.stack(errors).mean(dim=0)
    result = {
        "checkpoint": args.checkpoint, "mode": args.mode, "n": args.n, "num_balls": args.num_balls,
        "num_seeds": args.num_seeds, "num_steps": args.num_steps,
        "mean_position_error": {str(k): float(mean_err[min(k, len(mean_err) - 1)]) for k in report_steps},
        "identity_swaps": swaps, "token_count_mismatch_seeds": mismatch_seeds,
        "faded_token_fraction": tokens_faded / max(tokens_total, 1),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
