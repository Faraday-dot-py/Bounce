"""Instruments a TokenModel rollout to find WHY a ball's token drops out
during self-feed, instead of just observing that it does (see job 2852,
docs/debugging/experiment-log.md): per token, per step, records its own
rasterized PROB peak, whether the occlusion gate suppressed its
observation correction, and whether centroid_near bailed out (no mass in
its window). Tokens are matched to ground-truth ball identity once at
frame 1 (match_tokens_to_state) so "ball 2" means the same physical ball
across every seed.

Usage:
    PYTHONPATH=. python3 scripts/diagnose_token_dropout.py \
        --checkpoint checkpoints/token_model_h12_v9.pt \
        --num-seeds 16 --num-steps 20
"""
import argparse
import random

import numpy as np
import torch

import bounce
from model.dataset import make_scenario_uniform
from model.token_model import TokenModel
from model.token_match import match_tokens_to_state


def load_model(checkpoint_path, n, hidden_dim, neighbor_radius):
    model = TokenModel(n=n, radius=0.75, dt=0.15, hidden_dim=hidden_dim, neighbor_radius=neighbor_radius)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def simulate(n, num_balls, seed, num_steps, dt=0.15, gravity=9.0, radius=0.75,
             stiffness=400.0, substeps=8, vy=2.3):
    rng = random.Random(seed)
    balls = make_scenario_uniform(num_balls, n, vy, rng)
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    frames = [np.array(G, dtype=np.float32)]
    states = [{"x": [b["x"] for b in balls], "y": [b["y"] for b in balls]}]
    for _ in range(num_steps):
        bounce.step(G, n, balls, dt, gravity, radius, stiffness, substeps)
        frames.append(np.array(G, dtype=np.float32))
        states.append({"x": [b["x"] for b in balls], "y": [b["y"] for b in balls]})
    return frames, states


def peak_prob_at(prob, position, radius=0.75, margin=1.0):
    """Own-position PROB reading without the fallback-to-position-if-empty
    behavior of centroid_near -- just the raw mass in the window, so a
    token whose window has gone empty shows up as 0, not silently hidden."""
    n = prob.shape[0]
    half = int(np.ceil(radius + margin))
    cx = int(round(float(position[0])))
    cy = int(round(float(position[1])))
    i_lo, i_hi = max(0, cx - half), min(n - 1, cx + half)
    j_lo, j_hi = max(0, cy - half), min(n - 1, cy + half)
    if i_lo > i_hi or j_lo > j_hi:
        return 0.0
    return float(prob[i_lo:i_hi + 1, j_lo:j_hi + 1].max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--num-balls", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--num-seeds", type=int, default=16)
    ap.add_argument("--base-seed", type=int, default=4738)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--neighbor-radius", type=float, default=4.0)
    ap.add_argument("--dropout-threshold", type=float, default=0.05)
    args = ap.parse_args()

    model = load_model(args.checkpoint, args.n, args.hidden_dim, args.neighbor_radius)

    dropout_ball_idx = []
    frame1_peak_by_ball = {}
    frame1_nn_dist_by_ball = {}
    rank_at_dropout = []  # (peak_rank, nn_rank) of the dropped token within its own episode, 0=lowest

    for seed in range(args.base_seed, args.base_seed + args.num_seeds):
        frames, states = simulate(args.n, args.num_balls, seed, args.num_steps)
        g0 = torch.from_numpy(frames[0].transpose(2, 0, 1))
        g1 = torch.from_numpy(frames[1].transpose(2, 0, 1))

        with torch.no_grad():
            positions, velocities, hidden = model.init_tokens(g0, g1)

        state1 = {k: torch.tensor(v, dtype=torch.float32) for k, v in states[1].items()}
        token_to_ball = match_tokens_to_state(positions, state1)

        # frame-1 baseline: this token's own rasterized peak (ground truth,
        # not yet self-fed) and distance to its nearest sibling.
        episode_peaks = [peak_prob_at(g1[0], positions[t]) for t in range(positions.shape[0])]
        for t in range(positions.shape[0]):
            ball = int(token_to_ball[t])
            frame1_peak_by_ball.setdefault(ball, []).append(episode_peaks[t])
        episode_nn = [None] * positions.shape[0]
        if positions.shape[0] > 1:
            dists = torch.cdist(positions, positions) + torch.eye(positions.shape[0]) * 1e6
            nn = dists.min(dim=1).values
            episode_nn = nn.tolist()
            for t in range(positions.shape[0]):
                ball = int(token_to_ball[t])
                frame1_nn_dist_by_ball.setdefault(ball, []).append(episode_nn[t])

        observed = g1
        with torch.no_grad():
            for step in range(args.num_steps - 1):
                positions, velocities, hidden, pred_grid = model.step(positions, velocities, hidden, observed)
                observed = pred_grid
        last_peak = [peak_prob_at(observed[0], positions[t]) for t in range(positions.shape[0])]

        peak_order = sorted(range(len(episode_peaks)), key=lambda t: episode_peaks[t])
        nn_order = sorted(range(len(episode_nn)), key=lambda t: episode_nn[t])
        for t in range(positions.shape[0]):
            ball = int(token_to_ball[t])
            if last_peak[t] < args.dropout_threshold:
                dropout_ball_idx.append((seed, ball))
                rank_at_dropout.append((peak_order.index(t), nn_order.index(t), positions.shape[0]))

    print(f"=== dropout events (peak < {args.dropout_threshold} at step {args.num_steps - 1}) ===")
    for seed, ball in dropout_ball_idx:
        print(f"  seed {seed}: ball {ball} dropped out")
    from collections import Counter
    counts = Counter(b for _, b in dropout_ball_idx)
    print(f"\ndropout count by ball index: {dict(counts)}")

    print("\n=== frame-1 peak PROB by ball index (mean, min, max, n) ===")
    for ball in sorted(frame1_peak_by_ball):
        vals = frame1_peak_by_ball[ball]
        print(f"  ball {ball}: mean={np.mean(vals):.4f} min={np.min(vals):.4f} max={np.max(vals):.4f} n={len(vals)}")

    print("\n=== frame-1 nearest-neighbor distance by ball index (mean, min, n) ===")
    for ball in sorted(frame1_nn_dist_by_ball):
        vals = frame1_nn_dist_by_ball[ball]
        print(f"  ball {ball}: mean={np.mean(vals):.4f} min={np.min(vals):.4f} n={len(vals)}")

    print("\n=== within-episode rank of the dropped token (0 = lowest peak / closest neighbor) ===")
    for peak_rank, nn_rank, total in rank_at_dropout:
        print(f"  peak_rank={peak_rank}/{total - 1}  nn_rank={nn_rank}/{total - 1}")


if __name__ == "__main__":
    main()
