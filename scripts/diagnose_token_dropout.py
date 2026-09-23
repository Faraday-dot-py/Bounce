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
        --num-seeds 16 --num-steps 20 --trace
"""
import argparse
import random

import numpy as np
import torch

import bounce
from model.dataset import make_scenario_uniform
from model.token_model import TokenModel
from model.token_match import match_tokens_to_state
from model.token_gate import occluding_mask


def load_model(checkpoint_path, n, hidden_dim, neighbor_radius, velocity_weight=0.0):
    model = TokenModel(n=n, radius=0.75, dt=0.15, hidden_dim=hidden_dim, neighbor_radius=neighbor_radius,
                        velocity_weight=velocity_weight)
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


def window_total_at(prob, position, radius, margin, max_expansions=3):
    """Mirrors `centroid_near`'s own widened-search loop (base window plus
    up to `max_expansions` growing retries) and returns the mass sum of
    whichever window it would settle on -- 0.0 only if EVERY widened
    attempt comes up empty, i.e. the real bailout condition post-fix.
    Reporting just the base window's mass (the pre-widening behavior this
    function used to have) mislabeled steps the real widened
    `centroid_near` call recovers from as BAILOUT; see
    docs/debugging/experiment-log.md."""
    import math
    n = prob.shape[0]
    step = max(1, int(math.ceil(radius)))
    base_half = int(np.ceil(radius + margin))
    cx = int(round(float(position[0])))
    cy = int(round(float(position[1])))
    for expansion in range(max_expansions + 1):
        half = base_half + expansion * step
        i_lo_raw, i_hi_raw = cx - half, cx + half
        j_lo_raw, j_hi_raw = cy - half, cy + half
        if i_hi_raw < 0 or i_lo_raw > n - 1 or j_hi_raw < 0 or j_lo_raw > n - 1:
            continue
        i_lo, i_hi = max(0, i_lo_raw), min(n - 1, i_hi_raw)
        j_lo, j_hi = max(0, j_lo_raw), min(n - 1, j_hi_raw)
        total = float(prob[i_lo:i_hi + 1, j_lo:j_hi + 1].sum())
        if total > 1e-6:
            return total
    return 0.0


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
    ap.add_argument("--velocity-weight", type=float, default=0.0)
    ap.add_argument("--dropout-threshold", type=float, default=0.05)
    ap.add_argument("--trace", action="store_true",
                     help="print per-step window_total/occlusion for each dropped token")
    args = ap.parse_args()

    model = load_model(args.checkpoint, args.n, args.hidden_dim, args.neighbor_radius, args.velocity_weight)

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

        num_tokens = positions.shape[0]
        trace = {t: [] for t in range(num_tokens)}  # (step, window_total, occluding, gt_err, nearest_ball, nearest_err)
        observed = g1
        prev_obs_pos = positions
        with torch.no_grad():
            for step in range(args.num_steps - 1):
                occ = occluding_mask(positions, model._gate_radius())
                gt_state = states[step + 1]
                gt_pos = torch.stack([
                    torch.tensor(gt_state["x"], dtype=torch.float32),
                    torch.tensor(gt_state["y"], dtype=torch.float32),
                ], dim=1)
                for t in range(num_tokens):
                    wt = window_total_at(observed[0], positions[t], model.radius, model.detect_margin)
                    ball = int(token_to_ball[t])
                    gt_err = float(torch.norm(positions[t] - gt_pos[ball]))
                    all_err = torch.norm(gt_pos - positions[t].unsqueeze(0), dim=1)
                    nearest_ball = int(all_err.argmin())
                    nearest_err = float(all_err[nearest_ball])
                    trace[t].append((step, wt, bool(occ[t]), gt_err, nearest_ball, nearest_err))
                positions, velocities, hidden, pred_grid, prev_obs_pos = model.step(
                    positions, velocities, hidden, observed, prev_obs_pos
                )
                observed = pred_grid
        last_peak = [peak_prob_at(observed[0], positions[t]) for t in range(positions.shape[0])]

        peak_order = sorted(range(len(episode_peaks)), key=lambda t: episode_peaks[t])
        nn_order = sorted(range(len(episode_nn)), key=lambda t: episode_nn[t])
        for t in range(positions.shape[0]):
            ball = int(token_to_ball[t])
            if last_peak[t] < args.dropout_threshold:
                dropout_ball_idx.append((seed, ball))
                rank_at_dropout.append((peak_order.index(t), nn_order.index(t), positions.shape[0]))
                if args.trace:
                    print(f"\n--- trace: seed {seed}, ball {ball} (token {t}), frame1 peak={episode_peaks[t]:.4f} ---")
                    for step, wt, occ, gt_err, nearest_ball, nearest_err in trace[t]:
                        bail = "BAILOUT" if wt <= 1e-6 else ""
                        occ_str = "occluded" if occ else ""
                        swap = f"SWAP->ball{nearest_ball}(err={nearest_err:.2f})" if nearest_ball != ball and nearest_err < gt_err else ""
                        print(f"  step {step:2d}: window_total={wt:.4f} gt_err={gt_err:.3f} {occ_str} {bail} {swap}")

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
