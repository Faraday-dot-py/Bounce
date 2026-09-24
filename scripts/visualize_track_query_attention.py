"""Dumps per-step, per-token self-/cross-attention weights for a
track-query TokenModel rollout, aligned against diagnose_token_dropout.py's
per-step trace output (window total, occlusion state, ground-truth
error). Occlusion/window-total are gate-and-mask concepts that don't
apply to this architecture, so this script reports what DOES apply
here: which neighbors a token's self-attention weighted most heavily,
and how concentrated its cross-attention was over its own local window
(alongside ground-truth error, matched via match_tokens_to_state exactly
like diagnose_token_dropout.py). Built alongside the architecture (see
docs/superpowers/specs/2026-09-23-token-track-query-design.md's
Diagnostics section) so a still-dropping-out seed can be inspected for
WHAT changed in kind (e.g. attention diffusing across two tokens during
a close approach), not just whether the aggregate dropout count moved.

Reads TrackQueryDynamics._self_attention/_cross_attention_one directly
with `return_weights=True` rather than recomputing their math in a
second copy -- an earlier version of this script did the latter and
drifted out of sync with the real forward pass (wrong velocities fed to
the query, no window-widening, and a crash on an off-grid token -- see
final review of docs/superpowers/plans/2026-09-23-token-track-query.md).

Usage:
    PYTHONPATH=. python3 scripts/visualize_track_query_attention.py \
        --checkpoint checkpoints/token_model_h12_v17.pt \
        --seed 4738 --num-steps 20
"""
import argparse
import random

import numpy as np
import torch

import bounce
from model.dataset import make_scenario_uniform
from model.token_model import TokenModel
from model.token_match import match_tokens_to_state
from model.token_rasterize import rasterize_tokens


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


def step_with_attention(dynamics, positions, velocities, hidden, observed_frame):
    """Mirrors TrackQueryDynamics.forward's own orchestration exactly
    (same call order, same GRU input concatenation), but calls
    _self_attention/_cross_attention_one with return_weights=True so
    this script sees the actual weights forward() used, instead of a
    second, possibly-drifted computation of the same math."""
    attn_out_self, self_edge_weights = dynamics._self_attention(
        positions, velocities, hidden, return_weights=True
    )
    query_vecs = dynamics.cross_query(attn_out_self)

    prob, vx, vy = observed_frame[0], observed_frame[1], observed_frame[2]
    readouts = []
    obs_positions = []
    cross_window_infos = []
    for i in range(positions.shape[0]):
        readout, obs_pos, window_info = dynamics._cross_attention_one(
            prob, vx, vy, positions[i], query_vecs[i], return_weights=True
        )
        readouts.append(readout)
        obs_positions.append(obs_pos)
        cross_window_infos.append(window_info)
    cross_readout = torch.stack(readouts, dim=0)
    obs_pos = torch.stack(obs_positions, dim=0)

    gru_input = torch.cat([attn_out_self, cross_readout], dim=-1)
    new_hidden = dynamics.gru(gru_input, hidden)
    delta = dynamics.delta_head(new_hidden)
    delta_pos, delta_vel = delta[:, :2], delta[:, 2:]
    return delta_pos, delta_vel, new_hidden, obs_pos, self_edge_weights, cross_window_infos


def format_self_attention(self_edge_weights, num_tokens, this_token):
    if self_edge_weights is None:
        return ""
    src, dst, weights = self_edge_weights
    neighbors = [
        (int(src[e]), float(weights[e]))
        for e in range(dst.shape[0])
        if int(dst[e]) == this_token and int(src[e]) != this_token
    ]
    neighbors.sort(key=lambda p: -p[1])
    return ", ".join(f"tok{s}:{w:.2f}" for s, w in neighbors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--num-balls", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--neighbor-radius", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--gravity", type=float, default=9.0)
    args = ap.parse_args()

    model = TokenModel(n=args.n, radius=0.75, dt=0.15, hidden_dim=args.hidden_dim,
                        neighbor_radius=args.neighbor_radius, track_query=True)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    frames, states = simulate(args.n, args.num_balls, args.seed, args.num_steps, gravity=args.gravity)
    g0 = torch.from_numpy(frames[0].transpose(2, 0, 1))
    g1 = torch.from_numpy(frames[1].transpose(2, 0, 1))

    with torch.no_grad():
        positions, velocities, hidden = model.init_tokens(g0, g1)
        state1 = {k: torch.tensor(v, dtype=torch.float32) for k, v in states[1].items()}
        token_to_ball = match_tokens_to_state(positions, state1)

        observed = g1
        for step in range(args.num_steps - 1):
            gt_state = states[step + 1]
            gt_pos = torch.stack([
                torch.tensor(gt_state["x"], dtype=torch.float32),
                torch.tensor(gt_state["y"], dtype=torch.float32),
            ], dim=1)

            delta_pos, delta_vel, new_hidden, obs_pos, self_edge_weights, cross_window_infos = \
                step_with_attention(model.dynamics, positions, velocities, hidden, observed)

            print(f"--- step {step} ---")
            for t in range(positions.shape[0]):
                ball = int(token_to_ball[t])
                gt_err = float(torch.norm(positions[t] - gt_pos[ball]))
                self_str = format_self_attention(self_edge_weights, positions.shape[0], t)
                window_info = cross_window_infos[t]
                if window_info is None:
                    cross_str = "BAILOUT (no mass in any expansion)"
                else:
                    w = window_info["weights"]
                    cross_str = f"max={float(w.max()):.3f} window={tuple(w.shape)}"
                print(f"  token {t} (ball {ball}): gt_err={gt_err:.3f} self_attn->[{self_str}] cross_attn: {cross_str}")

            final_pos = positions + velocities * model.dt + delta_pos
            final_vel = velocities + delta_vel
            positions, velocities, hidden = final_pos, final_vel, new_hidden
            observed = rasterize_tokens(final_pos, final_vel, model.n, model.radius)


if __name__ == "__main__":
    main()
