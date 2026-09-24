"""Dumps per-step, per-token self-/cross-attention weights for a
track-query TokenModel rollout, aligned against diagnose_token_dropout.py's
per-step trace output (window total, occlusion state would be reported
by that script; occlusion/window-total are gate-and-mask concepts that
don't apply to this architecture, so this script reports the two things
that DO apply here: which neighbors a token's self-attention weighted
most heavily, and how concentrated its cross-attention was over its own
local window). Built alongside the architecture (see
docs/superpowers/specs/2026-09-23-token-track-query-design.md's
Diagnostics section) so a still-dropping-out seed can be inspected for
WHAT changed in kind (e.g. attention diffusing across two tokens during
a close approach), not just whether the aggregate dropout count moved.

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
from model.token_graph import build_radius_graph


def simulate(n, num_balls, seed, num_steps, dt=0.15, gravity=9.0, radius=0.75,
             stiffness=400.0, substeps=8, vy=2.3):
    rng = random.Random(seed)
    balls = make_scenario_uniform(num_balls, n, vy, rng)
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    frames = [np.array(G, dtype=np.float32)]
    for _ in range(num_steps):
        bounce.step(G, n, balls, dt, gravity, radius, stiffness, substeps)
        frames.append(np.array(G, dtype=np.float32))
    return frames


def self_attention_weights(dynamics, positions, velocities, hidden):
    """Recomputes _self_attention's per-edge softmax weights (not just
    its pooled output) for inspection -- mirrors
    TrackQueryDynamics._self_attention exactly, since that method itself
    only returns the pooled sum, not the weights."""
    n = positions.shape[0]
    node_state = torch.cat([velocities, hidden], dim=-1)
    q = dynamics.self_query(node_state)
    edge_index = build_radius_graph(positions, dynamics.neighbor_radius)
    self_loops = torch.arange(n)
    self_loops = torch.stack([self_loops, self_loops], dim=0)
    edge_index = torch.cat([edge_index, self_loops], dim=1)
    if edge_index.shape[1] == 0:
        return {}
    src, dst = edge_index[0], edge_index[1]
    rel_pos = positions[src] - positions[dst]
    edge_input = torch.cat([node_state[src], rel_pos], dim=-1)
    k = dynamics.self_key(edge_input)
    scores = (q[dst] * k).sum(dim=-1) / (dynamics.hidden_dim ** 0.5)
    weights = torch.exp(scores - scores.max())
    denom = torch.zeros(n)
    denom = denom.index_add(0, dst, weights)
    weights = weights / denom[dst].clamp(min=1e-6)
    by_dst = {}
    for e in range(edge_index.shape[1]):
        d, s, w = int(dst[e]), int(src[e]), float(weights[e])
        by_dst.setdefault(d, []).append((s, w))
    return by_dst


def cross_attention_concentration(dynamics, positions, hidden, observed_frame):
    """Runs _cross_attention_one per token and reports each one's max
    softmax weight (close to 1/window_size = diffuse/uncertain, close to
    1.0 = confidently locked onto one cell) -- the cross-attention analog
    of asking whether a token's read was decisive or smeared."""
    query_vecs = dynamics.cross_query(dynamics._self_attention(
        positions, torch.zeros_like(positions), hidden
    ))
    prob, vx, vy = observed_frame[0], observed_frame[1], observed_frame[2]
    results = []
    for i in range(positions.shape[0]):
        n = prob.shape[0]
        cx = int(round(float(positions[i, 0])))
        cy = int(round(float(positions[i, 1])))
        half = int(np.ceil(dynamics.radius + dynamics.margin))
        i_lo, i_hi = max(0, cx - half), min(n - 1, cx + half)
        j_lo, j_hi = max(0, cy - half), min(n - 1, cy + half)
        window_prob = prob[i_lo:i_hi + 1, j_lo:j_hi + 1]
        window_vx = vx[i_lo:i_hi + 1, j_lo:j_hi + 1]
        window_vy = vy[i_lo:i_hi + 1, j_lo:j_hi + 1]
        ii = torch.arange(i_lo, i_hi + 1, dtype=prob.dtype).view(-1, 1)
        jj = torch.arange(j_lo, j_hi + 1, dtype=prob.dtype).view(1, -1)
        dx = (ii - positions[i, 0]).expand_as(window_prob)
        dy = (jj - positions[i, 1]).expand_as(window_prob)
        cell_feat = torch.stack([window_prob, window_vx, window_vy, dx, dy], dim=-1).reshape(-1, 5)
        k = dynamics.cross_key(cell_feat)
        scores = (query_vecs[i].unsqueeze(0) * k).sum(dim=-1) / (dynamics.hidden_dim ** 0.5)
        weights = torch.softmax(scores, dim=0)
        results.append(float(weights.max()))
    return results


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

    frames = simulate(args.n, args.num_balls, args.seed, args.num_steps, gravity=args.gravity)
    g0 = torch.from_numpy(frames[0].transpose(2, 0, 1))
    g1 = torch.from_numpy(frames[1].transpose(2, 0, 1))

    with torch.no_grad():
        positions, velocities, hidden = model.init_tokens(g0, g1)
        observed = g1
        for step in range(args.num_steps - 1):
            self_weights = self_attention_weights(model.dynamics, positions, velocities, hidden)
            cross_conc = cross_attention_concentration(model.dynamics, positions, hidden, observed)
            print(f"--- step {step} ---")
            for t in range(positions.shape[0]):
                neighbors = sorted(self_weights.get(t, []), key=lambda p: -p[1])
                neighbor_str = ", ".join(f"tok{s}:{w:.2f}" for s, w in neighbors if s != t)
                print(f"  token {t}: self_attn->[{neighbor_str}] cross_attn_max={cross_conc[t]:.3f}")
            positions, velocities, hidden, pred_grid, _ = model.step(positions, velocities, hidden, observed)
            observed = pred_grid


if __name__ == "__main__":
    main()
