import math

import torch
import torch.nn as nn

from model.token_graph import build_radius_graph


class TrackQueryDynamics(nn.Module):
    """Drop-in alternative to model.token_net.TokenDynamics that resolves
    token identity through two stages of learned attention instead of a
    hard occlusion gate + spatial mask -- see
    docs/superpowers/specs/2026-09-23-token-track-query-design.md.

    Stage 1 (self-attention among tokens) reuses TokenDynamics's radius-
    graph GAT shape verbatim, including its self-loop convention (a
    token with exactly one neighbor otherwise forms a softmax group of
    size 1 with zero gradient to query/key).

    Stage 2 (cross-attention to the observed scene, model/token_track_query.py's
    _cross_attention_one) replaces occluding_mask + centroid_near
    entirely in this path: no hard gate, no territory mask, just a
    learned softmax read over the same bounded local window
    centroid_near already searches.

    Both stages depend only on relative offsets, never absolute
    position -- required for train-small/tile-large transfer."""

    def __init__(self, hidden_dim=32, neighbor_radius=3.0, radius=0.75,
                 margin=1.0, max_expansions=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.neighbor_radius = neighbor_radius
        self.radius = radius
        self.margin = margin
        self.max_expansions = max_expansions

        self_node_dim = 2 + hidden_dim  # velocity + hidden -- no absolute position
        self_edge_dim = 2  # relative position offset (dx, dy)
        self.self_query = nn.Linear(self_node_dim, hidden_dim)
        self.self_key = nn.Linear(self_node_dim + self_edge_dim, hidden_dim)
        self.self_value = nn.Linear(self_node_dim + self_edge_dim, hidden_dim)

        cross_cell_dim = 5  # PROB, VX, VY, dx, dy
        self.cross_query = nn.Linear(hidden_dim, hidden_dim)
        self.cross_key = nn.Linear(cross_cell_dim, hidden_dim)
        self.cross_value = nn.Linear(cross_cell_dim, hidden_dim)

        self.gru = nn.GRUCell(2 * hidden_dim, hidden_dim)
        self.delta_head = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def _self_attention(self, positions, velocities, hidden):
        n = positions.shape[0]
        node_state = torch.cat([velocities, hidden], dim=-1)
        q = self.self_query(node_state)

        edge_index = build_radius_graph(positions, self.neighbor_radius)
        self_loops = torch.arange(n, device=positions.device)
        self_loops = torch.stack([self_loops, self_loops], dim=0)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        attn_out = torch.zeros(n, self.hidden_dim, device=positions.device, dtype=positions.dtype)
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            rel_pos = positions[src] - positions[dst]
            edge_input = torch.cat([node_state[src], rel_pos], dim=-1)
            k = self.self_key(edge_input)
            v = self.self_value(edge_input)
            scores = (q[dst] * k).sum(dim=-1) / (self.hidden_dim ** 0.5)
            weights = torch.exp(scores - scores.max())
            denom = torch.zeros(n, device=positions.device, dtype=positions.dtype)
            denom = denom.index_add(0, dst, weights)
            weights = weights / denom[dst].clamp(min=1e-6)
            attn_out = attn_out.index_add(0, dst, weights.unsqueeze(-1) * v)
        return attn_out
