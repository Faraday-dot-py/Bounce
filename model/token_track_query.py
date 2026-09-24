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

    def _self_attention(self, positions, velocities, hidden, return_weights=False):
        """return_weights=True additionally returns (src, dst, weights) --
        the per-edge softmax weights this method itself computes but
        normally discards after pooling into attn_out -- so a diagnostic
        caller (scripts/visualize_track_query_attention.py) can inspect
        exactly what forward() used, instead of recomputing this method's
        math in a second copy that can drift out of sync with it."""
        n = positions.shape[0]
        node_state = torch.cat([velocities, hidden], dim=-1)
        q = self.self_query(node_state)

        edge_index = build_radius_graph(positions, self.neighbor_radius)
        self_loops = torch.arange(n, device=positions.device)
        self_loops = torch.stack([self_loops, self_loops], dim=0)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        attn_out = torch.zeros(n, self.hidden_dim, device=positions.device, dtype=positions.dtype)
        edge_weights = None
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
            if return_weights:
                edge_weights = (src, dst, weights)
        if return_weights:
            return attn_out, edge_weights
        return attn_out

    def _cross_attention_one(self, prob, vx, vy, position, query_vec, return_weights=False):
        """Cross-attention for one token's bounded local window --
        replaces occluding_mask + centroid_near entirely in this path.
        Gathers the same window centroid_near searches (base half-width
        ceil(radius + margin), widening by one cell per retry up to
        max_expansions if empty), builds a per-cell feature
        [PROB, VX, VY, dx, dy] (dx/dy relative to this token's own
        position -- no absolute position enters), and reads it with a
        learned softmax attention query built from the token's
        post-self-attention hidden state. Returns (readout, obs_pos);
        obs_pos is the attention-weighted centroid, kept only as a
        diagnostic/bookkeeping byproduct (never a hard blend). If every
        expansion's window is empty, falls back to an all-zero readout
        and obs_pos == position -- there is no mask here, so there is no
        masked-then-unmasked retry to replicate (unlike centroid_near).

        return_weights=True additionally returns a third value: a dict
        with the actual window bounds and softmax weights this call
        settled on (whichever expansion succeeded, or None if every
        expansion was empty) -- so a diagnostic caller can see exactly
        which cells and weights this method used, instead of
        recomputing the window-search in a second copy that can drift
        out of sync with it (in particular, silently missing the
        widening this method does)."""
        n = prob.shape[0]
        cx = int(round(float(position[0].detach())))
        cy = int(round(float(position[1].detach())))
        step = max(1, int(math.ceil(self.radius)))
        base_half = int(math.ceil(self.radius + self.margin))
        for expansion in range(self.max_expansions + 1):
            half = base_half + expansion * step
            i_lo_raw, i_hi_raw = cx - half, cx + half
            j_lo_raw, j_hi_raw = cy - half, cy + half
            if i_hi_raw < 0 or i_lo_raw > n - 1 or j_hi_raw < 0 or j_lo_raw > n - 1:
                continue
            i_lo, i_hi = max(0, i_lo_raw), min(n - 1, i_hi_raw)
            j_lo, j_hi = max(0, j_lo_raw), min(n - 1, j_hi_raw)
            window_prob = prob[i_lo:i_hi + 1, j_lo:j_hi + 1]
            if float(window_prob.sum()) <= 1e-6:
                continue
            window_vx = vx[i_lo:i_hi + 1, j_lo:j_hi + 1]
            window_vy = vy[i_lo:i_hi + 1, j_lo:j_hi + 1]
            ii = torch.arange(i_lo, i_hi + 1, device=prob.device, dtype=prob.dtype).view(-1, 1)
            jj = torch.arange(j_lo, j_hi + 1, device=prob.device, dtype=prob.dtype).view(1, -1)
            dx = (ii - position[0]).expand_as(window_prob)
            dy = (jj - position[1]).expand_as(window_prob)
            cell_feat = torch.stack([window_prob, window_vx, window_vy, dx, dy], dim=-1)
            cell_feat = cell_feat.reshape(-1, 5)
            k = self.cross_key(cell_feat)
            v = self.cross_value(cell_feat)
            scores = (query_vec.unsqueeze(0) * k).sum(dim=-1) / (self.hidden_dim ** 0.5)
            weights = torch.softmax(scores, dim=0)
            readout = (weights.unsqueeze(-1) * v).sum(dim=0)
            centroid_dx = (weights * dx.reshape(-1)).sum()
            centroid_dy = (weights * dy.reshape(-1)).sum()
            obs_pos = position + torch.stack([centroid_dx, centroid_dy])
            if return_weights:
                window_info = {
                    "i_lo": i_lo, "i_hi": i_hi, "j_lo": j_lo, "j_hi": j_hi,
                    "weights": weights.reshape(window_prob.shape),
                }
                return readout, obs_pos, window_info
            return readout, obs_pos
        if return_weights:
            return query_vec.new_zeros(self.hidden_dim), position, None
        return query_vec.new_zeros(self.hidden_dim), position

    def forward(self, positions, velocities, hidden, observed_frame):
        n = positions.shape[0]
        if n == 0:
            zeros2 = positions.new_zeros((0, 2))
            zeros_h = hidden.new_zeros((0, self.hidden_dim))
            return zeros2, zeros2, zeros_h, zeros2

        attn_out_self = self._self_attention(positions, velocities, hidden)
        query_vecs = self.cross_query(attn_out_self)

        prob, vx, vy = observed_frame[0], observed_frame[1], observed_frame[2]
        readouts = []
        obs_positions = []
        for i in range(n):
            readout, obs_pos = self._cross_attention_one(prob, vx, vy, positions[i], query_vecs[i])
            readouts.append(readout)
            obs_positions.append(obs_pos)
        cross_readout = torch.stack(readouts, dim=0)
        obs_pos = torch.stack(obs_positions, dim=0)

        gru_input = torch.cat([attn_out_self, cross_readout], dim=-1)
        new_hidden = self.gru(gru_input, hidden)
        delta = self.delta_head(new_hidden)
        return delta[:, :2], delta[:, 2:], new_hidden, obs_pos
