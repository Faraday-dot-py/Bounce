import torch
import torch.nn as nn

from model.token_graph import build_radius_graph


class TokenDynamics(nn.Module):
    """Predicts each token's next (position, velocity) delta and updates
    its hidden state, using a radius-graph attention layer so a token's
    update depends only on neighbors within `neighbor_radius` -- see
    design spec's Proximity-graph attention section. Every token also
    attends to itself in addition to its radius-neighbors. Absolute
    position is deliberately *not* a node feature: only the relative
    offset between an edge's endpoints enters the computation, so the
    update is translation-invariant by construction. That is what makes
    train-small/tile-large validation meaningful -- a token's update must
    depend on local geometry alone, not on where in the grid it sits.
    The delta head is
    zero-initialized so the model starts as an exact "coast at current
    velocity" identity (see TokenModel.step, which adds `velocities * dt`
    as the base prediction) -- the same zero-init-residual convention as
    model/net.py, for the same reason: a decent physics-agnostic identity
    is a strong starting point before any learned correction.
    """

    def __init__(self, hidden_dim=32, neighbor_radius=3.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.neighbor_radius = neighbor_radius
        node_dim = 2 + hidden_dim  # velocity + hidden -- no absolute position
        edge_dim = 2  # relative position offset (dx, dy)
        self.query = nn.Linear(node_dim, hidden_dim)
        self.key = nn.Linear(node_dim + edge_dim, hidden_dim)
        self.value = nn.Linear(node_dim + edge_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.delta_head = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, positions, velocities, hidden):
        n = positions.shape[0]
        node_state = torch.cat([velocities, hidden], dim=-1)
        q = self.query(node_state)

        # A token with exactly one radius-graph neighbor forms a softmax group
        # of size 1, whose output is always exactly 1.0 regardless of the
        # score -- zero gradient to query/key in that case. Add a self-loop
        # per token (standard GAT convention) so every group has size >= 2.
        edge_index = build_radius_graph(positions, self.neighbor_radius)
        self_loops = torch.arange(n, device=positions.device)
        self_loops = torch.stack([self_loops, self_loops], dim=0)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        attn_out = torch.zeros(n, self.hidden_dim, device=positions.device, dtype=positions.dtype)
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            rel_pos = positions[src] - positions[dst]
            edge_input = torch.cat([node_state[src], rel_pos], dim=-1)
            k = self.key(edge_input)
            v = self.value(edge_input)
            scores = (q[dst] * k).sum(dim=-1) / (self.hidden_dim ** 0.5)
            weights = torch.exp(scores - scores.max())
            denom = torch.zeros(n, device=positions.device, dtype=positions.dtype)
            denom = denom.index_add(0, dst, weights)
            weights = weights / denom[dst].clamp(min=1e-6)
            attn_out = attn_out.index_add(0, dst, weights.unsqueeze(-1) * v)

        new_hidden = self.gru(attn_out, hidden)
        delta = self.delta_head(new_hidden)
        return delta[:, :2], delta[:, 2:], new_hidden
