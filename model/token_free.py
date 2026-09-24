import torch
import torch.nn as nn

from model.token_graph import build_radius_graph


def wall_features(positions, n, wall_range):
    """(N, 4) proximity to each wall, 0 beyond `wall_range` cells and 1 at
    or past the wall itself. Zero in the interior, so interior translation
    invariance is preserved; the wall itself is the one absolute-position
    fact the dynamics needs once observation correction is gone."""
    x, y = positions[:, 0], positions[:, 1]
    d = torch.stack([x, (n - 1) - x, y, (n - 1) - y], dim=1)
    return (wall_range - d.clamp(0.0, wall_range)) / wall_range


def wall_contact_features(positions, velocities, n, radius, dt):
    """(N, 8): penetration depth into each wall now, then after one
    free-flight step (`pos + vel * dt`), as relu(radius - distance) / radius.
    Contact -- the stiff, near-elastic wall impulse in bounce.py -- starts
    at penetration > 0, and whether a step is an impulse step depends on
    where the ball will be, so the lookahead half tells the network that
    before it happens."""
    def penetration(pos):
        x, y = pos[:, 0], pos[:, 1]
        d = torch.stack([x, (n - 1) - x, y, (n - 1) - y], dim=1)
        return (radius - d).clamp(min=0.0) / radius
    return torch.cat([penetration(positions), penetration(positions + velocities * dt)], dim=1)


class TokenFreeDynamics(nn.Module):
    """TokenDynamics (radius-graph attention + GRU + zero-init delta head)
    with wall-proximity node features -- see
    docs/superpowers/specs/2026-09-23-token-free-rollout-design.md."""

    def __init__(self, n, hidden_dim=32, neighbor_radius=4.0, wall_range=3.0, mirror_sym=False,
                 wall_lookahead=False, wall_head=False, radius=0.75, dt=0.15):
        super().__init__()
        self.n = n
        self.radius = radius
        self.dt = dt
        self.wall_lookahead = wall_lookahead
        self.core_dim = hidden_dim
        self.mirror_sym = mirror_sym
        self.hidden_dim = hidden_dim * (2 if mirror_sym else 1)
        self.neighbor_radius = neighbor_radius
        self.wall_range = wall_range
        wall_dim = 4 + (8 if wall_lookahead else 0)
        node_dim = 2 + wall_dim + self.core_dim
        edge_dim = 2
        self.query = nn.Linear(node_dim, self.core_dim)
        self.key = nn.Linear(node_dim + edge_dim, self.core_dim)
        self.value = nn.Linear(node_dim + edge_dim, self.core_dim)
        self.gru = nn.GRUCell(self.core_dim, self.core_dim)
        self.delta_head = nn.Linear(self.core_dim, 4)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.wall_head = None
        if wall_head:
            # Separate two-layer path from (velocity, wall features) to the
            # delta, outside the GRU: the wall impulse is a sharp function
            # of position and velocity that the linear->GRU->linear path
            # only expresses as a smooth brake (see docs/debugging/
            # experiment-log.md, wall-bounce diagnosis).
            self.wall_head = nn.Sequential(nn.Linear(2 + wall_dim, self.core_dim), nn.ReLU(),
                                           nn.Linear(self.core_dim, 4))
            nn.init.zeros_(self.wall_head[2].weight)
            nn.init.zeros_(self.wall_head[2].bias)

    def forward(self, positions, velocities, hidden):
        if not self.mirror_sym:
            return self._core(positions, velocities, hidden)
        h, h_m = hidden[:, :self.core_dim], hidden[:, self.core_dim:]
        dp, dv, nh = self._core(positions, velocities, h)
        mirrored_pos = torch.stack([positions[:, 0], (self.n - 1) - positions[:, 1]], dim=1)
        mirrored_vel = velocities * velocities.new_tensor([1.0, -1.0])
        dp_m, dv_m, nh_m = self._core(mirrored_pos, mirrored_vel, h_m)
        flip = dp.new_tensor([1.0, -1.0])
        return (dp + dp_m * flip) / 2, (dv + dv_m * flip) / 2, torch.cat([nh, nh_m], dim=-1)

    def _core(self, positions, velocities, hidden):
        n = positions.shape[0]
        walls = wall_features(positions, self.n, self.wall_range)
        if self.wall_lookahead:
            walls = torch.cat([walls, wall_contact_features(positions, velocities, self.n, self.radius, self.dt)], dim=1)
        node_state = torch.cat([velocities, walls, hidden], dim=-1)
        q = self.query(node_state)

        edge_index = build_radius_graph(positions, self.neighbor_radius)
        self_loops = torch.arange(n, device=positions.device)
        self_loops = torch.stack([self_loops, self_loops], dim=0)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        attn_out = torch.zeros(n, self.core_dim, device=positions.device, dtype=positions.dtype)
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            rel_pos = positions[src] - positions[dst]
            edge_input = torch.cat([node_state[src], rel_pos], dim=-1)
            k = self.key(edge_input)
            v = self.value(edge_input)
            scores = (q[dst] * k).sum(dim=-1) / (self.core_dim ** 0.5)
            weights = torch.exp(scores - scores.max())
            denom = torch.zeros(n, device=positions.device, dtype=positions.dtype)
            denom = denom.index_add(0, dst, weights)
            weights = weights / denom[dst].clamp(min=1e-6)
            attn_out = attn_out.index_add(0, dst, weights.unsqueeze(-1) * v)

        new_hidden = self.gru(attn_out, hidden)
        delta = self.delta_head(new_hidden)
        if self.wall_head is not None:
            delta = delta + self.wall_head(torch.cat([velocities, walls], dim=-1))
        return delta[:, :2], delta[:, 2:], new_hidden
