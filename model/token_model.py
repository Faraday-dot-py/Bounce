import torch

from model.token_net import TokenDynamics
from model.token_detect import find_token_positions, centroid_near
from model.token_gate import occluding_mask
from model.token_rasterize import rasterize_tokens


class TokenModel(torch.nn.Module):
    """Wires together token initialization, per-step dynamics, the
    occlusion gate, and rasterization -- see
    docs/superpowers/specs/2026-09-22-token-per-ball-model-design.md."""

    def __init__(self, n, radius, dt, hidden_dim=32, neighbor_radius=3.0,
                 detect_threshold=0.1, observation_weight=0.5):
        super().__init__()
        self.n = n
        self.radius = radius
        self.dt = dt
        self.detect_threshold = detect_threshold
        self.observation_weight = observation_weight
        self.dynamics = TokenDynamics(hidden_dim=hidden_dim, neighbor_radius=neighbor_radius)

    def init_tokens(self, first_frame, second_frame):
        """Detects tokens from `second_frame`; estimates velocity by
        finite difference against `first_frame`'s nearest detection to
        each token (detection order isn't stable across frames)."""
        pos0 = find_token_positions(first_frame[0], self.radius, self.detect_threshold)
        pos1 = find_token_positions(second_frame[0], self.radius, self.detect_threshold)
        dtype = first_frame.dtype
        if pos1.shape[0] == 0:
            return pos1, torch.zeros((0, 2), dtype=dtype), torch.zeros((0, self.dynamics.hidden_dim), dtype=dtype)
        if pos0.shape[0] == 0:
            velocities = torch.zeros_like(pos1)
        else:
            dists = torch.cdist(pos1, pos0)
            nearest = dists.argmin(dim=1)
            velocities = (pos1 - pos0[nearest]) / self.dt
        hidden = torch.zeros((pos1.shape[0], self.dynamics.hidden_dim), dtype=dtype)
        return pos1, velocities, hidden

    def step(self, positions, velocities, hidden, observed_frame):
        """Advances one dt. `observed_frame` is whatever grid the
        observation branch should read from -- ground truth during
        teacher-forced training, the model's own previous rasterized
        output during self-feed rollout; the caller decides which."""
        delta_pos, delta_vel, new_hidden = self.dynamics(positions, velocities, hidden)
        predicted_pos = positions + velocities * self.dt + delta_pos
        predicted_vel = velocities + delta_vel

        occluding = occluding_mask(predicted_pos, self.radius)
        final_pos = predicted_pos.clone()
        final_vel = predicted_vel.clone()
        w = self.observation_weight
        for i in range(predicted_pos.shape[0]):
            if occluding[i] or w == 0.0:
                continue
            obs_pos = centroid_near(observed_frame[0], predicted_pos[i], self.radius)
            obs_vel = (obs_pos - positions[i]) / self.dt
            final_pos[i] = (1 - w) * predicted_pos[i] + w * obs_pos
            final_vel[i] = (1 - w) * predicted_vel[i] + w * obs_vel

        next_grid = rasterize_tokens(final_pos, final_vel, self.n, self.radius)
        return final_pos, final_vel, new_hidden, next_grid
