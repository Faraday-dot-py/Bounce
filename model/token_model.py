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
                 detect_threshold=0.1, observation_weight=0.5, detect_margin=1.0,
                 max_init_speed=20.0):
        super().__init__()
        self.n = n
        self.radius = radius
        self.dt = dt
        self.detect_threshold = detect_threshold
        self.observation_weight = observation_weight
        # Must match centroid_near's own `margin` default -- the gate width
        # below is derived from the detection window's extent, so the two
        # must not drift apart silently.
        self.detect_margin = detect_margin
        self.max_init_speed = max_init_speed
        self.dynamics = TokenDynamics(hidden_dim=hidden_dim, neighbor_radius=neighbor_radius)

    def _gate_radius(self):
        """The occlusion gate must be wider than the physical contact
        radius. centroid_near's window extends ceil(radius + detect_margin)
        cells from the query point, and a neighbouring ball's mass reaches
        `radius` beyond its own centre, so a neighbour contaminates this
        token's centroid once its centre is within
        ceil(radius + detect_margin) + radius of ours -- 2.75 cells at the
        default radius=0.75, margin=1.0. occluding_mask thresholds at
        2 * its argument, so returning radius + detect_margin gives an
        effective threshold of 2*(radius + detect_margin) = 3.5 cells,
        which covers that 2.75 with margin. Using the narrower physical
        threshold (2*radius = 1.5) instead leaves a band where the gate
        reports "safe" while centroid_near actually reads a blended
        midpoint between two balls -- precisely during every approach and
        separation, which is the collision-time signal this design exists
        to protect."""
        return self.radius + self.detect_margin

    def init_tokens(self, first_frame, second_frame):
        """Detects tokens from `second_frame`; estimates velocity by
        finite difference against `first_frame`'s nearest detection to
        each token (detection order isn't stable across frames)."""
        pos0 = find_token_positions(first_frame[0], self.radius, self.detect_threshold)
        pos1 = find_token_positions(second_frame[0], self.radius, self.detect_threshold)
        dtype = first_frame.dtype
        device = first_frame.device
        if pos1.shape[0] == 0:
            return (pos1,
                    torch.zeros((0, 2), dtype=dtype, device=device),
                    torch.zeros((0, self.dynamics.hidden_dim), dtype=dtype, device=device))
        if pos0.shape[0] == 0:
            velocities = torch.zeros_like(pos1)
        else:
            dists = torch.cdist(pos1, pos0)
            nearest = dists.argmin(dim=1)
            velocities = (pos1 - pos0[nearest]) / self.dt
            # Nearest-match pairing is unbounded, so a token that was missed
            # (or merged) in one of the two frames can pair with a different
            # ball entirely and yield an absurd velocity that then throws the
            # token off-grid. Reject any match implying a speed above
            # max_init_speed (default 20 cells/s, roughly free-fall speed
            # across a 20-cell grid at gravity=9.0 and far above the ~3.25
            # cells/s spawn maximum) and coast from rest instead -- a zero
            # velocity is recoverable by the dynamics network, a 20x-too-fast
            # one is not.
            max_match_distance = self.max_init_speed * self.dt
            matched_dist = torch.gather(dists, 1, nearest.unsqueeze(1)).squeeze(1)
            velocities = torch.where(
                (matched_dist > max_match_distance).unsqueeze(1),
                torch.zeros_like(velocities),
                velocities,
            )
        hidden = torch.zeros((pos1.shape[0], self.dynamics.hidden_dim), dtype=dtype, device=device)
        return pos1, velocities, hidden

    def step(self, positions, velocities, hidden, observed_frame):
        """Advances one dt. `observed_frame` is the grid at the SAME time
        as the input `positions`/`velocities` (time t) -- ground truth
        during teacher-forced training, the model's own previous
        rasterized output during self-feed rollout. The observation
        branch corrects the CURRENT (time-t) position estimate using this
        frame BEFORE advancing physics: correcting a not-yet-computed
        time-(t+1) prediction against a time-t frame is a category error,
        and differencing a time-t observation against the time-t input
        position yields not a velocity but ~0 by construction. A single
        frame carries no velocity information at all, so velocity is
        corrected only implicitly, through what the dynamics network
        learns from clean before/after pairs during training -- never
        synthesized from a single-frame position delta divided by dt."""
        # The gate is evaluated against the current positions, i.e. the
        # same time instant as `observed_frame`: the question it answers is
        # whether THIS frame's observation is contaminated by a neighbour,
        # which depends on where the tokens are now, not on a future
        # predicted position. (The spec's "predictive ... anticipates an
        # upcoming overlap" phrasing predates that timing being pinned
        # down; the widened gate radius below is what buys the intended
        # early suppression.)
        occluding = occluding_mask(positions, self._gate_radius())
        corrected_pos = positions.clone()
        w = self.observation_weight
        for i in range(positions.shape[0]):
            if occluding[i] or w == 0.0:
                continue
            obs_pos = centroid_near(observed_frame[0], positions[i], self.radius)
            corrected_pos[i] = (1 - w) * positions[i] + w * obs_pos

        delta_pos, delta_vel, new_hidden = self.dynamics(corrected_pos, velocities, hidden)
        final_pos = corrected_pos + velocities * self.dt + delta_pos
        final_vel = velocities + delta_vel

        next_grid = rasterize_tokens(final_pos, final_vel, self.n, self.radius)
        return final_pos, final_vel, new_hidden, next_grid
