import torch
from scipy.optimize import linear_sum_assignment

from model.token_free import TokenFreeDynamics
from model.token_net import TokenDynamics
from model.token_track_query import TrackQueryDynamics
from model.token_detect import find_token_positions, centroid_near, read_token_velocities
from model.token_refine import refine_positions
from model.token_gate import occluding_mask
from model.token_rasterize import rasterize_tokens


def _assign_velocity_pairs(dists, ambiguity_margin=0.5):
    """One-to-one optimal assignment (Hungarian) between pos1 rows and
    pos0 columns for velocity finite-differencing, replacing independent
    per-row nearest-match: two tokens whose frame0 candidates are close
    together can otherwise both claim the same frame0 detection under
    plain argmin, leaving one frame0 ball with no partner and silently
    producing a garbage velocity for whichever token loses the tie (see
    docs/debugging/experiment-log.md, seed 4750 duplicate-match case).
    When pos0 has fewer candidates than pos1, some rows go unmatched
    (`nearest` is -1 for those).

    Also flags a row as ambiguous when its assigned distance isn't at
    least `ambiguity_margin` cells better than its own next-best
    candidate distance: a near-tie in frame0 (two balls close together)
    means the *chosen* member of the pair could easily be the wrong one
    even under an optimal global assignment, and a confidently wrong
    velocity compounds into large rollout drift (seed 4752/4742 cases)
    -- an honest zero, recoverable by the dynamics network, is safer
    than committing to a coin-flip pairing."""
    n1, n0 = dists.shape
    dists_np = dists.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(dists_np)
    nearest = torch.full((n1,), -1, dtype=torch.long, device=dists.device)
    nearest[torch.as_tensor(row_ind, device=dists.device)] = torch.as_tensor(col_ind, device=dists.device)

    sorted_d, _ = torch.sort(dists, dim=1)
    if n0 > 1:
        second_best = sorted_d[:, 1]
    else:
        second_best = torch.full((n1,), float("inf"), dtype=dists.dtype, device=dists.device)

    valid = nearest >= 0
    safe_idx = nearest.clamp(min=0)
    matched_dist = torch.gather(dists, 1, safe_idx.unsqueeze(1)).squeeze(1)
    matched_dist = torch.where(valid, matched_dist, torch.full_like(matched_dist, float("inf")))
    ambiguous = (second_best - matched_dist) < ambiguity_margin
    return nearest, matched_dist, valid & ~ambiguous


class TokenModel(torch.nn.Module):
    """Wires together token initialization, per-step dynamics, the
    occlusion gate, and rasterization -- see
    docs/superpowers/specs/2026-09-22-token-per-ball-model-design.md."""

    def __init__(self, n, radius, dt, hidden_dim=32, neighbor_radius=3.0,
                 detect_threshold=0.1, observation_weight=0.5, detect_margin=1.0,
                 max_init_speed=20.0, velocity_weight=0.0, max_expansions=6,
                 territory_masking=False, track_query=False, free_rollout=False,
                 mirror_sym=False, velocity_readout=False,
                 wall_lookahead=False, wall_head=False, pair_impulse=False,
                 position_refine=False):
        super().__init__()
        # Initial velocity read from the frame's VX/VY channels instead of
        # finite-differenced from two detections (error 2.1 -> 0.01 cells/s,
        # see docs/debugging/experiment-log.md). Off by default so existing
        # checkpoints keep their trained-with init.
        self.velocity_readout = velocity_readout
        # Sub-cell init position refinement (model.token_refine); needs the
        # VX/VY-channel velocities, so it requires velocity_readout.
        if position_refine and not velocity_readout:
            raise ValueError("position_refine requires velocity_readout")
        self.position_refine = position_refine
        self.n = n
        self.radius = radius
        self.dt = dt
        self.detect_threshold = detect_threshold
        self.observation_weight = observation_weight
        # Give-up search width for centroid_near's per-step observation
        # read. Raised from centroid_near's own default (3) -- this helps
        # independent of territory_masking below (a wider unmasked search
        # also recovers more drifted tokens; see
        # docs/debugging/experiment-log.md's v14 final review, which
        # measured this directly: widening alone, with no masking, cut
        # dropout on both the v9 and v14 checkpoints).
        self.max_expansions = max_expansions
        # Opt-in per-token observation-window exclusivity (see
        # model.token_detect.territory_mask). Defaults to False so
        # existing checkpoints -- trained and previously evaluated without
        # it -- get byte-for-byte the same inference they always had;
        # turning this on by default silently changed v9's own measured
        # dropout count (6 -> 8) even though v9 was never trained with it
        # (docs/debugging/experiment-log.md's v14 final review). Pass True
        # to enable it for a model trained with it from the start.
        # centroid_near's masked search falls back to an unmasked retry
        # when a token's own territory is completely empty, so enabling
        # this does not reintroduce the permanent-loss regression found
        # in the first (fallback-less) version of this design -- see
        # centroid_near's own docstring.
        self.territory_masking = territory_masking
        # Explicit velocity re-anchoring: finite-difference of two
        # consecutive observation reads (centroid_near), same idea
        # init_tokens already uses across frame0/frame1, just applied every
        # step instead of only at init. Defaults to 0 (off) so existing
        # checkpoints/tests are unaffected -- see
        # docs/debugging/experiment-log.md's velocity-compounding
        # investigation for why this exists: position gets an explicit
        # observation correction every step, velocity never did, and
        # measurement showed velocity error growing ~5x over 9 self-fed
        # steps while position error (which depends on velocity through
        # `final_pos = corrected_pos + velocities * dt + delta_pos`) grew
        # in lockstep. A naive inference-only version of this (spliced onto
        # a checkpoint trained without it) made things WORSE, not better --
        # TokenDynamics's learned delta_vel implicitly relies on velocity
        # following its own internal trajectory, and perturbing it
        # externally is off-distribution for a frozen checkpoint. This is
        # only expected to help if the network is trained with it present
        # from the start, same as position's observation correction always
        # has been.
        self.velocity_weight = velocity_weight
        # Must match centroid_near's own `margin` default -- the gate width
        # below is derived from the detection window's extent, so the two
        # must not drift apart silently.
        self.detect_margin = detect_margin
        self.max_init_speed = max_init_speed
        # Opt-in architecture swap (see
        # docs/superpowers/specs/2026-09-23-token-track-query-design.md):
        # replaces occluding_mask/centroid_near/blending in `step` with
        # learned self- and cross-attention. Defaults to False so v9's
        # exact code path stays byte-for-byte unchanged and reachable as
        # the reference baseline.
        self.track_query = track_query
        if track_query and (territory_masking or velocity_weight > 0.0):
            # Both are v9-path-only options (occluding_mask/centroid_near
            # blending) -- track_query's step branch never reads either
            # one, so silently accepting the combination would produce a
            # model whose flags claim a combined experiment that never
            # actually runs (final review of
            # docs/superpowers/plans/2026-09-23-token-track-query.md).
            raise ValueError(
                "track_query is incompatible with territory_masking/velocity_weight "
                "(v9-path-only options) -- track_query has its own observation-"
                "correction mechanism (TrackQueryDynamics's cross-attention stage)"
            )
        self.free_rollout = free_rollout
        if free_rollout and (track_query or territory_masking or velocity_weight > 0.0):
            raise ValueError(
                "free_rollout is incompatible with track_query/territory_masking/velocity_weight "
                "-- it never reads an observed frame"
            )
        if (mirror_sym or wall_lookahead or wall_head or pair_impulse) and not free_rollout:
            raise ValueError("mirror_sym/wall_lookahead/wall_head/pair_impulse require free_rollout")
        if free_rollout:
            self.dynamics = TokenFreeDynamics(n=n, hidden_dim=hidden_dim, neighbor_radius=neighbor_radius,
                                              mirror_sym=mirror_sym, wall_lookahead=wall_lookahead,
                                              wall_head=wall_head, radius=radius, dt=dt, pair_impulse=pair_impulse)
        elif track_query:
            self.dynamics = TrackQueryDynamics(
                hidden_dim=hidden_dim, neighbor_radius=neighbor_radius,
                radius=radius, margin=detect_margin, max_expansions=max_expansions,
            )
        else:
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
        finite difference against `first_frame`'s detections, paired by
        optimal one-to-one assignment (see `_assign_velocity_pairs`;
        detection order isn't stable across frames, and two balls close
        together in frame0 can make plain nearest-match pick the same
        frame0 detection for two different frame1 tokens)."""
        pos0 = find_token_positions(first_frame[0], self.radius, self.detect_threshold)
        pos1 = find_token_positions(second_frame[0], self.radius, self.detect_threshold)
        dtype = first_frame.dtype
        device = first_frame.device
        if pos1.shape[0] == 0:
            return (pos1,
                    torch.zeros((0, 2), dtype=dtype, device=device),
                    torch.zeros((0, self.dynamics.hidden_dim), dtype=dtype, device=device))
        if self.velocity_readout:
            velocities = read_token_velocities(second_frame, pos1)
            if self.position_refine:
                pos1 = refine_positions(first_frame, second_frame, pos1, velocities,
                                        radius=self.radius, dt=self.dt)
                velocities = read_token_velocities(second_frame, pos1)
        elif pos0.shape[0] == 0:
            velocities = torch.zeros_like(pos1)
        else:
            dists = torch.cdist(pos1, pos0)
            nearest, matched_dist, keep = _assign_velocity_pairs(dists)
            safe_idx = nearest.clamp(min=0)
            velocities = (pos1 - pos0[safe_idx]) / self.dt
            # A rejected match (unassigned row, an ambiguous near-tie, or an
            # implausible implied speed) is unreliable, so coast from rest
            # instead -- a zero velocity is recoverable by the dynamics
            # network, a wrong-direction one is not. max_init_speed (default
            # 20 cells/s) is roughly free-fall speed across a 20-cell grid at
            # gravity=9.0, far above the ~3.25 cells/s spawn maximum.
            max_match_distance = self.max_init_speed * self.dt
            reject = ~keep | (matched_dist > max_match_distance)
            velocities = torch.where(
                reject.unsqueeze(1),
                torch.zeros_like(velocities),
                velocities,
            )
        hidden = torch.zeros((pos1.shape[0], self.dynamics.hidden_dim), dtype=dtype, device=device)
        return pos1, velocities, hidden

    def step(self, positions, velocities, hidden, observed_frame, prev_obs_pos=None):
        """Advances one dt. `observed_frame` is the grid at the SAME time
        as the input `positions`/`velocities` (time t) -- ground truth
        during teacher-forced training, the model's own previous
        rasterized output during self-feed rollout. The observation
        branch corrects the CURRENT (time-t) position estimate using this
        frame BEFORE advancing physics: correcting a not-yet-computed
        time-(t+1) prediction against a time-t frame is a category error,
        and differencing a time-t observation against the time-t input
        position yields not a velocity but ~0 by construction. A single
        frame carries no velocity information at all, so a *within-step*
        finite difference can't recover it -- but `prev_obs_pos`, the
        observation read one step ago (time t-1), can: `(obs_pos - prev_obs_pos)
        / dt` is a genuine two-frame velocity estimate, same idea
        `init_tokens` already uses at frame0/frame1, applied every step.
        This is blended into `velocities` at `self.velocity_weight` (0 by
        default -- see __init__), gated by the same occlusion mask as
        position, before the dynamics network sees it, matching how
        position's own observation correction happens before, not after,
        `self.dynamics`. Returns `obs_pos` (this step's observation read,
        or the unchanged input position where gated/skipped) so the
        caller can pass it back in as next step's `prev_obs_pos`; the
        first call in a rollout has no prior observation, so pass the
        detected position from `init_tokens` (frame 1) as `prev_obs_pos`
        or leave it `None` to skip velocity correction that step."""
        if self.track_query:
            delta_pos, delta_vel, new_hidden, obs_pos = self.dynamics(
                positions, velocities, hidden, observed_frame
            )
            final_pos = positions + velocities * self.dt + delta_pos
            final_vel = velocities + delta_vel
            next_grid = rasterize_tokens(final_pos, final_vel, self.n, self.radius)
            return final_pos, final_vel, new_hidden, next_grid, obs_pos

        # Existing v9 path -- byte-for-byte unchanged below.
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
        corrected_vel = velocities.clone()
        obs_pos = positions.clone()
        w = self.observation_weight
        vw = self.velocity_weight
        for i in range(positions.shape[0]):
            if occluding[i] or w == 0.0:
                continue
            op = centroid_near(observed_frame[0], positions[i], self.radius,
                                margin=self.detect_margin, max_expansions=self.max_expansions,
                                all_positions=positions if self.territory_masking else None,
                                self_idx=i if self.territory_masking else None)
            obs_pos[i] = op
            corrected_pos[i] = (1 - w) * positions[i] + w * op
            if prev_obs_pos is not None and vw > 0.0:
                obs_vel = (op - prev_obs_pos[i]) / self.dt
                corrected_vel[i] = (1 - vw) * velocities[i] + vw * obs_vel

        delta_pos, delta_vel, new_hidden = self.dynamics(corrected_pos, corrected_vel, hidden)
        final_pos = corrected_pos + corrected_vel * self.dt + delta_pos
        final_vel = corrected_vel + delta_vel

        next_grid = rasterize_tokens(final_pos, final_vel, self.n, self.radius)
        return final_pos, final_vel, new_hidden, next_grid, obs_pos

    def step_free(self, positions, velocities, hidden):
        """Observation-free step: no frame is read, so a token can never
        lose its ball to a faded self-rendered observation -- see
        docs/superpowers/specs/2026-09-23-token-free-rollout-design.md.
        The rasterized grid is output only."""
        if not self.free_rollout:
            raise RuntimeError("step_free requires TokenModel(free_rollout=True)")
        delta_pos, delta_vel, new_hidden = self.dynamics(positions, velocities, hidden)
        final_pos = positions + velocities * self.dt + delta_pos
        final_vel = velocities + delta_vel
        next_grid = rasterize_tokens(final_pos, final_vel, self.n, self.radius)
        return final_pos, final_vel, new_hidden, next_grid
