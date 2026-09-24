import torch

from model.token_model import TokenModel, _assign_velocity_pairs
from model.token_rasterize import rasterize_tokens


def test_init_tokens_recovers_position_and_velocity():
    n, radius, dt = 20, 1.5, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt)
    pos0 = torch.tensor([[5.0, 5.0]])
    vel = torch.tensor([[2.0, -1.0]])
    pos1 = pos0 + vel * dt
    frame0 = rasterize_tokens(pos0, vel, n, radius)
    frame1 = rasterize_tokens(pos1, vel, n, radius)

    positions, velocities, hidden = model.init_tokens(frame0, frame1)
    assert positions.shape == (1, 2)
    assert torch.allclose(positions[0], pos1[0], atol=0.1)
    assert torch.allclose(velocities[0], vel[0], atol=0.3)
    assert hidden.shape == (1, model.dynamics.hidden_dim)


def test_step_at_init_coasts_at_constant_velocity_when_observation_weight_is_zero():
    torch.manual_seed(4738)
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, observation_weight=0.0)
    positions = torch.tensor([[5.0, 5.0]])
    velocities = torch.tensor([[2.0, -1.0]])
    hidden = torch.zeros(1, model.dynamics.hidden_dim)
    observed_frame = torch.zeros(3, n, n)  # unused when observation_weight=0.0

    new_pos, new_vel, new_hidden, pred_grid, _ = model.step(positions, velocities, hidden, observed_frame)

    assert torch.allclose(new_pos, positions + velocities * dt, atol=1e-5)
    assert torch.allclose(new_vel, velocities, atol=1e-5)
    assert pred_grid.shape == (3, n, n)


def test_step_output_grid_matches_direct_rasterization():
    torch.manual_seed(4738)
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, observation_weight=0.0)
    positions = torch.tensor([[5.0, 5.0]])
    velocities = torch.tensor([[2.0, -1.0]])
    hidden = torch.zeros(1, model.dynamics.hidden_dim)
    observed_frame = torch.zeros(3, n, n)

    new_pos, new_vel, _, pred_grid, _ = model.step(positions, velocities, hidden, observed_frame)
    expected_grid = rasterize_tokens(new_pos, new_vel, n, radius)
    assert torch.allclose(pred_grid, expected_grid)


def test_occluding_tokens_ignore_observation_even_at_full_observation_weight():
    torch.manual_seed(4738)
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, observation_weight=1.0)
    # two tokens within 2*radius -> both flagged occluding after their
    # (zero-init, coast-only) predicted step
    positions = torch.tensor([[10.0, 10.0], [10.6, 10.0]])
    velocities = torch.zeros(2, 2)
    hidden = torch.zeros(2, model.dynamics.hidden_dim)
    observed_frame = torch.rand(3, n, n)  # arbitrary/irrelevant if gate works

    new_pos, new_vel, _, _, _ = model.step(positions, velocities, hidden, observed_frame)

    assert torch.allclose(new_pos, positions, atol=1e-5)  # coast (velocity=0), obs ignored


def test_step_blends_toward_observation_for_non_occluding_token():
    torch.manual_seed(4738)
    n, radius, dt = 20, 1.5, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, observation_weight=1.0)
    # zero velocity -> predicted_pos == positions exactly (delta_pos is
    # zero-init); the observed ball sits elsewhere, so a genuine blend
    # must pull final_pos away from the stale `positions` input.
    positions = torch.tensor([[10.0, 10.0]])
    velocities = torch.zeros(1, 2)
    hidden = torch.zeros(1, model.dynamics.hidden_dim)
    true_obs_pos = torch.tensor([[10.6, 10.0]])
    observed_frame = rasterize_tokens(true_obs_pos, torch.zeros(1, 2), n, radius)

    new_pos, _, _, _, _ = model.step(positions, velocities, hidden, observed_frame)

    assert torch.allclose(new_pos, true_obs_pos, atol=0.1)
    assert not torch.allclose(new_pos, positions, atol=0.1)


def test_isolated_token_tracks_its_velocity_under_default_observation_weight():
    # Regression for the time-alignment bug: `observed_frame` is the grid
    # at the same time as the input positions, so blending it into an
    # already-advanced prediction (and differencing it against the input
    # position to fake a velocity) drove tracked velocity toward zero
    # every step. A single isolated token fed its own true frame must
    # advance by exactly velocity*dt per step at the shipped default
    # observation_weight, not freeze or oscillate.
    torch.manual_seed(4738)
    # radius=1.5 (the fixture radius the other observation tests in this
    # file use) so centroid_near reads the true centre accurately; at the
    # production radius=0.75 a ball covers too few cells for an unbiased
    # centroid, an accepted separate limitation that would otherwise blur
    # the timing property under test here.
    n, radius, dt = 20, 1.5, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt)
    assert model.observation_weight == 0.5

    velocity = torch.tensor([[2.0, 1.0]])
    true_pos = torch.tensor([[5.0, 5.0]])
    positions = true_pos.clone()
    velocities = velocity.clone()
    hidden = torch.zeros(1, model.dynamics.hidden_dim)

    for _ in range(6):
        observed_frame = rasterize_tokens(true_pos, velocity, n, radius)
        previous = positions
        positions, velocities, hidden, _, _ = model.step(positions, velocities, hidden, observed_frame)
        true_pos = true_pos + velocity * dt
        assert torch.allclose(positions - previous, velocity * dt, atol=0.05)
        assert torch.allclose(velocities, velocity, atol=1e-5)

    assert torch.allclose(positions, true_pos, atol=0.05)


def test_gate_suppresses_observation_inside_the_detection_window_band():
    # 2.0 cells apart at radius=0.75 is outside the physical contact
    # threshold (2*radius = 1.5) but inside centroid_near's window reach,
    # so the observation is contaminated and must be gated off.
    torch.manual_seed(4738)
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, observation_weight=1.0)
    positions = torch.tensor([[10.0, 10.0], [12.0, 10.0]])
    velocities = torch.zeros(2, 2)
    hidden = torch.zeros(2, model.dynamics.hidden_dim)
    observed_frame = torch.rand(3, n, n)  # arbitrary/irrelevant if gate works

    new_pos, _, _, _, _ = model.step(positions, velocities, hidden, observed_frame)

    assert torch.allclose(new_pos, positions, atol=1e-5)


def test_step_velocity_correction_blends_toward_two_frame_observed_velocity():
    # Isolated token (no occlusion), true velocity [1, 0] but the tracked
    # `velocities` input is wrong ([0, 0], as if drifted); with
    # velocity_weight=1.0 and prev_obs_pos one dt behind the current
    # observation, the corrected velocity must move toward the
    # finite-difference of the two observed positions, not stay at the
    # wrong input value.
    torch.manual_seed(4738)
    n, radius, dt = 20, 1.5, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, observation_weight=1.0, velocity_weight=1.0)
    true_vel = torch.tensor([[1.0, 0.0]])
    positions = torch.tensor([[10.0, 10.0]])
    velocities = torch.zeros(1, 2)
    hidden = torch.zeros(1, model.dynamics.hidden_dim)
    prev_obs_pos = positions - true_vel * dt
    observed_frame = rasterize_tokens(positions, true_vel, n, radius)

    _, new_vel, _, _, obs_pos = model.step(positions, velocities, hidden, observed_frame, prev_obs_pos)

    assert torch.allclose(obs_pos, positions, atol=0.1)
    assert torch.allclose(new_vel, true_vel, atol=0.2)


def test_step_velocity_correction_off_by_default_and_without_prev_obs_pos():
    torch.manual_seed(4738)
    n, radius, dt = 20, 1.5, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, observation_weight=1.0)
    assert model.velocity_weight == 0.0
    positions = torch.tensor([[10.0, 10.0]])
    velocities = torch.tensor([[0.0, 0.0]])
    hidden = torch.zeros(1, model.dynamics.hidden_dim)
    observed_frame = rasterize_tokens(positions, torch.tensor([[1.0, 0.0]]), n, radius)

    # No prev_obs_pos passed at all -- must not error, and velocity must
    # be unaffected (velocity_weight=0.0 default).
    _, new_vel, _, _, _ = model.step(positions, velocities, hidden, observed_frame)
    assert torch.allclose(new_vel, velocities, atol=1e-5)


def test_init_tokens_zeros_velocity_for_implausible_nearest_match():
    n, radius, dt = 20, 1.5, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt)
    zero_vel = torch.zeros(1, 2)
    frame0 = rasterize_tokens(torch.tensor([[5.0, 5.0]]), zero_vel, n, radius)
    # frame1 keeps the real ball (small, plausible displacement) and adds a
    # spurious detection far away, whose only possible match is the ball at
    # (5, 5) roughly 14 cells off -- ~94 cells/s, physically impossible.
    real = torch.tensor([[5.3, 5.15]])
    spurious = torch.tensor([[15.0, 15.0]])
    frame1 = rasterize_tokens(torch.cat([real, spurious]), torch.zeros(2, 2), n, radius)

    positions, velocities, _ = model.init_tokens(frame0, frame1)
    assert positions.shape[0] == 2
    far = torch.norm(positions - torch.tensor([[15.0, 15.0]]), dim=1).argmin()
    near = 1 - far
    assert torch.allclose(velocities[far], torch.zeros(2), atol=1e-6)
    assert torch.norm(velocities[near]) > 0.5


def test_assign_velocity_pairs_gives_unique_match_when_frame0_balls_are_close():
    # Two frame0 candidates 0.3 cells apart; independent per-row argmin
    # would let both frame1 rows claim column 0 (seed 4750's duplicate-
    # match bug in docs/debugging/experiment-log.md), silently dropping
    # column 1's real partner. Optimal one-to-one assignment must not
    # reuse a column across rows.
    dists = torch.tensor([
        [0.5, 0.8],
        [0.6, 0.9],
    ])
    nearest, _, _ = _assign_velocity_pairs(dists)
    assert nearest[0] != nearest[1]
    assert set(nearest.tolist()) == {0, 1}


def test_assign_velocity_pairs_rejects_ambiguous_near_tie():
    # Row 0's best (col 0, dist 1.0) and second-best (col 1, dist 1.05)
    # are within the ambiguity margin -- close enough that an optimal
    # assignment can still pick the wrong physical ball (seed 4752's
    # smooth-divergence case). Must be flagged unkept even though it's
    # a clean unique assignment.
    dists = torch.tensor([[1.0, 1.05, 9.0]])
    _, _, keep = _assign_velocity_pairs(dists, ambiguity_margin=0.5)
    assert not bool(keep[0])


def test_assign_velocity_pairs_keeps_unambiguous_match():
    dists = torch.tensor([[1.0, 9.0, 9.0]])
    _, _, keep = _assign_velocity_pairs(dists, ambiguity_margin=0.5)
    assert bool(keep[0])


def test_territory_masking_defaults_to_off():
    # Masking must be opt-in: TokenModel() with no explicit
    # territory_masking must reproduce pre-fix inference behavior exactly
    # (see docs/debugging/experiment-log.md's v14 final review -- turning
    # masking on by default silently changed inference for every existing
    # checkpoint, e.g. v9's own dropout count 6 -> 8, even though those
    # checkpoints were trained without it).
    model = TokenModel(n=20, radius=0.75, dt=0.15)
    assert model.territory_masking is False


def test_step_steals_neighbours_mass_by_default_when_own_mass_is_gone():
    # Reproduces the pre-fix (and now default, territory_masking=False)
    # behavior directly: with masking off, an unmasked widened search
    # still grabs a nearby token's mass when this token's own ball has
    # vanished from the frame -- this is the "theft" v13/v14 were about,
    # but it is also the ONLY recovery path an unmasked model has, which
    # is exactly why it must remain the default (see
    # test_step_territory_masking_falls_back_when_own_mass_is_gone for
    # the masked-but-still-recovers version).
    torch.manual_seed(4738)
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, observation_weight=1.0)
    assert model.territory_masking is False
    positions = torch.tensor([[9.0, 10.0], [13.0, 10.0]])  # 4.0 apart -> not occluding (threshold 3.5)
    velocities = torch.zeros(2, 2)
    hidden = torch.zeros(2, model.dynamics.hidden_dim)
    observed_frame = rasterize_tokens(torch.tensor([[13.0, 10.0]]), torch.zeros(1, 2), n, radius)

    _, _, _, _, obs_pos = model.step(positions, velocities, hidden, observed_frame)

    assert torch.allclose(obs_pos[0], torch.tensor([13.0, 10.0]), atol=0.5)


def test_step_territory_masking_prevents_theft_when_own_mass_exists():
    # With territory_masking=True and token 0's own ball actually present
    # (not vanished), masking must still block a nearby token's mass from
    # being read -- the fallback only engages when a token's own
    # territory is completely empty (see
    # test_centroid_near_still_masks_when_own_territory_has_some_mass for
    # the centroid_near-level version of this same property).
    torch.manual_seed(4738)
    n, radius, dt = 20, 1.5, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, observation_weight=1.0, territory_masking=True)
    positions = torch.tensor([[8.0, 10.0], [12.0, 10.0]])  # 4.0 apart -> not occluding (threshold 5.0)
    velocities = torch.zeros(2, 2)
    hidden = torch.zeros(2, model.dynamics.hidden_dim)
    observed_frame = rasterize_tokens(positions, torch.zeros(2, 2), n, radius)

    _, _, _, _, obs_pos = model.step(positions, velocities, hidden, observed_frame)

    assert torch.allclose(obs_pos[0], torch.tensor([8.0, 10.0]), atol=0.3)


def test_step_territory_masking_falls_back_when_own_mass_is_gone():
    # With territory_masking=True but token 0's own ball genuinely
    # vanished from the frame, the fix (unmasked fallback in
    # centroid_near) must still recover it via token 1's mass, exactly
    # like the default-off case -- masking must not reintroduce
    # permanent loss for a token with nothing of its own left to find.
    torch.manual_seed(4738)
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, observation_weight=1.0, territory_masking=True)
    positions = torch.tensor([[9.0, 10.0], [13.0, 10.0]])
    velocities = torch.zeros(2, 2)
    hidden = torch.zeros(2, model.dynamics.hidden_dim)
    observed_frame = rasterize_tokens(torch.tensor([[13.0, 10.0]]), torch.zeros(1, 2), n, radius)

    _, _, _, _, obs_pos = model.step(positions, velocities, hidden, observed_frame)

    assert torch.allclose(obs_pos[0], torch.tensor([13.0, 10.0]), atol=0.5)


def test_occluding_tokens_still_ignore_observation_with_territory_masking():
    torch.manual_seed(4738)
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, observation_weight=1.0, territory_masking=True)
    positions = torch.tensor([[10.0, 10.0], [10.6, 10.0]])
    velocities = torch.zeros(2, 2)
    hidden = torch.zeros(2, model.dynamics.hidden_dim)
    observed_frame = torch.rand(3, n, n)

    new_pos, new_vel, _, _, _ = model.step(positions, velocities, hidden, observed_frame)
    assert torch.allclose(new_pos, positions, atol=1e-5)


def test_token_model_defaults_to_widened_give_up_search():
    model = TokenModel(n=20, radius=0.75, dt=0.15)
    assert model.max_expansions == 6


def test_step_recovers_drifted_token_with_widened_search():
    # A token has drifted 6 cells from its ball -- beyond the old
    # max_expansions=3's reach at radius=0.75/margin=1.0 (base_half=2,
    # +1 cell/expansion -> max half=5, misses a 6-cell drift by 1) but
    # within the widened default (max_expansions=6 -> max half=8). With
    # the widened default it must recover the ball via observation
    # correction rather than giving up.
    torch.manual_seed(4738)
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, observation_weight=1.0)
    true_ball_pos = torch.tensor([[10.0, 10.0]])
    drifted_position = torch.tensor([[16.0, 10.0]])  # 6 cells off
    velocities = torch.zeros(1, 2)
    hidden = torch.zeros(1, model.dynamics.hidden_dim)
    observed_frame = rasterize_tokens(true_ball_pos, torch.zeros(1, 2), n, radius)

    _, _, _, _, obs_pos = model.step(drifted_position, velocities, hidden, observed_frame)

    assert torch.allclose(obs_pos[0], true_ball_pos[0], atol=0.5)


def test_init_tokens_avoids_duplicate_frame0_match():
    n, radius, dt = 20, 1.5, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt)
    # Both frame1 balls end up much nearer to frame0's second detection
    # (16, 5) than to its first (1, 5) -- plain per-row argmin would match
    # both frame1 tokens to that same frame0 detection, silently
    # dropping (1, 5) as anyone's partner. Detections stay individually
    # resolvable (balls kept well apart so peak detection itself -- a
    # separate, undiagnosed issue at close range -- doesn't interfere).
    pos0 = torch.tensor([[1.0, 5.0], [16.0, 5.0]])
    pos1 = torch.tensor([[13.0, 5.0], [17.0, 5.0]])
    zero_vel = torch.zeros(2, 2)
    frame0 = rasterize_tokens(pos0, zero_vel, n, radius)
    frame1 = rasterize_tokens(pos1, zero_vel, n, radius)

    positions, velocities, _ = model.init_tokens(frame0, frame1)
    assert positions.shape[0] == 2
    assert torch.isfinite(velocities).all()


def test_track_query_step_returns_same_five_tuple_shape():
    torch.manual_seed(4738)
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, hidden_dim=8, track_query=True)
    positions = torch.tensor([[10.0, 10.0], [12.0, 8.0]])
    velocities = torch.tensor([[1.0, -0.5], [0.0, 1.0]])
    hidden = torch.zeros(2, model.dynamics.hidden_dim)
    observed_frame = rasterize_tokens(positions, velocities, n, radius)

    new_pos, new_vel, new_hidden, pred_grid, obs_pos = model.step(positions, velocities, hidden, observed_frame)

    assert new_pos.shape == (2, 2)
    assert new_vel.shape == (2, 2)
    assert new_hidden.shape == (2, model.dynamics.hidden_dim)
    assert pred_grid.shape == (3, n, n)
    assert obs_pos.shape == (2, 2)
    assert not torch.isnan(new_pos).any()


def test_track_query_step_coasts_at_constant_velocity_when_delta_head_is_zero_init():
    torch.manual_seed(4738)
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, hidden_dim=8, track_query=True)
    positions = torch.tensor([[10.0, 10.0]])
    velocities = torch.tensor([[2.0, -1.0]])
    hidden = torch.zeros(1, model.dynamics.hidden_dim)
    observed_frame = rasterize_tokens(positions, velocities, n, radius)

    new_pos, new_vel, _, _, _ = model.step(positions, velocities, hidden, observed_frame)

    assert torch.allclose(new_pos, positions + velocities * dt, atol=1e-5)
    assert torch.allclose(new_vel, velocities, atol=1e-5)


def test_track_query_model_uses_track_query_dynamics():
    from model.token_track_query import TrackQueryDynamics
    model = TokenModel(n=20, radius=0.75, dt=0.15, track_query=True)
    assert isinstance(model.dynamics, TrackQueryDynamics)


def test_default_model_still_uses_token_dynamics():
    from model.token_net import TokenDynamics
    model = TokenModel(n=20, radius=0.75, dt=0.15)
    assert isinstance(model.dynamics, TokenDynamics)
