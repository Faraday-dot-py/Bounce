import torch

from model.token_model import TokenModel
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

    new_pos, new_vel, new_hidden, pred_grid = model.step(positions, velocities, hidden, observed_frame)

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

    new_pos, new_vel, _, pred_grid = model.step(positions, velocities, hidden, observed_frame)
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

    new_pos, new_vel, _, _ = model.step(positions, velocities, hidden, observed_frame)

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

    new_pos, _, _, _ = model.step(positions, velocities, hidden, observed_frame)

    assert torch.allclose(new_pos, true_obs_pos, atol=0.1)
    assert not torch.allclose(new_pos, positions, atol=0.1)
