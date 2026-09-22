import torch

from model.token_net import TokenDynamics


def test_forward_shapes():
    torch.manual_seed(4738)
    model = TokenDynamics(hidden_dim=8, neighbor_radius=3.0)
    positions = torch.rand(5, 2) * 20
    velocities = torch.randn(5, 2)
    hidden = torch.zeros(5, 8)
    delta_pos, delta_vel, new_hidden = model(positions, velocities, hidden)
    assert delta_pos.shape == (5, 2)
    assert delta_vel.shape == (5, 2)
    assert new_hidden.shape == (5, 8)


def test_delta_head_is_zero_at_init():
    torch.manual_seed(4738)
    model = TokenDynamics(hidden_dim=8, neighbor_radius=3.0)
    positions = torch.rand(4, 2) * 20
    velocities = torch.randn(4, 2)
    hidden = torch.randn(4, 8)
    delta_pos, delta_vel, _ = model(positions, velocities, hidden)
    assert torch.allclose(delta_pos, torch.zeros_like(delta_pos))
    assert torch.allclose(delta_vel, torch.zeros_like(delta_vel))


def test_gradients_flow_to_attention_parameters():
    torch.manual_seed(4738)
    model = TokenDynamics(hidden_dim=8, neighbor_radius=3.0)
    positions = torch.tensor([[5.0, 5.0], [6.0, 5.0]], requires_grad=True)
    velocities = torch.zeros(2, 2)
    hidden = torch.zeros(2, 8)
    delta_pos, delta_vel, new_hidden = model(positions, velocities, hidden)
    loss = new_hidden.sum()
    loss.backward()
    assert model.query.weight.grad is not None
    assert torch.any(model.query.weight.grad != 0.0)


def test_isolated_token_unaffected_by_distant_tokens():
    # Locality property required for train-small/tile-large transfer:
    # a token's output must not depend on tokens outside neighbor_radius.
    torch.manual_seed(4738)
    model = TokenDynamics(hidden_dim=8, neighbor_radius=3.0)
    hidden = torch.randn(3, 8)
    positions_a = torch.tensor([[5.0, 5.0], [5.5, 5.0], [50.0, 50.0]])
    positions_b = torch.tensor([[5.0, 5.0], [5.5, 5.0], [90.0, 90.0]])
    velocities = torch.zeros(3, 2)

    _, _, hidden_a = model(positions_a, velocities, hidden)
    _, _, hidden_b = model(positions_b, velocities, hidden)

    assert torch.allclose(hidden_a[0], hidden_b[0], atol=1e-6)
    assert torch.allclose(hidden_a[1], hidden_b[1], atol=1e-6)
