import torch

from model.token_track_query import TrackQueryDynamics


def test_self_attention_shape():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=3.0)
    positions = torch.rand(5, 2) * 20
    velocities = torch.randn(5, 2)
    hidden = torch.zeros(5, 8)
    out = model._self_attention(positions, velocities, hidden)
    assert out.shape == (5, 8)


def test_self_attention_gradients_flow_to_attention_parameters():
    # Two tokens (not one): a lone token's only edge is its self-loop,
    # forming a softmax group of size 1 whose weight is always exactly
    # 1.0 regardless of score -- zero gradient to query/key by
    # construction (same documented property as TokenDynamics, whose own
    # equivalent test also uses >=2 tokens for this reason).
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=3.0)
    positions = torch.tensor([[5.0, 5.0], [6.0, 5.0]], requires_grad=True)
    velocities = torch.tensor([[1.0, -0.5], [0.25, 2.0]])
    hidden = torch.randn(2, 8)
    out = model._self_attention(positions, velocities, hidden)
    out.sum().backward()
    assert model.self_query.weight.grad is not None
    assert torch.any(model.self_query.weight.grad != 0.0)


def test_self_attention_isolated_token_unaffected_by_distant_tokens():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=3.0)
    hidden = torch.randn(3, 8)
    positions_a = torch.tensor([[5.0, 5.0], [5.5, 5.0], [50.0, 50.0]])
    positions_b = torch.tensor([[5.0, 5.0], [5.5, 5.0], [90.0, 90.0]])
    velocities = torch.zeros(3, 2)

    out_a = model._self_attention(positions_a, velocities, hidden)
    out_b = model._self_attention(positions_b, velocities, hidden)

    assert torch.allclose(out_a[0], out_b[0], atol=1e-6)
    assert torch.allclose(out_a[1], out_b[1], atol=1e-6)


def test_self_attention_translation_invariant():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=3.0)
    positions = torch.tensor([[5.0, 5.0], [6.0, 5.5], [5.2, 7.0]])
    velocities = torch.tensor([[1.0, -0.5], [-0.3, 0.8], [0.0, 2.0]])
    hidden = torch.randn(3, 8)

    base = model._self_attention(positions, velocities, hidden)
    shifted = model._self_attention(positions + 500.0, velocities, hidden)

    # atol=1e-4, not 1e-6: float32 has ~1.2e-7 relative precision, so
    # subtracting two ~505-magnitude positions to recover a small
    # rel_pos already carries ~6e-5 absolute error -- tighter than that
    # bounds against float32 itself, not against the invariance property
    # under test.
    assert torch.allclose(base, shifted, atol=1e-4)
