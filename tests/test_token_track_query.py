import torch
import torch.nn as nn

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


def _grid_with_ball(n, x, y, radius=0.75):
    from model.token_rasterize import rasterize_tokens
    positions = torch.tensor([[x, y]])
    velocities = torch.tensor([[1.5, -0.5]])
    grid = rasterize_tokens(positions, velocities, n, radius)
    return grid[0], grid[1], grid[2]


def test_cross_attention_finds_mass_and_returns_centroid_near_position():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, radius=0.75, margin=1.0, max_expansions=6)
    # Zero cross_key so every cell's score is 0 regardless of its
    # content -- softmax attention is then uniform over the window,
    # isolating this test's actual target (window-gather/geometry
    # correctness) from cross_key/cross_value's untrained random init,
    # which would otherwise make obs_pos an arbitrary learned readout
    # with no reason to be near the true ball position before training.
    # A uniform weighting over a window centered on `position` averages
    # dx/dy to exactly 0 by symmetry, so obs_pos should land on
    # `position` itself, not just nearby.
    nn.init.zeros_(model.cross_key.weight)
    nn.init.zeros_(model.cross_key.bias)
    prob, vx, vy = _grid_with_ball(20, 10.0, 10.0)
    query_vec = torch.randn(8)
    readout, obs_pos = model._cross_attention_one(prob, vx, vy, torch.tensor([10.0, 10.0]), query_vec)
    assert readout.shape == (8,)
    assert torch.allclose(obs_pos, torch.tensor([10.0, 10.0]), atol=1e-4)


def test_cross_attention_empty_window_falls_back_to_zero_readout_and_unchanged_position():
    # No mass anywhere near this position, even after widening -- must
    # fall back to an all-zero readout and obs_pos == position, matching
    # centroid_near's own give-up convention (no mask-then-unmasked-retry
    # here, since there is no mask to begin with).
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, radius=0.75, margin=1.0, max_expansions=2)
    n = 20
    prob = torch.zeros(n, n)
    vx = torch.zeros(n, n)
    vy = torch.zeros(n, n)
    position = torch.tensor([10.0, 10.0])
    query_vec = torch.randn(8)
    readout, obs_pos = model._cross_attention_one(prob, vx, vy, position, query_vec)
    assert torch.allclose(readout, torch.zeros(8))
    assert torch.allclose(obs_pos, position)


def test_cross_attention_translation_invariant():
    # Same relative window content, shifted to a different absolute
    # position via torch.roll (cyclic, but the window never reaches the
    # wrap boundary at either position) -- readout must match, obs_pos
    # must shift by the same amount.
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, radius=0.75, margin=1.0, max_expansions=6)
    prob, vx, vy = _grid_with_ball(20, 10.0, 10.0)
    shift = 3
    prob_s = torch.roll(prob, shifts=(shift, shift), dims=(0, 1))
    vx_s = torch.roll(vx, shifts=(shift, shift), dims=(0, 1))
    vy_s = torch.roll(vy, shifts=(shift, shift), dims=(0, 1))
    query_vec = torch.randn(8)

    readout, obs_pos = model._cross_attention_one(prob, vx, vy, torch.tensor([10.0, 10.0]), query_vec)
    readout_s, obs_pos_s = model._cross_attention_one(
        prob_s, vx_s, vy_s, torch.tensor([10.0 + shift, 10.0 + shift]), query_vec
    )
    assert torch.allclose(readout, readout_s, atol=1e-5)
    assert torch.allclose(obs_pos_s - obs_pos, torch.tensor([float(shift), float(shift)]), atol=1e-5)
