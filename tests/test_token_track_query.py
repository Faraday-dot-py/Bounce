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


def test_self_attention_single_token_still_carries_gradient_via_self_loop():
    # A single token has no radius-graph neighbors, only its self-loop --
    # a softmax group of size 1 whose weight is always exactly 1.0
    # regardless of score, so self_query/self_key get zero gradient there
    # (same documented property as TokenDynamics). But the group still
    # exists: self_value's output IS the pooled result (weight 1.0 times
    # its own value), and the output still depends on hidden state, so
    # this is not a dead path -- self_value and the input hidden state
    # must still carry gradient.
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=3.0)
    positions = torch.tensor([[5.0, 5.0]])
    velocities = torch.tensor([[1.0, -0.5]])
    hidden = torch.randn(1, 8, requires_grad=True)
    out = model._self_attention(positions, velocities, hidden)
    out.sum().backward()
    assert model.self_value.weight.grad is not None
    assert torch.any(model.self_value.weight.grad != 0.0)
    assert hidden.grad is not None
    assert torch.any(hidden.grad != 0.0)


def test_self_attention_return_weights_matches_pooled_output():
    # return_weights=True must be a pure addition -- the pooled attn_out
    # it returns alongside the weights must be identical to what
    # return_weights=False returns, and the returned (src, dst, weights)
    # must reproduce attn_out via the same index_add/normalize this
    # method does internally (proves the exposed weights are the real
    # ones used, not a second, possibly-drifted computation).
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=4.0)
    positions = torch.tensor([[5.0, 5.0], [6.0, 5.0], [5.5, 6.0]])
    velocities = torch.tensor([[1.0, -0.5], [0.25, 2.0], [-1.0, 0.0]])
    hidden = torch.randn(3, 8)

    plain_out = model._self_attention(positions, velocities, hidden)
    out, edge_weights = model._self_attention(positions, velocities, hidden, return_weights=True)
    assert torch.allclose(out, plain_out)

    src, dst, weights = edge_weights
    node_state = torch.cat([velocities, hidden], dim=-1)
    v = model.self_value(torch.cat([node_state[src], positions[src] - positions[dst]], dim=-1))
    reconstructed = torch.zeros(3, 8).index_add(0, dst, weights.unsqueeze(-1) * v)
    assert torch.allclose(reconstructed, out, atol=1e-5)


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
    # A uniform weighting over a window recovers the window's own
    # geometric center regardless of where `position` sits inside it, so
    # querying from an off-center position (10.3, not the ball's exact
    # 10.0) and asserting obs_pos == 10.0 (not 10.3) proves this is a
    # real read of window content -- not the empty-window fallback,
    # which would instead return obs_pos == position (10.3) unchanged.
    # (Ball, window and rounded query all still line up on cell 10, so
    # this isolates window-gather/geometry correctness the same way the
    # on-center version did, just without an ambiguity against the
    # fallback path.)
    nn.init.zeros_(model.cross_key.weight)
    nn.init.zeros_(model.cross_key.bias)
    prob, vx, vy = _grid_with_ball(20, 10.0, 10.0)
    query_vec = torch.randn(8)
    readout, obs_pos = model._cross_attention_one(prob, vx, vy, torch.tensor([10.3, 10.3]), query_vec)
    assert readout.shape == (8,)
    assert torch.any(readout != 0.0)
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


def test_cross_attention_return_weights_matches_plain_output_and_reports_actual_window():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, radius=0.75, margin=1.0, max_expansions=6)
    prob, vx, vy = _grid_with_ball(20, 10.0, 10.0)
    query_vec = torch.randn(8)

    readout, obs_pos = model._cross_attention_one(prob, vx, vy, torch.tensor([10.0, 10.0]), query_vec)
    readout_w, obs_pos_w, window_info = model._cross_attention_one(
        prob, vx, vy, torch.tensor([10.0, 10.0]), query_vec, return_weights=True
    )
    assert torch.allclose(readout, readout_w)
    assert torch.allclose(obs_pos, obs_pos_w)
    assert window_info is not None
    assert abs(window_info["weights"].sum().item() - 1.0) < 1e-5
    assert window_info["i_lo"] <= 10 <= window_info["i_hi"]
    assert window_info["j_lo"] <= 10 <= window_info["j_hi"]


def test_cross_attention_return_weights_reports_none_on_empty_window():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, radius=0.75, margin=1.0, max_expansions=2)
    n = 20
    prob = torch.zeros(n, n)
    vx = torch.zeros(n, n)
    vy = torch.zeros(n, n)
    position = torch.tensor([10.0, 10.0])
    query_vec = torch.randn(8)
    readout, obs_pos, window_info = model._cross_attention_one(
        prob, vx, vy, position, query_vec, return_weights=True
    )
    assert window_info is None
    assert torch.allclose(readout, torch.zeros(8))
    assert torch.allclose(obs_pos, position)


def _random_frame(n, num_balls, seed=4738):
    from model.token_rasterize import rasterize_tokens
    g = torch.Generator().manual_seed(seed)
    positions = torch.rand(num_balls, 2, generator=g) * (n - 4) + 2
    velocities = torch.randn(num_balls, 2, generator=g)
    return rasterize_tokens(positions, velocities, n, 0.75), positions, velocities


def test_forward_shapes():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=3.0)
    observed_frame, positions, velocities = _random_frame(20, 5)
    hidden = torch.zeros(5, 8)
    delta_pos, delta_vel, new_hidden, obs_pos = model(positions, velocities, hidden, observed_frame)
    assert delta_pos.shape == (5, 2)
    assert delta_vel.shape == (5, 2)
    assert new_hidden.shape == (5, 8)
    assert obs_pos.shape == (5, 2)


def test_delta_head_is_zero_at_init():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=3.0)
    observed_frame, positions, velocities = _random_frame(20, 4)
    hidden = torch.randn(4, 8)
    delta_pos, delta_vel, _, _ = model(positions, velocities, hidden, observed_frame)
    assert torch.allclose(delta_pos, torch.zeros_like(delta_pos))
    assert torch.allclose(delta_vel, torch.zeros_like(delta_vel))


def test_forward_gradients_flow_to_both_attention_stages():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=3.0)
    observed_frame, positions, velocities = _random_frame(20, 4)
    positions = positions.clone().requires_grad_(True)
    hidden = torch.randn(4, 8)
    delta_pos, delta_vel, new_hidden, _ = model(positions, velocities, hidden, observed_frame)
    new_hidden.sum().backward()
    assert model.self_query.weight.grad is not None
    assert torch.any(model.self_query.weight.grad != 0.0)
    assert model.cross_query.weight.grad is not None
    assert torch.any(model.cross_query.weight.grad != 0.0)


def test_forward_empty_token_set():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=3.0)
    n = 20
    observed_frame = torch.zeros(3, n, n)
    positions = torch.zeros(0, 2)
    velocities = torch.zeros(0, 2)
    hidden = torch.zeros(0, 8)
    delta_pos, delta_vel, new_hidden, obs_pos = model(positions, velocities, hidden, observed_frame)
    assert delta_pos.shape == (0, 2)
    assert delta_vel.shape == (0, 2)
    assert new_hidden.shape == (0, 8)
    assert obs_pos.shape == (0, 2)
    assert not torch.isnan(delta_pos).any()


def test_forward_permutation_equivariant():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=4.0)
    observed_frame, positions, velocities = _random_frame(20, 4)
    hidden = torch.randn(4, 8)
    perm = torch.tensor([2, 0, 3, 1])

    delta_pos, delta_vel, new_hidden, obs_pos = model(positions, velocities, hidden, observed_frame)
    delta_pos_p, delta_vel_p, new_hidden_p, obs_pos_p = model(
        positions[perm], velocities[perm], hidden[perm], observed_frame
    )

    assert torch.allclose(delta_pos[perm], delta_pos_p, atol=1e-5)
    assert torch.allclose(delta_vel[perm], delta_vel_p, atol=1e-5)
    assert torch.allclose(new_hidden[perm], new_hidden_p, atol=1e-5)
    assert torch.allclose(obs_pos[perm], obs_pos_p, atol=1e-5)


def test_forward_translation_invariant():
    # Shift positions and the observed frame together (cyclic roll,
    # tokens kept well clear of the wrap boundary before and after) --
    # delta_pos/delta_vel/new_hidden must be unchanged (obs_pos is
    # absolute and is expected to shift by the same amount).
    torch.manual_seed(4738)
    from model.token_rasterize import rasterize_tokens
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=4.0)
    n = 20
    g = torch.Generator().manual_seed(4738)
    positions = torch.rand(3, 2, generator=g) * 10 + 5.0  # kept in [5, 15] before AND after clamping below
    velocities = torch.randn(3, 2, generator=g)
    positions = torch.clamp(positions, 5.0, n - 5.0)
    observed_frame = rasterize_tokens(positions, velocities, n, 0.75)
    hidden = torch.randn(3, 8)
    shift = 2

    delta_pos, delta_vel, new_hidden, obs_pos = model(positions, velocities, hidden, observed_frame)

    shifted_frame = torch.roll(observed_frame, shifts=(shift, shift), dims=(1, 2))
    shifted_positions = positions + shift
    delta_pos_s, delta_vel_s, new_hidden_s, obs_pos_s = model(
        shifted_positions, velocities, hidden, shifted_frame
    )

    assert torch.allclose(delta_pos, delta_pos_s, atol=1e-4)
    assert torch.allclose(delta_vel, delta_vel_s, atol=1e-4)
    assert torch.allclose(new_hidden, new_hidden_s, atol=1e-4)
    assert torch.allclose(obs_pos_s - obs_pos, torch.full_like(obs_pos, float(shift)), atol=1e-4)
