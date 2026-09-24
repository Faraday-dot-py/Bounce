import torch

from model.token_losses import boundary_loss, token_grid_loss, token_state_loss, window_collapse_loss


def test_token_grid_loss_is_zero_for_identical_grids():
    grid = torch.rand(3, 5, 5)
    weights = torch.tensor([1.0, 0.1, 0.1])
    loss = token_grid_loss(grid, grid, grid, weights, peak_weight=0.0, mass_weight=0.0)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_token_grid_loss_penalizes_occupied_mismatch_more_than_background():
    # Two candidate predictions against the same target: one wrong at an
    # occupied cell, one wrong at a background cell by the same amount.
    # occupancy_weighted_mse must penalize the occupied-cell error more
    # (bg_weight < 1), unlike a plain per-pixel MSE which would treat
    # them identically -- this is the property the give-up degenerate
    # solution (job 2840) exploited when this loss was plain MSE.
    n = 6
    target = torch.zeros(3, n, n)
    target[0, 0, 0] = 1.0  # one occupied cell
    source = target.clone()
    weights = torch.tensor([1.0, 0.1, 0.1])

    pred_wrong_at_occupied = target.clone()
    pred_wrong_at_occupied[0, 0, 0] = 0.0  # misses the occupied cell

    pred_wrong_at_background = target.clone()
    pred_wrong_at_background[0, 3, 3] = 1.0  # false positive at background

    loss_occupied_miss = token_grid_loss(
        pred_wrong_at_occupied, target, source, weights, peak_weight=0.0, mass_weight=0.0
    )
    loss_background_miss = token_grid_loss(
        pred_wrong_at_background, target, source, weights, peak_weight=0.0, mass_weight=0.0
    )
    assert loss_occupied_miss > loss_background_miss


def test_token_grid_loss_give_up_prediction_is_not_free():
    # The specific degenerate solution job 2840 found: predict all-zero
    # (empty grid) regardless of where the target's ball actually is.
    # occupancy_weighted_mse's peak term must make this strictly worse
    # than a correctly-positioned prediction, even though an all-zero
    # prediction is "free" under a plain, unweighted per-pixel MSE once
    # occupancy is sparse.
    n = 10
    target = torch.zeros(3, n, n)
    target[0, 5, 5] = 0.8
    source = target.clone()
    weights = torch.tensor([1.0, 0.1, 0.1])

    give_up_pred = torch.zeros(3, n, n)
    correct_pred = target.clone()

    loss_give_up = token_grid_loss(give_up_pred, target, source, weights)
    loss_correct = token_grid_loss(correct_pred, target, source, weights)
    assert loss_give_up > loss_correct
    assert torch.allclose(loss_correct, torch.tensor(0.0), atol=1e-6)


def test_boundary_loss_empty_tokens_returns_zero():
    # A rare degenerate rollout sample (init_tokens detects zero balls at
    # frame 1) previously produced NaN here: `.mean()` on a zero-element
    # tensor is NaN in PyTorch, and 0.0 * nan is still nan -- this leaked
    # into training logs even at boundary_weight=0.0 (job 2857/v13,
    # docs/debugging/experiment-log.md). Confirmed harmless to trained
    # weights (backward through a zero-cardinality path never actually
    # multiplies a real nan into any parameter's gradient), but the loss
    # value itself should still be a real, poison-free zero.
    positions = torch.zeros((0, 2))
    loss = boundary_loss(positions, n=20)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_boundary_loss_is_zero_for_in_range_positions():
    n = 20
    positions = torch.tensor([[5.0, 5.0], [0.0, 19.9], [10.0, 10.0]])
    loss = boundary_loss(positions, n)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_boundary_loss_grows_with_distance_past_boundary():
    n = 20
    just_past = torch.tensor([[-0.5, 5.0]])
    far_past = torch.tensor([[-5.0, 5.0]])
    loss_just_past = boundary_loss(just_past, n)
    loss_far_past = boundary_loss(far_past, n)
    assert loss_just_past > 0
    assert loss_far_past > loss_just_past


def test_boundary_loss_penalizes_both_low_and_high_side():
    n = 20
    below = torch.tensor([[-3.0, 5.0]])
    above = torch.tensor([[23.0, 5.0]])
    loss_below = boundary_loss(below, n)
    loss_above = boundary_loss(above, n)
    assert loss_below > 0
    assert loss_above > 0
    assert torch.allclose(loss_below, loss_above, atol=1e-6)


def test_boundary_loss_respects_margin():
    n = 20
    positions = torch.tensor([[1.0, 5.0]])
    loss_no_margin = boundary_loss(positions, n, margin=0.0)
    loss_with_margin = boundary_loss(positions, n, margin=2.0)
    assert torch.allclose(loss_no_margin, torch.tensor(0.0), atol=1e-6)
    assert loss_with_margin > 0


def test_boundary_loss_gradient_flows_to_positions():
    n = 20
    positions = torch.tensor([[-3.0, 5.0]], requires_grad=True)
    loss = boundary_loss(positions, n)
    loss.backward()
    assert positions.grad is not None
    assert positions.grad[0, 0] != 0.0
    assert positions.grad[0, 1] == 0.0


def test_token_state_loss_is_zero_for_exact_match():
    final_pos = torch.tensor([[5.0, 3.0], [1.0, 1.0]])
    final_vel = torch.tensor([[0.5, -0.5], [0.0, 2.0]])
    target = {
        "x": torch.tensor([1.0, 5.0]),
        "y": torch.tensor([1.0, 3.0]),
        "vx": torch.tensor([0.0, 0.5]),
        "vy": torch.tensor([2.0, -0.5]),
    }
    match_idx = torch.tensor([1, 0])  # final_pos[0] <-> target index 1, final_pos[1] <-> target index 0
    loss = token_state_loss(final_pos, final_vel, target, match_idx)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_token_state_loss_grows_with_position_offset():
    target = {
        "x": torch.tensor([5.0]),
        "y": torch.tensor([5.0]),
        "vx": torch.tensor([0.0]),
        "vy": torch.tensor([0.0]),
    }
    match_idx = torch.tensor([0])
    final_vel = torch.tensor([[0.0, 0.0]])

    close = token_state_loss(torch.tensor([[5.1, 5.0]]), final_vel, target, match_idx)
    far = token_state_loss(torch.tensor([[8.0, 5.0]]), final_vel, target, match_idx)
    assert far > close > 0


def test_token_state_loss_no_free_pass_for_background_prediction():
    # The whole point of this loss: unlike token_grid_loss, predicting
    # "nothing" isn't an option here -- there's no background cell to
    # match to for free, every real token has a required numeric target.
    target = {
        "x": torch.tensor([5.0]),
        "y": torch.tensor([5.0]),
        "vx": torch.tensor([1.0]),
        "vy": torch.tensor([1.0]),
    }
    match_idx = torch.tensor([0])
    give_up_pos = torch.tensor([[0.0, 0.0]])
    give_up_vel = torch.tensor([[0.0, 0.0]])
    loss = token_state_loss(give_up_pos, give_up_vel, target, match_idx)
    assert loss > 0


def test_token_state_loss_empty_tokens_returns_zero():
    final_pos = torch.zeros((0, 2))
    final_vel = torch.zeros((0, 2))
    target = {"x": torch.zeros(0), "y": torch.zeros(0), "vx": torch.zeros(0), "vy": torch.zeros(0)}
    match_idx = torch.zeros((0,), dtype=torch.long)
    loss = token_state_loss(final_pos, final_vel, target, match_idx)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_token_state_loss_gradient_flows_to_positions_and_velocities():
    final_pos = torch.tensor([[3.0, 3.0]], requires_grad=True)
    final_vel = torch.tensor([[1.0, 1.0]], requires_grad=True)
    target = {"x": torch.tensor([5.0]), "y": torch.tensor([5.0]),
              "vx": torch.tensor([0.0]), "vy": torch.tensor([0.0])}
    match_idx = torch.tensor([0])
    loss = token_state_loss(final_pos, final_vel, target, match_idx)
    loss.backward()
    assert final_pos.grad is not None and (final_pos.grad != 0).all()
    assert final_vel.grad is not None and (final_vel.grad != 0).all()


def test_window_collapse_loss_is_zero_when_mass_present_at_token():
    n = 10
    prob = torch.zeros(n, n)
    prob[5, 5] = 0.8
    positions = torch.tensor([[5.0, 5.0]])
    loss = window_collapse_loss(prob, positions, radius=0.75, floor=0.3)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_window_collapse_loss_penalizes_empty_window():
    # Token claims to be at (5, 5) but the rasterized PROB channel has no
    # mass anywhere near it -- this is the give-up shortcut identified in
    # docs/debugging/experiment-log.md: token_state_loss never looks at
    # whether the token's own rasterization actually deposits detectable
    # mass at its tracked position, so a vanished token costs nothing extra.
    n = 10
    prob = torch.zeros(n, n)
    positions = torch.tensor([[5.0, 5.0]])
    loss = window_collapse_loss(prob, positions, radius=0.75, floor=0.3)
    assert loss > 0


def test_window_collapse_loss_grows_as_mass_shrinks():
    n = 10
    positions = torch.tensor([[5.0, 5.0]])
    prob_high = torch.zeros(n, n)
    prob_high[5, 5] = 0.6
    prob_low = torch.zeros(n, n)
    prob_low[5, 5] = 0.1
    loss_high = window_collapse_loss(prob_high, positions, radius=0.75, floor=0.3)
    loss_low = window_collapse_loss(prob_low, positions, radius=0.75, floor=0.3)
    assert loss_low > loss_high


def test_window_collapse_loss_ignores_mass_outside_window():
    # Mass exists in the grid, just nowhere near this token -- shouldn't
    # get credit for a different token's (or a stray) blob.
    n = 20
    positions = torch.tensor([[2.0, 2.0]])
    prob = torch.zeros(n, n)
    prob[15, 15] = 0.9
    loss = window_collapse_loss(prob, positions, radius=0.75, margin=1.0, floor=0.3)
    assert loss > 0


def test_window_collapse_loss_empty_tokens_returns_zero():
    prob = torch.zeros(5, 5)
    positions = torch.zeros((0, 2))
    loss = window_collapse_loss(prob, positions, radius=0.75)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_window_collapse_loss_gradient_flows_to_prob():
    n = 10
    prob = torch.zeros(n, n, requires_grad=True)
    positions = torch.tensor([[5.0, 5.0]])
    loss = window_collapse_loss(prob, positions, radius=0.75, floor=0.3)
    loss.backward()
    assert prob.grad is not None
    assert prob.grad[5, 5] != 0.0


def test_window_collapse_loss_territory_masks_out_neighbour_mass():
    # Token 0 sits right next to token 1's real mass but has none of its
    # own -- unmasked, its window would see token 1's mass and score
    # fine; territory-masked, it must still be penalized.
    n = 20
    prob = torch.zeros(n, n)
    prob[12, 10] = 0.9  # token 1's real mass
    positions = torch.tensor([[8.0, 10.0], [12.0, 10.0]])

    loss_unmasked = window_collapse_loss(prob, positions[:1], radius=1.5, margin=3.0, floor=0.3)
    loss_masked = window_collapse_loss(prob, positions[:1], radius=1.5, margin=3.0, floor=0.3,
                                        all_positions=positions, self_idx_offset=0)
    assert loss_unmasked == 0.0  # unmasked window at radius+margin=4.5 reaches x=12
    assert loss_masked > 0.0


def test_window_collapse_loss_territory_default_unaffected():
    n = 10
    prob = torch.zeros(n, n)
    prob[5, 5] = 0.8
    positions = torch.tensor([[5.0, 5.0]])
    baseline = window_collapse_loss(prob, positions, radius=0.75, floor=0.3)
    with_none = window_collapse_loss(prob, positions, radius=0.75, floor=0.3,
                                      all_positions=None)
    assert torch.allclose(baseline, with_none)


def test_token_state_loss_speed_weight_penalizes_shrunk_velocity_not_direction_only():
    target = {"x": torch.tensor([1.0]), "y": torch.tensor([1.0]), "vx": torch.tensor([3.0]), "vy": torch.tensor([4.0])}
    match_idx = torch.tensor([0])
    pos = torch.tensor([[1.0, 1.0]])
    shrunk = torch.tensor([[1.5, 2.0]])
    rotated = torch.tensor([[4.0, -3.0]])
    base_s = token_state_loss(pos, shrunk, target, match_idx, vel_weight=0.0)
    with_s = token_state_loss(pos, shrunk, target, match_idx, vel_weight=0.0, speed_weight=1.0)
    assert with_s > base_s
    assert torch.isclose(with_s - base_s, torch.tensor((2.5 - 5.0) ** 2), atol=1e-3)
    rot = token_state_loss(pos, rotated, target, match_idx, vel_weight=0.0, speed_weight=1.0)
    assert torch.isclose(rot, torch.tensor(0.0), atol=1e-3)


def test_contact_mask_flags_close_pairs_at_previous_or_target_step():
    from model.token_losses import contact_mask
    prev = {"x": torch.tensor([5.0, 5.9, 12.0]), "y": torch.tensor([5.0, 5.0, 3.0])}
    target = {"x": torch.tensor([5.0, 8.0, 12.0]), "y": torch.tensor([5.0, 5.0, 3.0])}
    mask = contact_mask(prev, target, torch.tensor([0, 1, 2]))
    assert mask.tolist() == [True, True, False]
    single = {"x": torch.tensor([1.0]), "y": torch.tensor([1.0])}
    assert contact_mask(single, single, torch.tensor([0])).tolist() == [False]


def test_token_state_loss_token_weights_scale_selected_tokens_only():
    target = {"x": torch.tensor([1.0, 4.0]), "y": torch.tensor([1.0, 4.0]),
              "vx": torch.zeros(2), "vy": torch.zeros(2)}
    match_idx = torch.tensor([0, 1])
    pos = torch.tensor([[2.0, 1.0], [4.0, 4.0]])
    vel = torch.zeros(2, 2)
    base = token_state_loss(pos, vel, target, match_idx)
    weighted = token_state_loss(pos, vel, target, match_idx, token_weights=torch.tensor([3.0, 1.0]))
    assert torch.isclose(weighted, base * 3.0)
    unweighted = token_state_loss(pos, vel, target, match_idx, token_weights=torch.tensor([1.0, 1.0]))
    assert torch.isclose(unweighted, base)
