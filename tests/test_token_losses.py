import torch

from model.token_losses import token_grid_loss


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
