import torch
from model.losses import weighted_channel_mse, occupancy_weighted_mse


def test_zero_loss_when_pred_equals_target():
    pred = torch.randn(2, 3, 10, 10)
    target = pred.clone()
    weights = torch.tensor([1.0, 0.1, 0.1])
    loss = weighted_channel_mse(pred, target, weights)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_channel_weighting_scales_contribution():
    pred = torch.zeros(1, 3, 10, 10)
    target = torch.zeros(1, 3, 10, 10)
    target[:, 0, :, :] = 1.0  # error only in channel 0
    weights_a = torch.tensor([1.0, 0.0, 0.0])
    weights_b = torch.tensor([2.0, 0.0, 0.0])
    loss_a = weighted_channel_mse(pred, target, weights_a)
    loss_b = weighted_channel_mse(pred, target, weights_b)
    assert torch.isclose(loss_b, loss_a * 2.0)


def test_bg_weight_zero_ignores_background_error():
    target = torch.zeros(1, 3, 10, 10)
    target[:, 0, 5, 5] = 1.0  # occupied pixel
    source = target.clone()
    weights = torch.tensor([1.0, 0.0, 0.0])

    pred_bg_err = target.clone()
    pred_bg_err[:, 0, 0, 0] = 5.0  # error only in a background pixel
    loss_bg = occupancy_weighted_mse(pred_bg_err, target, source, weights, bg_weight=0.0)
    assert torch.isclose(loss_bg, torch.tensor(0.0), atol=1e-6)

    pred_occ_err = target.clone()
    pred_occ_err[:, 0, 5, 5] = 5.0  # error only at the occupied pixel
    loss_occ = occupancy_weighted_mse(pred_occ_err, target, source, weights, bg_weight=0.0)
    assert loss_occ > 0.0


def test_bg_weight_only_discounts_prob_channel():
    target = torch.zeros(1, 3, 10, 10)
    target[:, 0, 5, 5] = 1.0  # occupied pixel; rest of grid is background
    source = target.clone()

    pred_vel_err = target.clone()
    pred_vel_err[:, 1, 0, 0] = 5.0  # VX error at a background pixel
    weights = torch.tensor([0.0, 1.0, 0.0])
    loss_vel_bg = occupancy_weighted_mse(pred_vel_err, target, source, weights, bg_weight=0.05)
    loss_vel_uniform = weighted_channel_mse(pred_vel_err, target, weights)
    assert torch.isclose(loss_vel_bg, loss_vel_uniform)


def test_bg_weight_one_matches_weighted_channel_mse():
    pred = torch.randn(2, 3, 10, 10)
    target = torch.randn(2, 3, 10, 10)
    source = torch.randn(2, 3, 10, 10)
    weights = torch.tensor([1.0, 0.1, 0.1])
    loss_occ = occupancy_weighted_mse(pred, target, source, weights, bg_weight=1.0)
    loss_uniform = weighted_channel_mse(pred, target, weights)
    assert torch.isclose(loss_occ, loss_uniform)


def test_background_error_less_diluted_than_grid_wide_mean():
    # Regression test for the root-cause fix: previously, background PROB
    # error was averaged over the *entire* grid (occ+bg combined), diluting
    # a widespread small background error (e.g. a growing periodic artifact)
    # into near-invisibility once background pixels dominate the grid (as
    # they do in practice: ~10% occupancy). The fixed formula normalizes
    # background error by the background pixel count itself, so the same
    # absolute error produces a much larger, size-appropriate contribution.
    n = 50
    occ_frac = 0.1
    num_occ = int(occ_frac * n * n)
    target = torch.zeros(1, 3, n, n)
    flat = target[0, 0].view(-1)
    flat[:num_occ] = 1.0
    source = target.clone()
    pred = target.clone()
    pred[:, 0, :, :] += 0.05  # small widespread error, e.g. a periodic ripple
    pred_flat = pred[0, 0].view(-1)
    pred_flat[:num_occ] = 1.0  # occupied cells predicted exactly

    weights = torch.tensor([1.0, 0.0, 0.0])
    bg_weight = 0.05
    fixed_loss = occupancy_weighted_mse(pred, target, source, weights, bg_weight=bg_weight)

    occ_count = num_occ
    bg_count = n * n - num_occ
    bg_sum_err = bg_count * (0.05 ** 2)
    old_grid_wide_loss = (bg_weight * bg_sum_err) / (n * n)
    new_formula_expected = (bg_weight * bg_sum_err) / (occ_count + bg_weight * bg_count)

    assert torch.isclose(fixed_loss, torch.tensor(new_formula_expected), rtol=1e-4)
    assert fixed_loss > old_grid_wide_loss * 5


def test_vacated_cell_keeps_full_weight_via_source_occupancy():
    source = torch.zeros(1, 3, 10, 10)
    source[:, 0, 5, 5] = 1.0  # occupied at t
    target = torch.zeros(1, 3, 10, 10)  # empty at t+1 (vacated)
    weights = torch.tensor([1.0, 0.0, 0.0])

    pred_correct = target.clone()  # correctly predicts the cell clears
    pred_wrong = target.clone()
    pred_wrong[:, 0, 5, 5] = 5.0  # incorrectly predicts mass stayed

    loss_correct = occupancy_weighted_mse(pred_correct, target, source, weights, bg_weight=0.0)
    loss_wrong = occupancy_weighted_mse(pred_wrong, target, source, weights, bg_weight=0.0)
    assert torch.isclose(loss_correct, torch.tensor(0.0), atol=1e-6)
    assert loss_wrong > 0.0
