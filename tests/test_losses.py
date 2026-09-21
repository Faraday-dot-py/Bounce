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
    weights = torch.tensor([1.0, 0.0, 0.0])

    pred_bg_err = target.clone()
    pred_bg_err[:, 0, 0, 0] = 5.0  # error only in a background pixel
    loss_bg = occupancy_weighted_mse(pred_bg_err, target, weights, bg_weight=0.0)
    assert torch.isclose(loss_bg, torch.tensor(0.0), atol=1e-6)

    pred_occ_err = target.clone()
    pred_occ_err[:, 0, 5, 5] = 5.0  # error only at the occupied pixel
    loss_occ = occupancy_weighted_mse(pred_occ_err, target, weights, bg_weight=0.0)
    assert loss_occ > 0.0


def test_bg_weight_only_discounts_prob_channel():
    target = torch.zeros(1, 3, 10, 10)
    target[:, 0, 5, 5] = 1.0  # occupied pixel; rest of grid is background

    pred_vel_err = target.clone()
    pred_vel_err[:, 1, 0, 0] = 5.0  # VX error at a background pixel
    weights = torch.tensor([0.0, 1.0, 0.0])
    loss_vel_bg = occupancy_weighted_mse(pred_vel_err, target, weights, bg_weight=0.05)
    loss_vel_uniform = weighted_channel_mse(pred_vel_err, target, weights)
    assert torch.isclose(loss_vel_bg, loss_vel_uniform)


def test_bg_weight_one_matches_weighted_channel_mse():
    pred = torch.randn(2, 3, 10, 10)
    target = torch.randn(2, 3, 10, 10)
    weights = torch.tensor([1.0, 0.1, 0.1])
    loss_occ = occupancy_weighted_mse(pred, target, weights, bg_weight=1.0)
    loss_uniform = weighted_channel_mse(pred, target, weights)
    assert torch.isclose(loss_occ, loss_uniform)
