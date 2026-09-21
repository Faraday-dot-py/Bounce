import torch
from model.losses import weighted_channel_mse


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
