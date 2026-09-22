import torch

from model.token_losses import token_grid_loss


def test_token_grid_loss_matches_manual_weighted_mse():
    pred = torch.zeros(3, 4, 4)
    target = torch.zeros(3, 4, 4)
    target[0] = 1.0  # PROB channel differs by 1.0 everywhere
    weights = torch.tensor([2.0, 0.5, 0.5])

    loss = token_grid_loss(pred, target, weights)
    expected = 2.0 * 1.0  # channel-0 MSE=1.0 * weight 2.0, other channels 0
    assert torch.allclose(loss, torch.tensor(expected))


def test_token_grid_loss_is_zero_for_identical_grids():
    grid = torch.rand(3, 5, 5)
    weights = torch.tensor([1.0, 0.1, 0.1])
    loss = token_grid_loss(grid, grid, weights)
    assert torch.allclose(loss, torch.tensor(0.0))
