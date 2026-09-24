import torch

from scripts.eval_free_rollout import count_identity_swaps, position_errors


def test_position_errors_zero_for_perfect_track():
    T, B = 4, 3
    gt_x = torch.rand(T, B) * 10
    gt_y = torch.rand(T, B) * 10
    pred = torch.stack([gt_x, gt_y], dim=-1)
    err = position_errors(pred, torch.arange(B), gt_x, gt_y)
    assert err.shape == (T,) and torch.allclose(err, torch.zeros(T), atol=1e-6)


def test_position_errors_respects_token_to_ball_permutation():
    gt_x = torch.tensor([[0.0, 5.0]])
    gt_y = torch.tensor([[0.0, 5.0]])
    pred = torch.tensor([[[5.0, 5.0], [0.0, 0.0]]])
    err = position_errors(pred, torch.tensor([1, 0]), gt_x, gt_y)
    assert torch.allclose(err, torch.zeros(1), atol=1e-6)


def test_count_identity_swaps():
    gt_x = torch.tensor([[0.0, 5.0], [0.0, 5.0]])
    gt_y = torch.tensor([[0.0, 5.0], [0.0, 5.0]])
    pred_ok = torch.tensor([[[0.0, 0.0], [5.0, 5.0]], [[0.1, 0.0], [5.0, 5.1]]])
    assert count_identity_swaps(pred_ok, gt_x, gt_y, torch.tensor([0, 1])) == 0
    pred_swapped = torch.tensor([[[0.0, 0.0], [5.0, 5.0]], [[5.0, 5.0], [0.0, 0.0]]])
    assert count_identity_swaps(pred_swapped, gt_x, gt_y, torch.tensor([0, 1])) == 2


def test_helpers_handle_zero_tokens():
    pred = torch.zeros(3, 0, 2)
    err = position_errors(pred, torch.zeros(0, dtype=torch.long), torch.zeros(3, 2), torch.zeros(3, 2))
    assert err.shape == (3,) and torch.isfinite(err).all()
    assert count_identity_swaps(pred, torch.zeros(3, 2), torch.zeros(3, 2), torch.zeros(0, dtype=torch.long)) == 0
