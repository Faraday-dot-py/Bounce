import torch

from model.net import BounceNextFrameModel
from model.evaluate import held_out_loss, rollout_divergence, generalization_check


def test_held_out_loss_is_finite_nonnegative():
    torch.manual_seed(4738)
    model = BounceNextFrameModel(channels=16, depth=2)
    weights = torch.tensor([1.0, 0.1, 0.1])
    loss = held_out_loss(model, n=20, ball_range=(3, 6), num_samples=3, seed=4738, weights=weights)
    assert loss >= 0.0
    assert loss == loss  # not NaN


def test_rollout_divergence_returns_one_value_per_step():
    torch.manual_seed(4738)
    model = BounceNextFrameModel(channels=16, depth=2)
    divergence = rollout_divergence(model, n=20, num_balls=5, seed=4738, num_steps=4)
    assert len(divergence) == 4
    assert all(d >= 0.0 for d in divergence)


def test_generalization_check_runs_at_higher_ball_count():
    torch.manual_seed(4738)
    model = BounceNextFrameModel(channels=16, depth=2)
    weights = torch.tensor([1.0, 0.1, 0.1])
    loss = generalization_check(model, n=20, ball_count=40, num_samples=2, seed=4738, weights=weights)
    assert loss >= 0.0
