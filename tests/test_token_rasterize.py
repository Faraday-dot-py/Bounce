import numpy as np
import torch

import bounce
from model.token_rasterize import rasterize_tokens


def _numpy_ground_truth(balls, n, radius):
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    return np.array(G, dtype=np.float32).transpose(2, 0, 1)


def test_rasterize_tokens_matches_numpy_ground_truth_non_overlapping():
    balls = [
        {"x": 5.0, "y": 5.0, "vx": 1.0, "vy": -2.0},
        {"x": 15.0, "y": 12.0, "vx": -0.5, "vy": 0.3},
    ]
    n, radius = 20, 0.75
    expected = _numpy_ground_truth(balls, n, radius)

    positions = torch.tensor([[b["x"], b["y"]] for b in balls])
    velocities = torch.tensor([[b["vx"], b["vy"]] for b in balls])
    actual = rasterize_tokens(positions, velocities, n, radius).numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_rasterize_tokens_matches_numpy_ground_truth_overlapping():
    balls = [
        {"x": 10.0, "y": 10.0, "vx": 2.0, "vy": 0.0},
        {"x": 10.6, "y": 10.0, "vx": -2.0, "vy": 0.0},
    ]
    n, radius = 20, 0.75
    expected = _numpy_ground_truth(balls, n, radius)

    positions = torch.tensor([[b["x"], b["y"]] for b in balls])
    velocities = torch.tensor([[b["vx"], b["vy"]] for b in balls])
    actual = rasterize_tokens(positions, velocities, n, radius).numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_rasterize_tokens_handles_zero_tokens():
    n, radius = 10, 0.75
    positions = torch.zeros((0, 2))
    velocities = torch.zeros((0, 2))
    out = rasterize_tokens(positions, velocities, n, radius)
    assert out.shape == (3, n, n)
    assert torch.all(out == 0.0)


def test_rasterize_tokens_is_differentiable():
    positions = torch.tensor([[1.3, 2.7]], requires_grad=True)
    velocities = torch.tensor([[1.0, 0.0]], requires_grad=True)
    out = rasterize_tokens(positions, velocities, 10, 1.5)
    out.sum().backward()
    assert positions.grad is not None
    assert torch.any(positions.grad != 0.0)
