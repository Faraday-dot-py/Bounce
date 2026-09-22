import numpy as np
import torch

import bounce
from model.token_detect import centroid_near, find_token_positions


def _prob_channel(balls, n, radius):
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    return torch.from_numpy(np.array(G, dtype=np.float32)[:, :, bounce.PROB])


def test_centroid_near_recovers_isolated_ball_center():
    balls = [{"x": 10.5, "y": 7.5, "vx": 0.0, "vy": 0.0}]
    prob = _prob_channel(balls, 20, 0.75)
    result = centroid_near(prob, torch.tensor([10.0, 7.0]), radius=0.75)
    assert torch.allclose(result, torch.tensor([10.5, 7.5]), atol=0.05)


def test_centroid_near_returns_input_position_when_window_is_empty():
    prob = torch.zeros((20, 20))
    pos = torch.tensor([5.0, 5.0])
    result = centroid_near(prob, pos, radius=0.75)
    assert torch.equal(result, pos)


def test_find_token_positions_recovers_well_separated_balls():
    balls = [
        {"x": 5.0, "y": 5.0, "vx": 0.0, "vy": 0.0},
        {"x": 15.0, "y": 8.0, "vx": 0.0, "vy": 0.0},
        {"x": 8.0, "y": 16.0, "vx": 0.0, "vy": 0.0},
    ]
    prob = _prob_channel(balls, 20, 0.75)
    detected = find_token_positions(prob, radius=0.75)
    assert detected.shape[0] == 3

    expected = torch.tensor([[b["x"], b["y"]] for b in balls])
    for row in detected:
        dists = torch.norm(expected - row, dim=1)
        assert dists.min() < 0.2


def test_find_token_positions_merges_heavily_overlapping_balls():
    # Accepted edge case (design spec): balls closer together than one
    # ball's footprint are detected as a single token, not two.
    balls = [
        {"x": 10.0, "y": 10.0, "vx": 0.0, "vy": 0.0},
        {"x": 10.3, "y": 10.0, "vx": 0.0, "vy": 0.0},
    ]
    prob = _prob_channel(balls, 20, 0.75)
    detected = find_token_positions(prob, radius=0.75)
    assert detected.shape[0] < 2


def test_find_token_positions_handles_empty_grid():
    prob = torch.zeros((20, 20))
    detected = find_token_positions(prob, radius=0.75)
    assert detected.shape == (0, 2)
