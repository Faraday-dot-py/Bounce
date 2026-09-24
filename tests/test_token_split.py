import numpy as np
import torch

import bounce
from model.token_model import TokenModel
from model.token_split import detect_balls


def _frames(balls1, dt=0.15, n=20, radius=0.75):
    balls0 = [dict(b, x=b["x"] - b["vx"] * dt, y=b["y"] - b["vy"] * dt) for b in balls1]
    out = []
    for balls in (balls0, balls1):
        G = bounce.make_grid(n)
        bounce.splat_all(G, n, balls, radius)
        out.append(torch.from_numpy(np.array(G, dtype=np.float32).transpose(2, 0, 1)))
    return out


def _balls():
    return [
        {"x": 6.2, "y": 6.4, "vx": 2.0, "vy": 0.5},
        {"x": 6.9, "y": 7.1, "vx": -1.5, "vy": -0.5},
        {"x": 14.3, "y": 12.6, "vx": 1.0, "vy": 2.0},
        {"x": 10.7, "y": 3.3, "vx": -2.0, "vy": 1.0},
    ]


def test_detect_balls_resolves_overlapping_pair():
    f0, f1 = _frames(_balls())
    pos, vel = detect_balls(f0, f1)
    truth = torch.tensor([[b["x"], b["y"]] for b in _balls()])
    assert pos.shape[0] == 4
    assert torch.cdist(pos, truth).min(dim=1).values.max() < 0.35


def test_detect_balls_does_not_split_isolated_balls():
    balls = [b for b in _balls() if b["x"] > 10]
    f0, f1 = _frames(balls)
    pos, _ = detect_balls(f0, f1)
    assert pos.shape[0] == 2


def test_detect_balls_skips_fit_above_max_tokens():
    f0, f1 = _frames(_balls())
    pos, vel = detect_balls(f0, f1, max_tokens=1)
    assert pos.shape[0] == vel.shape[0]


def test_model_ball_split_requires_readout_and_returns_all_balls():
    try:
        TokenModel(n=20, radius=0.75, dt=0.15, free_rollout=True, ball_split=True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    model = TokenModel(n=20, radius=0.75, dt=0.15, free_rollout=True, velocity_readout=True,
                       position_refine=True, ball_split=True)
    f0, f1 = _frames(_balls())
    pos, vel, hidden = model.init_tokens(f0, f1)
    assert pos.shape[0] == 4 and vel.shape == pos.shape and hidden.shape[0] == 4
