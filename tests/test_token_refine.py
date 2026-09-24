import numpy as np
import torch

import bounce
from model.token_detect import find_token_positions, read_token_velocities
from model.token_refine import refine_positions
from model.token_model import TokenModel


def _frames(balls0, balls1, n=20, radius=0.75):
    out = []
    for balls in (balls0, balls1):
        G = bounce.make_grid(n)
        bounce.splat_all(G, n, balls, radius)
        out.append(torch.from_numpy(np.array(G, dtype=np.float32).transpose(2, 0, 1)))
    return out


def _scene():
    dt = 0.15
    balls1 = [
        {"x": 5.37, "y": 6.61, "vx": 2.1, "vy": -1.4},
        {"x": 13.22, "y": 12.48, "vx": -3.0, "vy": 0.7},
        {"x": 8.61, "y": 15.33, "vx": 1.0, "vy": 2.0},
    ]
    balls0 = [dict(b, x=b["x"] - b["vx"] * dt, y=b["y"] - b["vy"] * dt) for b in balls1]
    return balls0, balls1


def test_refine_beats_detector_on_isolated_balls():
    balls0, balls1 = _scene()
    f0, f1 = _frames(balls0, balls1)
    truth = torch.tensor([[b["x"], b["y"]] for b in balls1])
    coarse = find_token_positions(f1[0], 0.75, 0.1)
    vel = read_token_velocities(f1, coarse)
    refined = refine_positions(f0, f1, coarse, vel)
    err = lambda p: torch.cdist(p, truth).min(dim=1).values.mean()
    assert err(refined) < err(coarse)
    assert err(refined) < 0.12


def test_refine_empty_and_single_token():
    f0, f1 = _frames(*_scene())
    empty = torch.zeros((0, 2))
    assert refine_positions(f0, f1, empty, empty).shape == (0, 2)
    one = torch.tensor([[5.4, 6.6]])
    out = refine_positions(f0, f1, one, torch.tensor([[2.1, -1.4]]))
    assert out.shape == (1, 2) and torch.isfinite(out).all()
    assert torch.norm(out - one) <= 1.0


def test_model_position_refine_requires_readout_and_runs():
    try:
        TokenModel(n=20, radius=0.75, dt=0.15, free_rollout=True, position_refine=True)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    model = TokenModel(n=20, radius=0.75, dt=0.15, free_rollout=True, velocity_readout=True, position_refine=True)
    f0, f1 = _frames(*_scene())
    pos, vel, hidden = model.init_tokens(f0, f1)
    assert pos.shape[0] == 3 and vel.shape == pos.shape
