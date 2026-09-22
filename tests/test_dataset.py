import os
import random
import numpy as np
import bounce
from model.dataset import generate_sequence, make_scenario_uniform


def test_generate_sequence_shapes_and_prob_bounds():
    rng = random.Random(4738)
    balls = make_scenario_uniform(10, 50, 2.3, rng)
    frames = generate_sequence(balls, 50, 0.15, 9.0, 0.75, 400.0, 8, horizon=3)
    assert len(frames) == 4
    for frame in frames:
        assert frame.shape == (50, 50, bounce.NUM_CHANNELS)
        assert frame.dtype == np.float32
        assert (frame[:, :, bounce.PROB] >= 0.0).all()
        assert (frame[:, :, bounce.PROB] < 1.0).all()


def test_generate_sequence_frames_differ_across_steps():
    rng = random.Random(4738)
    balls = make_scenario_uniform(10, 50, 2.3, rng)
    frames = generate_sequence(balls, 50, 0.15, 9.0, 0.75, 400.0, 8, horizon=3)
    assert not np.allclose(frames[0], frames[1])
    assert not np.allclose(frames[1], frames[2])
    assert not np.allclose(frames[2], frames[3])


from model.dataset import make_scenario_clustered, make_scenario_settled


def test_clustered_scenario_balls_stay_within_cluster_radius():
    rng = random.Random(4738)
    balls = make_scenario_clustered(20, 50, 2.3, rng, cluster_radius=3.0)
    xs = [b["x"] for b in balls]
    ys = [b["y"] for b in balls]
    assert max(xs) - min(xs) <= 2 * 3.0 + 1e-6
    assert max(ys) - min(ys) <= 2 * 3.0 + 1e-6


def test_settled_scenario_balls_move_toward_high_x_under_gravity():
    rng = random.Random(4738)
    start = make_scenario_uniform(20, 50, 0.0, rng)
    start_mean_x = sum(b["x"] for b in start) / len(start)
    rng2 = random.Random(4738)
    settled = make_scenario_settled(
        20, 50, 0.0, rng2, radius=0.75, gravity=9.0, stiffness=400.0,
        dt=0.15, substeps=8, settle_steps=200,
    )
    settled_mean_x = sum(b["x"] for b in settled) / len(settled)
    assert settled_mean_x > start_mean_x


import torch
from model.dataset import BounceSequenceDataset


def test_bounce_sequence_dataset_shapes_and_determinism():
    ds1 = BounceSequenceDataset(num_samples=6, n=50, ball_range=(5, 15), seed=4738, horizon=3)
    ds2 = BounceSequenceDataset(num_samples=6, n=50, ball_range=(5, 15), seed=4738, horizon=3)
    assert len(ds1) == 6
    seq = ds1[0]
    assert seq.shape == (4, 3, 50, 50)
    assert isinstance(seq, torch.Tensor)
    seq_again = ds2[0]
    assert torch.equal(seq, seq_again)


def test_bounce_sequence_dataset_cache_roundtrip(tmp_path):
    cache_path = str(tmp_path / "cache.npz")
    ds1 = BounceSequenceDataset(num_samples=4, n=50, ball_range=(5, 15), seed=4738, horizon=3, cache_path=cache_path)
    assert os.path.exists(cache_path)
    ds2 = BounceSequenceDataset(num_samples=4, n=50, ball_range=(5, 15), seed=4738, horizon=3, cache_path=cache_path)
    assert len(ds2) == len(ds1)
    for i in range(len(ds1)):
        assert torch.equal(ds1[i], ds2[i])


def test_bounce_sequence_dataset_cache_rejects_mismatched_config(tmp_path):
    cache_path = str(tmp_path / "cache.npz")
    BounceSequenceDataset(num_samples=4, n=50, ball_range=(5, 15), seed=4738, horizon=3, cache_path=cache_path)
    try:
        BounceSequenceDataset(num_samples=4, n=50, ball_range=(5, 15), seed=4738, horizon=2, cache_path=cache_path)
        assert False, "expected ValueError for mismatched cache config"
    except ValueError:
        pass
