import random
import numpy as np
import bounce
from model.dataset import generate_pair, make_scenario_uniform


def test_generate_pair_shapes_and_prob_bounds():
    rng = random.Random(4738)
    balls = make_scenario_uniform(10, 50, 2.3, rng)
    g_t, g_t1 = generate_pair(balls, 50, 0.15, 9.0, 0.75, 400.0, 8)
    assert g_t.shape == (50, 50, bounce.NUM_CHANNELS)
    assert g_t1.shape == (50, 50, bounce.NUM_CHANNELS)
    assert g_t.dtype == np.float32
    assert (g_t[:, :, bounce.PROB] >= 0.0).all()
    assert (g_t[:, :, bounce.PROB] < 1.0).all()


def test_generate_pair_differs_after_one_step():
    rng = random.Random(4738)
    balls = make_scenario_uniform(10, 50, 2.3, rng)
    g_t, g_t1 = generate_pair(balls, 50, 0.15, 9.0, 0.75, 400.0, 8)
    assert not np.allclose(g_t, g_t1)


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
