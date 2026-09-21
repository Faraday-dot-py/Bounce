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
