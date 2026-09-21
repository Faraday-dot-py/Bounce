import numpy as np
import bounce


def make_scenario_uniform(num_balls, n, vy, rng):
    return bounce.init_balls(num_balls, n, vy, rng)


def generate_pair(balls, n, dt, gravity, radius, stiffness, substeps):
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    g_t = np.array(G, dtype=np.float32)
    bounce.step(G, n, balls, dt, gravity, radius, stiffness, substeps)
    g_t1 = np.array(G, dtype=np.float32)
    return g_t, g_t1
