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


def make_scenario_clustered(num_balls, n, vy, rng, cluster_radius):
    lo, hi = cluster_radius, (n - 1.0) - cluster_radius
    cx = rng.uniform(lo, hi)
    cy = rng.uniform(lo, hi)
    balls = []
    for _ in range(num_balls):
        x = min(max(cx + rng.uniform(-cluster_radius, cluster_radius), 0.0), n - 1.0)
        y = min(max(cy + rng.uniform(-cluster_radius, cluster_radius), 0.0), n - 1.0)
        vx = rng.uniform(-vy, vy)
        vy_ = rng.uniform(-vy, vy)
        balls.append({"x": x, "y": y, "vx": vx, "vy": vy_})
    return balls


def make_scenario_settled(num_balls, n, vy, rng, radius, gravity, stiffness, dt, substeps, settle_steps):
    balls = bounce.init_balls(num_balls, n, vy, rng)
    G = bounce.make_grid(n)
    for _ in range(settle_steps):
        bounce.step(G, n, balls, dt, gravity, radius, stiffness, substeps)
    return balls
