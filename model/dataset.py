import random
import numpy as np
import torch
import bounce
from torch.utils.data import Dataset


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


class BouncePairDataset(Dataset):
    SCENARIOS = ("uniform", "clustered", "settled")

    def __init__(self, num_samples, n, ball_range, seed, dt=0.15, gravity=9.0,
                 radius=0.75, stiffness=400.0, substeps=8, vy=2.3,
                 cluster_radius=3.0, settle_steps=200):
        self.samples = []
        rng = random.Random(seed)
        for i in range(num_samples):
            scenario = self.SCENARIOS[i % len(self.SCENARIOS)]
            num_balls = rng.randint(*ball_range)
            if scenario == "uniform":
                balls = make_scenario_uniform(num_balls, n, vy, rng)
            elif scenario == "clustered":
                balls = make_scenario_clustered(num_balls, n, vy, rng, cluster_radius)
            else:
                balls = make_scenario_settled(
                    num_balls, n, vy, rng, radius, gravity, stiffness, dt,
                    substeps, settle_steps,
                )
            g_t, g_t1 = generate_pair(balls, n, dt, gravity, radius, stiffness, substeps)
            self.samples.append((g_t, g_t1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        g_t, g_t1 = self.samples[idx]
        g_t = torch.from_numpy(g_t.transpose(2, 0, 1)).clone()
        g_t1 = torch.from_numpy(g_t1.transpose(2, 0, 1)).clone()
        return g_t, g_t1
