import os
import random
import time
import numpy as np
import torch
import bounce
from torch.utils.data import Dataset


def make_scenario_uniform(num_balls, n, vy, rng):
    return bounce.init_balls(num_balls, n, vy, rng)


def generate_sequence(balls, n, dt, gravity, radius, stiffness, substeps, horizon):
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    frames = [np.array(G, dtype=np.float32)]
    for _ in range(horizon):
        bounce.step(G, n, balls, dt, gravity, radius, stiffness, substeps)
        frames.append(np.array(G, dtype=np.float32))
    return frames


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


def generate_scenario_balls(scenario, num_balls, n, vy, rng, cluster_radius, radius, gravity,
                            stiffness, dt, substeps, settle_steps):
    """Dispatch to the scenario builders. Shared by BounceSequenceDataset
    and model.token_dataset.BounceTokenSequenceDataset so the two can't
    drift apart on how a scenario is constructed."""
    if scenario == "uniform":
        return make_scenario_uniform(num_balls, n, vy, rng)
    elif scenario == "clustered":
        return make_scenario_clustered(num_balls, n, vy, rng, cluster_radius)
    else:
        return make_scenario_settled(
            num_balls, n, vy, rng, radius, gravity, stiffness, dt, substeps, settle_steps,
        )


class BounceSequenceDataset(Dataset):
    SCENARIOS = ("uniform", "clustered", "settled")

    def __init__(self, num_samples, n, ball_range, seed, horizon=3, dt=0.15, gravity=9.0,
                 radius=0.75, stiffness=400.0, substeps=8, vy=2.3,
                 cluster_radius=3.0, settle_steps=200, cache_path=None):
        config = {
            "num_samples": num_samples, "n": n, "ball_range": tuple(ball_range),
            "seed": seed, "horizon": horizon,
        }
        if cache_path is not None and os.path.exists(cache_path):
            self.samples = self._load_cache(cache_path, config)
            print(f"[dataset] loaded {len(self.samples)} samples from {cache_path}", flush=True)
            return

        self.samples = []
        rng = random.Random(seed)
        t_start = time.time()
        for i in range(num_samples):
            scenario = self.SCENARIOS[i % len(self.SCENARIOS)]
            num_balls = rng.randint(*ball_range)
            balls = generate_scenario_balls(
                scenario, num_balls, n, vy, rng, cluster_radius, radius, gravity,
                stiffness, dt, substeps, settle_steps,
            )
            frames = generate_sequence(balls, n, dt, gravity, radius, stiffness, substeps, horizon)
            self.samples.append(np.stack(frames))
            if (i + 1) % 100 == 0 or (i + 1) == num_samples:
                elapsed = time.time() - t_start
                print(f"[dataset] generated {i + 1}/{num_samples} samples ({elapsed:.1f}s elapsed)", flush=True)

        if cache_path is not None:
            self._save_cache(cache_path, config)
            print(f"[dataset] saved {len(self.samples)} samples to {cache_path}", flush=True)

    def _save_cache(self, cache_path, config):
        seq_arr = np.stack(self.samples)
        np.savez(
            cache_path, sequences=seq_arr,
            num_samples=config["num_samples"], n=config["n"],
            ball_range=np.array(config["ball_range"]), seed=config["seed"],
            horizon=config["horizon"],
        )

    @staticmethod
    def _load_cache(cache_path, config):
        data = np.load(cache_path)
        cached_config = {
            "num_samples": int(data["num_samples"]), "n": int(data["n"]),
            "ball_range": tuple(int(x) for x in data["ball_range"]), "seed": int(data["seed"]),
            "horizon": int(data["horizon"]),
        }
        if cached_config != config:
            raise ValueError(
                f"dataset cache at {cache_path} was generated with config {cached_config}, "
                f"but this run requested {config}. Delete the cache or use a different --cache-path."
            )
        seq_arr = data["sequences"]
        return [seq_arr[i] for i in range(seq_arr.shape[0])]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.samples[idx]
        return torch.from_numpy(seq.transpose(0, 3, 1, 2)).clone()
