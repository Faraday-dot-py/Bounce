import random

import numpy as np
import torch
from torch.utils.data import Dataset

import bounce
from model.dataset import generate_scenario_balls


def _copy_states(balls):
    return [dict(b) for b in balls]


def generate_token_sequence(balls, n, dt, gravity, radius, stiffness, substeps, horizon):
    """Same stepping as model.dataset.generate_sequence, but also records
    each frame's exact ball states. model.dataset discards these after
    rendering; the token model needs them for initialization and for
    collision-specific eval metrics that the grid-only dataset can't
    supply (see design spec's Validation plan)."""
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    frames = [np.array(G, dtype=np.float32)]
    states = [_copy_states(balls)]
    for _ in range(horizon):
        bounce.step(G, n, balls, dt, gravity, radius, stiffness, substeps)
        frames.append(np.array(G, dtype=np.float32))
        states.append(_copy_states(balls))
    return frames, states


SCENARIOS = ("uniform", "clustered", "settled")


def generate_dataset_samples(num_samples, n, ball_range, seed, horizon=3, dt=0.15, gravity=9.0,
                              radius=0.75, stiffness=400.0, substeps=8, vy=2.3,
                              cluster_radius=3.0, settle_steps=200, log_every=0):
    samples = []
    rng = random.Random(seed)
    for i in range(num_samples):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        num_balls = rng.randint(*ball_range)
        balls = generate_scenario_balls(
            scenario, num_balls, n, vy, rng, cluster_radius, radius, gravity,
            stiffness, dt, substeps, settle_steps,
        )
        frames, states = generate_token_sequence(
            balls, n, dt, gravity, radius, stiffness, substeps, horizon
        )
        samples.append((np.stack(frames), states))
        if log_every and (i + 1) % log_every == 0:
            print(f"generated {i + 1}/{num_samples} samples", flush=True)
    return samples


def save_dataset_samples(samples, path):
    torch.save(samples, path)


def load_dataset_samples(path):
    # Samples are (np.ndarray, list[dict]) tuples, not just tensors, so this
    # needs the full unpickler; only load caches this script itself wrote.
    return torch.load(path, weights_only=False)


class BounceTokenSequenceDataset(Dataset):
    SCENARIOS = SCENARIOS

    def __init__(self, num_samples, n, ball_range, seed, horizon=3, dt=0.15, gravity=9.0,
                 radius=0.75, stiffness=400.0, substeps=8, vy=2.3,
                 cluster_radius=3.0, settle_steps=200, log_every=0):
        self.samples = generate_dataset_samples(
            num_samples, n, ball_range, seed, horizon, dt, gravity, radius,
            stiffness, substeps, vy, cluster_radius, settle_steps, log_every,
        )

    @classmethod
    def from_samples(cls, samples):
        obj = cls.__new__(cls)
        obj.samples = samples
        return obj

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames, states = self.samples[idx]
        grid_seq = torch.from_numpy(frames.transpose(0, 3, 1, 2)).clone()
        state_seq = [
            {
                "x": torch.tensor([b["x"] for b in frame_states], dtype=torch.float32),
                "y": torch.tensor([b["y"] for b in frame_states], dtype=torch.float32),
                "vx": torch.tensor([b["vx"] for b in frame_states], dtype=torch.float32),
                "vy": torch.tensor([b["vy"] for b in frame_states], dtype=torch.float32),
            }
            for frame_states in states
        ]
        return grid_seq, state_seq
