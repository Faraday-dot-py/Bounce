import random

import torch

import bounce
from model.dataset import make_scenario_uniform, generate_pair
from model.losses import weighted_channel_mse


def _pair_to_batch(g_t, g_t1, device):
    g_t = torch.from_numpy(g_t.transpose(2, 0, 1)).unsqueeze(0).to(device)
    g_t1 = torch.from_numpy(g_t1.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return g_t, g_t1


def held_out_loss(model, n, ball_range, num_samples, seed, weights,
                   dt=0.15, gravity=9.0, radius=0.75, stiffness=400.0, substeps=8, vy=2.3):
    device = next(model.parameters()).device
    rng = random.Random(seed)
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(num_samples):
            num_balls = rng.randint(*ball_range)
            balls = make_scenario_uniform(num_balls, n, vy, rng)
            g_t, g_t1 = generate_pair(balls, n, dt, gravity, radius, stiffness, substeps)
            g_t, g_t1 = _pair_to_batch(g_t, g_t1, device)
            pred = model(g_t)
            total += weighted_channel_mse(pred, g_t1, weights.to(device)).item()
    return total / num_samples


def rollout_divergence(model, n, num_balls, seed, num_steps,
                        dt=0.15, gravity=9.0, radius=0.75, stiffness=400.0, substeps=8, vy=2.3):
    device = next(model.parameters()).device
    rng = random.Random(seed)
    model.eval()

    balls = make_scenario_uniform(num_balls, n, vy, rng)
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    import numpy as np
    g_true = np.array(G, dtype=np.float32)
    g_pred = torch.from_numpy(g_true.transpose(2, 0, 1)).unsqueeze(0).to(device)

    divergences = []
    with torch.no_grad():
        for _ in range(num_steps):
            bounce.step(G, n, balls, dt, gravity, radius, stiffness, substeps)
            g_true = np.array(G, dtype=np.float32)
            g_true_t = torch.from_numpy(g_true.transpose(2, 0, 1)).unsqueeze(0).to(device)

            g_pred = model(g_pred)
            divergences.append(((g_pred - g_true_t) ** 2).mean().item())
    return divergences


def generalization_check(model, n, ball_count, num_samples, seed, weights,
                          dt=0.15, gravity=9.0, radius=0.75, stiffness=400.0, substeps=8, vy=2.3):
    return held_out_loss(
        model, n, ball_range=(ball_count, ball_count), num_samples=num_samples,
        seed=seed, weights=weights, dt=dt, gravity=gravity, radius=radius,
        stiffness=stiffness, substeps=substeps, vy=vy,
    )
