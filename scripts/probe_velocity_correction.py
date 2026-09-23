"""Probes whether an EXPLICIT velocity re-anchoring correction reduces
self-fed rollout error, before committing to any architecture/training
change.

Motivation (see scripts/measure_error_compounding.py output,
docs/debugging/experiment-log.md): TokenModel.step already corrects
POSITION every step by blending the dynamics prediction with an
observation read (centroid_near), gated by occlusion. VELOCITY has no
analogous correction -- TokenModel.step's own docstring says so
explicitly ("velocity is corrected only implicitly ... never
synthesized from a single-frame position delta"). Direct measurement
confirms this asymmetry matters: under self-feed, velocity error grows
from ~1.7 cells/s at step 0 to ~9+ cells/s by step 9 on the v9
checkpoint (roughly 5x), while position error over the same window
grows from ~0.5 to ~5 cells (roughly 10x, and closely tracking the
velocity error's shape) -- consistent with a compounding uncorrected
velocity bias driving most of the position drift through
`final_pos = corrected_pos + velocities * dt + delta_pos`.

This script re-implements TokenModel.step's rollout loop (reusing its
internal pieces directly: dynamics, centroid_near, occluding_mask,
rasterize_tokens) with ONE addition: it keeps the previous step's
observation-corrected position (`prev_obs_pos`) and, each step, also
computes an observation-implied velocity via finite difference of two
consecutive observation reads -- exactly init_tokens' own approach to
estimating velocity from two frames, just applied every step instead
of only at initialization. This is blended into the tracked velocity
with weight `velocity_weight`, gated by the SAME occlusion mask used
for position (an occluded/ambiguous reading is exactly as untrustworthy
for velocity as it is for position).

Purely inference-time: no retraining, no change to model/token_model.py.
If this reduces self-fed error on the existing frozen v9 checkpoint,
that's strong evidence the correction is worth building into the real
architecture (and retraining with it present, since the model currently
has never seen this correction during training).

Usage:
    PYTHONPATH=. python3 scripts/probe_velocity_correction.py \
        --checkpoint checkpoints/token_model_h12_v9.pt --num-seeds 48 --num-steps 20 \
        --velocity-weight 0.5
"""
import argparse
import random

import numpy as np
import torch

import bounce
from model.dataset import make_scenario_uniform
from model.token_model import TokenModel
from model.token_match import match_tokens_to_state
from model.token_detect import centroid_near
from model.token_gate import occluding_mask
from model.token_rasterize import rasterize_tokens


def load_model(checkpoint_path, n, hidden_dim, neighbor_radius):
    model = TokenModel(n=n, radius=0.75, dt=0.15, hidden_dim=hidden_dim, neighbor_radius=neighbor_radius)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def simulate(n, num_balls, seed, num_steps, dt=0.15, gravity=9.0, radius=0.75,
             stiffness=400.0, substeps=8, vy=2.3):
    rng = random.Random(seed)
    balls = make_scenario_uniform(num_balls, n, vy, rng)
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    frames = [np.array(G, dtype=np.float32)]
    states = [{"x": [b["x"] for b in balls], "y": [b["y"] for b in balls],
               "vx": [b["vx"] for b in balls], "vy": [b["vy"] for b in balls]}]
    for _ in range(num_steps):
        bounce.step(G, n, balls, dt, gravity, radius, stiffness, substeps)
        frames.append(np.array(G, dtype=np.float32))
        states.append({"x": [b["x"] for b in balls], "y": [b["y"] for b in balls],
                        "vx": [b["vx"] for b in balls], "vy": [b["vy"] for b in balls]})
    return frames, states


def gt_pos_vel(state):
    pos = torch.stack([torch.tensor(state["x"], dtype=torch.float32),
                        torch.tensor(state["y"], dtype=torch.float32)], dim=1)
    vel = torch.stack([torch.tensor(state["vx"], dtype=torch.float32),
                        torch.tensor(state["vy"], dtype=torch.float32)], dim=1)
    return pos, vel


def step_with_velocity_correction(model, positions, velocities, hidden, prev_obs_pos,
                                   observed_frame, velocity_weight):
    """Mirrors TokenModel.step exactly for the position-correction and
    dynamics parts; adds a velocity re-anchor from two consecutive
    observation reads. `prev_obs_pos` is None on the very first call
    (no prior observation to difference against -- velocity correction
    is skipped that step, same as init_tokens has no correction before
    frame 1)."""
    occluding = occluding_mask(positions, model._gate_radius())
    corrected_pos = positions.clone()
    corrected_vel = velocities.clone()
    w = model.observation_weight
    obs_pos = positions.clone()
    for i in range(positions.shape[0]):
        if occluding[i] or w == 0.0:
            obs_pos[i] = positions[i]
            continue
        op = centroid_near(observed_frame[0], positions[i], model.radius)
        obs_pos[i] = op
        corrected_pos[i] = (1 - w) * positions[i] + w * op
        if prev_obs_pos is not None and velocity_weight > 0.0 and not occluding[i]:
            obs_vel = (op - prev_obs_pos[i]) / model.dt
            corrected_vel[i] = (1 - velocity_weight) * velocities[i] + velocity_weight * obs_vel

    delta_pos, delta_vel, new_hidden = model.dynamics(corrected_pos, corrected_vel, hidden)
    final_pos = corrected_pos + corrected_vel * model.dt + delta_pos
    final_vel = corrected_vel + delta_vel

    next_grid = rasterize_tokens(final_pos, final_vel, model.n, model.radius)
    return final_pos, final_vel, new_hidden, next_grid, obs_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--num-balls", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--num-seeds", type=int, default=48)
    ap.add_argument("--base-seed", type=int, default=4738)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--neighbor-radius", type=float, default=4.0)
    ap.add_argument("--velocity-weight", type=float, default=0.5)
    args = ap.parse_args()

    model = load_model(args.checkpoint, args.n, args.hidden_dim, args.neighbor_radius)

    baseline_err = [[] for _ in range(args.num_steps - 1)]
    corrected_err = [[] for _ in range(args.num_steps - 1)]
    corrected_vel_err = [[] for _ in range(args.num_steps - 1)]

    for seed in range(args.base_seed, args.base_seed + args.num_seeds):
        frames, states = simulate(args.n, args.num_balls, seed, args.num_steps)
        g0 = torch.from_numpy(frames[0].transpose(2, 0, 1))
        g1 = torch.from_numpy(frames[1].transpose(2, 0, 1))

        with torch.no_grad():
            positions0, velocities0, hidden0 = model.init_tokens(g0, g1)
        state1 = {k: torch.tensor(v, dtype=torch.float32) for k, v in states[1].items()}
        token_to_ball = match_tokens_to_state(positions0, state1)
        if positions0.shape[0] == 0:
            continue

        # --- baseline: model.step, unmodified ---
        positions, velocities, hidden = positions0.clone(), velocities0.clone(), hidden0.clone()
        observed = g1
        with torch.no_grad():
            for step in range(args.num_steps - 1):
                positions, velocities, hidden, pred_grid, _ = model.step(positions, velocities, hidden, observed)
                observed = pred_grid
                gt_next, _ = gt_pos_vel(states[step + 2])
                gt_next = gt_next[token_to_ball]
                baseline_err[step].append(torch.norm(positions - gt_next, dim=1).mean().item())

        # --- corrected: with explicit velocity re-anchoring ---
        positions, velocities, hidden = positions0.clone(), velocities0.clone(), hidden0.clone()
        observed = g1
        prev_obs_pos = None
        with torch.no_grad():
            for step in range(args.num_steps - 1):
                positions, velocities, hidden, pred_grid, obs_pos = step_with_velocity_correction(
                    model, positions, velocities, hidden, prev_obs_pos, observed, args.velocity_weight
                )
                prev_obs_pos = obs_pos
                observed = pred_grid
                gt_next, gt_vel_next = gt_pos_vel(states[step + 2])
                gt_next, gt_vel_next = gt_next[token_to_ball], gt_vel_next[token_to_ball]
                corrected_err[step].append(torch.norm(positions - gt_next, dim=1).mean().item())
                corrected_vel_err[step].append(torch.norm(velocities - gt_vel_next, dim=1).mean().item())

    print(f"{'step':>5} {'baseline pos':>13} {'corrected pos':>14} {'corrected vel':>14}")
    for step in range(args.num_steps - 1):
        b = np.mean(baseline_err[step]) if baseline_err[step] else float("nan")
        c = np.mean(corrected_err[step]) if corrected_err[step] else float("nan")
        cv = np.mean(corrected_vel_err[step]) if corrected_vel_err[step] else float("nan")
        print(f"{step:5d} {b:13.4f} {c:14.4f} {cv:14.4f}")


if __name__ == "__main__":
    main()
