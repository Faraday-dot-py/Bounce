"""Directly measures where rollout error comes from for TokenModel: is
one dynamics step (t -> t+1) itself nearly exact, with error only
compounding through self-feed, or is even the teacher-forced per-step
error already large enough to explain the observed drift on its own?

Two conditions per step, same seeds/model:
  - teacher-forced: positions/velocities reset to ground truth every
    step before calling model.step (hidden state still carries forward,
    since it isn't part of state_seq and has no ground truth), observed
    frame is the real simulator frame. Isolates ONE dynamics step's own
    error, independent of any prior drift.
  - self-fed: standard autoregressive rollout (model's own prediction
    feeds the next step, both for tracked state and observed_frame).

Reports mean position error vs. step for both conditions -- if
teacher-forced error stays flat and small while self-fed error grows,
that confirms compounding rather than per-step inaccuracy. Also reports
the mean per-step *increment* (error growth rate) under self-feed to
see whether it's constant (linear drift, e.g. an uncorrected velocity
bias) or accelerating (unstable feedback).

Usage:
    PYTHONPATH=. python3 scripts/measure_error_compounding.py \
        --checkpoint checkpoints/token_model_h12_v9.pt --num-seeds 48 --num-steps 20
"""
import argparse

import numpy as np
import torch

import bounce
from model.dataset import make_scenario_uniform
from model.token_model import TokenModel
from model.token_match import match_tokens_to_state
import random


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
    args = ap.parse_args()

    model = load_model(args.checkpoint, args.n, args.hidden_dim, args.neighbor_radius)

    tf_err = [[] for _ in range(args.num_steps - 1)]
    sf_err = [[] for _ in range(args.num_steps - 1)]
    sf_vel_err = [[] for _ in range(args.num_steps - 1)]

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

        # --- teacher-forced: reset pos/vel to ground truth every step ---
        positions, velocities, hidden = positions0.clone(), velocities0.clone(), hidden0.clone()
        with torch.no_grad():
            for step in range(args.num_steps - 1):
                gt_pos, gt_vel = gt_pos_vel(states[step + 1])
                gt_pos, gt_vel = gt_pos[token_to_ball], gt_vel[token_to_ball]
                observed = torch.from_numpy(frames[step + 1].transpose(2, 0, 1))
                positions, velocities, hidden, _, _ = model.step(gt_pos, gt_vel, hidden, observed)
                gt_next, _ = gt_pos_vel(states[step + 2])
                gt_next = gt_next[token_to_ball]
                err = torch.norm(positions - gt_next, dim=1).mean().item()
                tf_err[step].append(err)

        # --- self-fed: standard autoregressive rollout ---
        positions, velocities, hidden = positions0.clone(), velocities0.clone(), hidden0.clone()
        observed = g1
        with torch.no_grad():
            for step in range(args.num_steps - 1):
                positions, velocities, hidden, pred_grid, _ = model.step(positions, velocities, hidden, observed)
                observed = pred_grid
                gt_next, gt_vel_next = gt_pos_vel(states[step + 2])
                gt_next, gt_vel_next = gt_next[token_to_ball], gt_vel_next[token_to_ball]
                err = torch.norm(positions - gt_next, dim=1).mean().item()
                vel_err = torch.norm(velocities - gt_vel_next, dim=1).mean().item()
                sf_err[step].append(err)
                sf_vel_err[step].append(vel_err)

    print(f"{'step':>5} {'teacher-forced':>15} {'self-fed':>10} {'sf increment':>13} {'sf vel_err':>11}")
    prev = 0.0
    for step in range(args.num_steps - 1):
        tf_mean = np.mean(tf_err[step]) if tf_err[step] else float("nan")
        sf_mean = np.mean(sf_err[step]) if sf_err[step] else float("nan")
        sf_vel_mean = np.mean(sf_vel_err[step]) if sf_vel_err[step] else float("nan")
        inc = sf_mean - prev
        prev = sf_mean
        print(f"{step:5d} {tf_mean:15.4f} {sf_mean:10.4f} {inc:13.4f} {sf_vel_mean:11.4f}")


if __name__ == "__main__":
    main()
