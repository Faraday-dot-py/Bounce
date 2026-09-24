"""Dump PyTorch rollouts (token_model_soup_b, cell graph, contain_state as in
scripts/realtime_sim.py) to web/tests/vectors.json for web/tests/verify.mjs.

Usage:
    PYTHONPATH=. python3 scripts/dump_web_testvectors.py
"""
import json

import numpy as np
import torch

from scripts.eval_free_rollout import load_model
from scripts.realtime_sim import contain_state

MAX_SPEED = 30.0


def scene(rng, count, lo, hi, speed):
    pos = rng.uniform(lo, hi, (count, 2))
    vel = rng.uniform(-speed, speed, (count, 2))
    return torch.tensor(pos, dtype=torch.float32), torch.tensor(vel, dtype=torch.float32)


def rollout(model, pos, vel, steps, n):
    hidden = torch.zeros((pos.shape[0], model.dynamics.hidden_dim))
    states = [(pos, vel, hidden)]
    hooks = {}
    d = model.dynamics
    for name, mod in [("q", d.query), ("gru", d.gru), ("delta", d.delta_head), ("wall", d.wall_head)]:
        mod.register_forward_hook(lambda m, i, o, name=name: hooks.__setitem__(name, o.detach().clone()))
    first = None
    with torch.no_grad():
        for s in range(steps):
            pos, vel, hidden, _ = model.step_free(pos, vel, hidden, render=False)
            pos, vel, hidden = contain_state(pos, vel, hidden, n, MAX_SPEED)
            states.append((pos, vel, hidden))
            if s == 1:
                first = {k: v.tolist() for k, v in hooks.items()}
    return states, first


def main():
    n = 100
    model = load_model("checkpoints/token_model_soup_b.pt", "free", n, 32, 4.0, False, True, True, True, True, True, True)
    model.dynamics.cell_graph = True
    rng = np.random.default_rng(4738)
    scenes = {
        "spread12": (12, 5.0, 95.0, 2.3, 60),
        "cluster30": (30, 40.0, 60.0, 2.3, 20),
        "dense100": (100, 30.0, 70.0, 2.3, 6),
    }
    out = {"n": n, "max_speed": MAX_SPEED, "scenes": {}}
    for name, (count, lo, hi, speed, steps) in scenes.items():
        pos, vel = scene(rng, count, lo, hi, speed)
        states, first = rollout(model, pos, vel, steps, n)
        out["scenes"][name] = {
            "steps": steps,
            "pos0": states[0][0].tolist(), "vel0": states[0][1].tolist(),
            "pos": [[s[0].tolist() for s in states]][0],
            "vel": [s[1].tolist() for s in states],
            "hidden": [s[2].tolist() for s in states],
            "count": [s[0].shape[0] for s in states],
            "hooks_step2": first,
        }
        print(name, count, "->", states[-1][0].shape[0], "balls at end")
    with open("web/tests/vectors.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
