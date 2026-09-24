"""Spike: where does the y-momentum bias come from? (a) two interior tokens
on a y-axis / x-axis head-on course, total momentum change per step vs the
true simulator; (b) single token bouncing off the y=0 vs y=19 wall,
mirror-image starts. Throwaway.

Usage:
    PYTHONPATH=. python3 scripts/probe_y_bias_source.py --checkpoint checkpoints/token_model_h24_v18.pt
"""
import argparse

import torch

import bounce
from scripts.eval_free_rollout import load_model


def true_run(balls, steps):
    G = bounce.make_grid(20)
    out = []
    for _ in range(steps):
        bounce.step(G, 20, balls, 0.15, 9.0, 0.75, 400.0, 8)
        out.append([(round(b["x"], 2), round(b["y"], 2), round(b["vx"], 2), round(b["vy"], 2)) for b in balls])
    return out


def model_run(model, pos, vel, steps):
    p = torch.tensor(pos); v = torch.tensor(vel); h = torch.zeros(len(pos), 32)
    out = []
    with torch.no_grad():
        for _ in range(steps):
            p, v, h, _ = model.step_free(p, v, h)
            out.append([(round(float(a[0]), 2), round(float(a[1]), 2), round(float(b[0]), 2), round(float(b[1]), 2)) for a, b in zip(p, v)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    model = load_model(args.checkpoint, "free", 20, 32, 4.0)

    print("(a) head-on along y: balls at (5,8.0) vy+2 and (5,11.0) vy-2 (no gravity effect on relative y)")
    for name, out in [("model", model_run(model, [[5., 8.], [5., 11.]], [[0., 2.], [0., -2.]], 8)),
                      ("truth", true_run([dict(x=5., y=8., vx=0., vy=2.), dict(x=5., y=11., vx=0., vy=-2.)], 8))]:
        print(" ", name, "total vy per step:", [round(o[0][3] + o[1][3], 2) for o in out])
    print("(a2) same, mirrored y (2 balls, opposite ordering):")
    print("  model total vy:", [round(o[0][3] + o[1][3], 2) for o in model_run(model, [[5., 11.], [5., 8.]], [[0., -2.], [0., 2.]], 8)])

    print("(b) wall bounce, single token vy toward wall, x=5:")
    for label, y0, vy0 in [("left y=1.5 vy-3", 1.5, -3.), ("right y=17.5 vy+3", 17.5, 3.)]:
        m = model_run(model, [[5., y0]], [[0., vy0]], 8)
        t = true_run([dict(x=5., y=y0, vx=0., vy=vy0)], 8)
        print(f"  {label}: model vy", [o[0][3] for o in m], " truth vy", [o[0][3] for o in t])
    print("(b2) x-wall for reference, token near x=0 moving vx-3 at y=10 / near x=19 vx+3:")
    for label, x0, vx0 in [("x=1.5 vx-3", 1.5, -3.), ("x=17.5 vx+3", 17.5, 3.)]:
        m = model_run(model, [[x0, 10.]], [[vx0, 0.]], 8)
        print(f"  {label}: model vx", [o[0][2] for o in m])


if __name__ == "__main__":
    main()
