"""Dump bounce.py rollouts to web/tests/physics_ref.json for web/tests/physics.test.mjs.

Usage:
    PYTHONPATH=. python3 scripts/dump_web_physics_ref.py
    node web/tests/physics.test.mjs web/tests/physics_ref.json
"""
import json
import random

import bounce

cases = []
for name, num, n in [("n100_12", 12, 100), ("n20_40", 40, 20)]:
    rng = random.Random(4738)
    balls = bounce.init_balls(num, n, 3.0, rng)
    init = [dict(b) for b in balls]
    G = bounce.make_grid(n)
    frames = []
    for _ in range(200):
        bounce.step(G, n, balls, 0.15, 9.0, 0.75, 400.0, 8)
        frames.append([[b["x"], b["y"], b["vx"], b["vy"]] for b in balls])
    cases.append({"name": name, "n": n, "init": init, "frames": frames})
with open("web/tests/physics_ref.json", "w") as f:
    json.dump(cases, f)
