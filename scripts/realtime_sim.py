"""Real-time interactive token-model sim. Starts empty; hold the left mouse
button to spawn a stream of balls at the cursor. Space pauses, c clears,
q quits. Gravity pulls toward the bottom of the window.

A step is O(N) in the ball count and independent of the grid size: the
neighbour graph uses a cell list and no grid is rasterized -- balls are drawn
as canvas discs.

Usage:
    PYTHONPATH=. python3 scripts/realtime_sim.py --checkpoint checkpoints/token_model_soup_b.pt
"""
import argparse
import time

import numpy as np
import torch

from scripts.eval_free_rollout import load_model


def contain_state(positions, velocities, hidden, n, max_speed):
    """Guard for the learned dynamics, which can blow up when balls are
    stacked far outside its training density: drop non-finite tokens,
    clamp to the box (zeroing outward velocity) and cap speed."""
    ok = torch.isfinite(positions).all(dim=1) & torch.isfinite(velocities).all(dim=1)
    if not bool(ok.all()):
        positions, velocities, hidden = positions[ok], velocities[ok], hidden[ok]
    top = n - 1.0
    outward = ((positions < 0) & (velocities < 0)) | ((positions > top) & (velocities > 0))
    velocities = torch.where(outward, torch.zeros_like(velocities), velocities)
    positions = positions.clamp(0.0, top)
    speed = velocities.norm(dim=1, keepdim=True).clamp(min=1e-6)
    velocities = velocities * (speed.clamp(max=max_speed) / speed)
    return positions, velocities, hidden


class Sim:
    def __init__(self, model, max_balls, spawn_every, spawn_speed, seed):
        self.model = model
        self.model.dynamics.cell_graph = True
        self.n = model.n
        self.max_balls = max_balls
        self.spawn_every = spawn_every
        self.spawn_speed = spawn_speed
        self.min_gap = 1.5
        self.max_speed = 30.0
        self.rng = np.random.default_rng(seed)
        self.clear()

    def clear(self):
        self.positions = torch.zeros((0, 2))
        self.velocities = torch.zeros((0, 2))
        self.hidden = torch.zeros((0, self.model.dynamics.hidden_dim))
        self.ticks = 0

    def spawn(self, x, y):
        lo, hi = 1.0, self.n - 2.0
        p = np.clip([x, y] + self.rng.normal(0, 0.4, 2), lo, hi)
        if self.positions.shape[0] and float((self.positions - torch.tensor(p, dtype=torch.float32)).norm(dim=1).min()) < self.min_gap:
            return
        v = self.rng.uniform(-self.spawn_speed, self.spawn_speed, 2)
        self.positions = torch.cat([self.positions, torch.tensor(p, dtype=torch.float32)[None]])
        self.velocities = torch.cat([self.velocities, torch.tensor(v, dtype=torch.float32)[None]])
        self.hidden = torch.cat([self.hidden, torch.zeros((1, self.hidden.shape[1]))])
        if self.positions.shape[0] > self.max_balls:
            self.positions, self.velocities, self.hidden = (
                self.positions[1:], self.velocities[1:], self.hidden[1:])

    def step(self, source):
        self.ticks += 1
        if source is not None and self.ticks % self.spawn_every == 0:
            self.spawn(*source)
        if self.positions.shape[0] == 0:
            return
        with torch.no_grad():
            self.positions, self.velocities, self.hidden, _ = self.model.step_free(
                self.positions, self.velocities, self.hidden, render=False)
        self.contain()

    def contain(self):
        self.positions, self.velocities, self.hidden = contain_state(
            self.positions, self.velocities, self.hidden, self.n, self.max_speed)


def speed_color(speed):
    t = min(speed / 10.0, 1.0)
    return "#%02x%02x%02x" % (int(255 * (0.25 + 0.75 * t)), int(255 * (0.55 + 0.1 * t - 0.4 * t ** 2)),
                              int(255 * (1.0 - 0.85 * t)))


def main():
    import tkinter as tk

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/token_model_soup_b.pt")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--scale", type=int, default=7)
    ap.add_argument("--fps", type=float, default=30.0, help="model steps per wall-clock second")
    ap.add_argument("--spawn-every", type=int, default=2, help="steps between spawns while held")
    ap.add_argument("--spawn-speed", type=float, default=2.3, help="max initial speed per axis")
    ap.add_argument("--max-balls", type=int, default=300)
    ap.add_argument("--seed", type=int, default=4738)
    args = ap.parse_args()

    model = load_model(args.checkpoint, "free", args.n, 32, 4.0, False, True, True, True, True, True, True)
    sim = Sim(model, args.max_balls, args.spawn_every, args.spawn_speed, args.seed)

    root = tk.Tk()
    root.title("Bounce real-time")
    canvas = tk.Canvas(root, width=args.n * args.scale, height=args.n * args.scale, highlightthickness=0, bg="black")
    canvas.pack()
    label = tk.Label(root, text="", anchor="w", font=("monospace", 10))
    label.pack(fill="x")
    discs = []
    disc_radius = model.radius * args.scale

    state = {"held": False, "cursor": (0.0, 0.0), "paused": False, "ms": 0.0}

    def set_cursor(e):
        state["cursor"] = (e.y / args.scale, e.x / args.scale)

    def press(e):
        set_cursor(e)
        state["held"] = True

    def release(_):
        state["held"] = False

    canvas.bind("<ButtonPress-1>", press)
    canvas.bind("<B1-Motion>", set_cursor)
    canvas.bind("<ButtonRelease-1>", release)
    root.bind("<space>", lambda _: state.update(paused=not state["paused"]))
    root.bind("c", lambda _: sim.clear())
    root.bind("q", lambda _: root.destroy())

    period = 1.0 / args.fps

    def tick():
        t0 = time.perf_counter()
        if not state["paused"]:
            sim.step(state["cursor"] if state["held"] else None)
            count = sim.positions.shape[0]
            while len(discs) > count:
                canvas.delete(discs.pop())
            while len(discs) < count:
                discs.append(canvas.create_oval(0, 0, 0, 0, outline=""))
            xy = ((sim.positions + 0.5) * args.scale).tolist()
            speeds = sim.velocities.norm(dim=1).tolist()
            for disc, (x, y), speed in zip(discs, xy, speeds):
                canvas.coords(disc, y - disc_radius, x - disc_radius, y + disc_radius, x + disc_radius)
                canvas.itemconfig(disc, fill=speed_color(speed))
        ms = (time.perf_counter() - t0) * 1000
        state["ms"] = 0.9 * state["ms"] + 0.1 * ms
        label.config(text=f"balls {sim.positions.shape[0]:3d}   step {state['ms']:5.1f} ms   "
                          f"target {period * 1000:.0f} ms   hold mouse to spawn, space pause, c clear")
        root.after(max(1, int((period - (time.perf_counter() - t0)) * 1000)), tick)

    tick()
    root.mainloop()


if __name__ == "__main__":
    main()
