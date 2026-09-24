"""Real-time interactive token-model sim. Starts empty; hold the left mouse
button to spawn a stream of balls at the cursor. Space pauses, c clears,
q quits. Gravity pulls toward the bottom of the window.

Usage:
    PYTHONPATH=. python3 scripts/realtime_sim.py --checkpoint checkpoints/token_model_soup_b.pt
"""
import argparse
import time
import tkinter as tk

import numpy as np
import torch

from scripts.eval_free_rollout import load_model


class Sim:
    def __init__(self, model, max_balls, spawn_every, spawn_speed, seed):
        self.model = model
        self.n = model.n
        self.max_balls = max_balls
        self.spawn_every = spawn_every
        self.spawn_speed = spawn_speed
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
            return None
        with torch.no_grad():
            self.positions, self.velocities, self.hidden, grid = self.model.step_free(
                self.positions, self.velocities, self.hidden)
        return grid


def to_rgb(grid, n):
    if grid is None:
        return np.zeros((n, n, 3), dtype=np.uint8)
    prob = grid[0].numpy()
    speed = np.sqrt(grid[1].numpy() ** 2 + grid[2].numpy() ** 2)
    t = np.clip(speed / 10.0, 0.0, 1.0)
    color = np.stack([0.25 + 0.75 * t, 0.55 + 0.1 * t - 0.4 * t ** 2, 1.0 - 0.85 * t], axis=-1)
    return (np.clip(prob, 0, 1)[..., None] * color * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/token_model_soup_b.pt")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--scale", type=int, default=7)
    ap.add_argument("--fps", type=float, default=30.0, help="model steps per wall-clock second")
    ap.add_argument("--spawn-every", type=int, default=2, help="steps between spawns while held")
    ap.add_argument("--spawn-speed", type=float, default=2.3, help="max initial speed per axis")
    ap.add_argument("--max-balls", type=int, default=150)
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
    image_item = canvas.create_image(0, 0, anchor="nw")

    state = {"held": False, "cursor": (0.0, 0.0), "paused": False, "image": None, "ms": 0.0}

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
            grid = sim.step(state["cursor"] if state["held"] else None)
            rgb = to_rgb(grid, args.n)
            data = b"P6 %d %d 255\n" % (args.n, args.n) + rgb.tobytes()
            state["image"] = tk.PhotoImage(data=data, format="PPM").zoom(args.scale)
            canvas.itemconfig(image_item, image=state["image"])
        ms = (time.perf_counter() - t0) * 1000
        state["ms"] = 0.9 * state["ms"] + 0.1 * ms
        label.config(text=f"balls {sim.positions.shape[0]:3d}   step {state['ms']:5.1f} ms   "
                          f"target {period * 1000:.0f} ms   hold mouse to spawn, space pause, c clear")
        root.after(max(1, int((period - (time.perf_counter() - t0)) * 1000)), tick)

    tick()
    root.mainloop()


if __name__ == "__main__":
    main()
