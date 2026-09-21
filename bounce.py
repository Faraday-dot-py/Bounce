#!/usr/bin/env python3
"""N-ball bounce simulation. Physics runs on an explicit list of ball
states (arbitrary length, not stored in the grid). The grid G is a fixed
3-channel field derived fresh from that list each frame -- its shape never
changes with the number of balls:
  0: combined likelihood a ball occupies (i, j) -- what gets rendered
  1: probability-weighted mean vx of whatever ball(s) cover (i, j)
  2: probability-weighted mean vy of whatever ball(s) cover (i, j)
Each ball's likelihood is splatted as a disk of the given radius (linear
falloff from center to edge). Where balls' disks overlap, channels 1-2
hold the probability-weighted average velocity, not any one ball's exact
value -- the grid is a rendering/summary view, not a lossless encoding.
Channel 0 saturates smoothly toward 1 as overlapping weight accumulates
(1 - exp(-sum)), rather than growing unbounded under heavy local density.

Wall and ball-ball contact are continuous, force-based (a C1-smooth
penalty force that turns on at overlap and grows with penetration depth),
integrated via symplectic Euler alongside gravity -- no instantaneous
velocity/position corrections anywhere, so total energy stays in a
bounded oscillation instead of drifting.
"""
import argparse
import math
import random
import sys
import time

PROB, VX, VY = 0, 1, 2
NUM_CHANNELS = 3


def make_grid(n):
    return [[[0.0] * NUM_CHANNELS for _ in range(n)] for _ in range(n)]


def clear_grid(G, n):
    for row in G:
        for cell in row:
            for c in range(NUM_CHANNELS):
                cell[c] = 0.0


def splat_ball(G, n, state, radius):
    """Splat one ball's likelihood as a disk of the given radius (linear
    falloff from center to edge), accumulating weighted vx/vy sums into
    channels 1-2 so they can be normalized into an average once every
    ball has been splatted."""
    x, y, vx, vy = state["x"], state["y"], state["vx"], state["vy"]
    r = max(radius, 1e-6)
    i_lo = max(0, math.floor(x - r))
    i_hi = min(n - 1, math.ceil(x + r))
    j_lo = max(0, math.floor(y - r))
    j_hi = min(n - 1, math.ceil(y + r))
    for i in range(i_lo, i_hi + 1):
        for j in range(j_lo, j_hi + 1):
            d = math.hypot(i - x, j - y)
            if d > r:
                continue
            w = 1.0 - d / r
            G[i][j][PROB] += w
            G[i][j][VX] += w * vx
            G[i][j][VY] += w * vy


def splat_all(G, n, balls, radius):
    clear_grid(G, n)
    for state in balls:
        splat_ball(G, n, state, radius)
    for row in G:
        for cell in row:
            if cell[PROB] > 0.0:
                cell[VX] /= cell[PROB]
                cell[VY] /= cell[PROB]
                # saturate after using the raw weight sum as the averaging
                # denominator above -- smooth (C-infinity), asymptotes to 1
                # instead of growing unbounded under many overlapping balls
                cell[PROB] = 1.0 - math.exp(-cell[PROB])


def penalty_force(penetration, stiffness):
    """C1-smooth repulsive force: zero (and zero-slope) at penetration <= 0,
    growing as the square of penetration depth once bodies overlap. No jump
    in value or derivative at the contact boundary."""
    if penetration <= 0.0:
        return 0.0
    return stiffness * penetration * penetration


def wall_force(state, n, radius, stiffness):
    """Continuous force pushing a ball back once it overlaps a wall, in
    place of the old hard position-reflect. Walls are treated as immovable,
    so all of the force/energy goes into the ball."""
    lo, hi = 0.0, n - 1.0
    fx = fy = 0.0
    if state["x"] - lo < radius:
        fx += penalty_force(radius - (state["x"] - lo), stiffness)
    elif hi - state["x"] < radius:
        fx -= penalty_force(radius - (hi - state["x"]), stiffness)
    if state["y"] - lo < radius:
        fy += penalty_force(radius - (state["y"] - lo), stiffness)
    elif hi - state["y"] < radius:
        fy -= penalty_force(radius - (hi - state["y"]), stiffness)
    return fx, fy


def ball_pair_force(b1, b2, radius, stiffness):
    """Continuous repulsive force between two overlapping balls, directed
    along their center line. Returns the force applied to b2 -- apply its
    negation to b1 (Newton's third law, conserves momentum)."""
    dx = b2["x"] - b1["x"]
    dy = b2["y"] - b1["y"]
    dist = math.hypot(dx, dy)
    if dist < 1e-9:
        nx, ny = 1.0, 0.0
        dist = 0.0
    else:
        nx, ny = dx / dist, dy / dist
    f = penalty_force(2 * radius - dist, stiffness)
    return f * nx, f * ny


def compute_forces(balls, n, gravity, radius, stiffness):
    """Net force (== acceleration, unit mass) on each ball this step:
    gravity + wall contact + pairwise ball contact."""
    forces = [[gravity, 0.0] for _ in balls]
    for idx, state in enumerate(balls):
        fx, fy = wall_force(state, n, radius, stiffness)
        forces[idx][0] += fx
        forces[idx][1] += fy
    for i in range(len(balls)):
        for j in range(i + 1, len(balls)):
            fx, fy = ball_pair_force(balls[i], balls[j], radius, stiffness)
            forces[i][0] -= fx
            forces[i][1] -= fy
            forces[j][0] += fx
            forces[j][1] += fy
    return forces


def integrate(balls, forces, dt):
    """Semi-implicit (symplectic) Euler: velocity updated from force first,
    then position updated from the new velocity. Keeps total energy
    oscillating in a bounded band instead of drifting."""
    for state, (fx, fy) in zip(balls, forces):
        state["vx"] += fx * dt
        state["vy"] += fy * dt
        state["x"] += state["vx"] * dt
        state["y"] += state["vy"] * dt


def step(G, n, balls, dt, gravity, radius, stiffness, substeps):
    """Advance by dt total, but in `substeps` smaller physics steps: the
    penalty force is stiff enough that symplectic Euler needs a finer
    resolution than one step per render frame to stay stable."""
    sub_dt = dt / substeps
    for _ in range(substeps):
        forces = compute_forces(balls, n, gravity, radius, stiffness)
        integrate(balls, forces, sub_dt)
    splat_all(G, n, balls, radius)


def color_for(value):
    # 0 -> dim blue-gray, 1 -> bright red/yellow, via truecolor ANSI
    v = max(0.0, min(1.0, value))
    r = int(40 + v * 215)
    g = int(40 + (1 - abs(v - 0.5) * 2) * 100)
    b = int(80 * (1 - v))
    return f"\x1b[38;2;{r};{g};{b}m"


RESET = "\x1b[0m"
CLEAR = "\x1b[H\x1b[2J"


def render(G, n):
    lines = [CLEAR]
    for i in range(n):
        cells = []
        for j in range(n):
            v = G[i][j][PROB]
            cells.append(f"{color_for(v)}{v:0.2f}{RESET}")
        lines.append(" ".join(cells))
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


def init_balls(num_balls, n, vy, rng):
    balls = []
    lo, hi = 0.0, n - 1.0
    for _ in range(num_balls):
        x = rng.uniform(lo, hi)
        y = rng.uniform(lo, hi)
        vx = rng.uniform(-vy, vy)
        vy_ = rng.uniform(-vy, vy)
        balls.append({"x": x, "y": y, "vx": vx, "vy": vy_})
    return balls


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=15, help="grid size")
    ap.add_argument("--balls", type=int, default=2, help="number of balls")
    ap.add_argument("--dt", type=float, default=0.15, help="time step")
    ap.add_argument("--fps", type=float, default=12.0, help="frames per second")
    ap.add_argument("--vy", type=float, default=2.3, help="max initial speed magnitude, per axis")
    ap.add_argument("--gravity", type=float, default=9.0, help="acceleration toward bottom (row n-1)")
    ap.add_argument("--radius", type=float, default=0.75, help="ball radius, in grid cells (visual size and hitbox)")
    ap.add_argument("--stiffness", type=float, default=400.0, help="penalty-force spring constant for wall/ball contact")
    ap.add_argument("--substeps", type=int, default=8, help="physics sub-steps per rendered frame, for integrator stability")
    ap.add_argument("--seed", type=int, default=4738, help="random seed for ball starts/velocities")
    args = ap.parse_args()

    n = args.n
    rng = random.Random(args.seed)
    balls = init_balls(args.balls, n, args.vy, rng)
    G = make_grid(n)
    splat_all(G, n, balls, args.radius)
    frame_delay = 1.0 / args.fps

    try:
        while True:
            render(G, n)
            time.sleep(frame_delay)
            step(G, n, balls, args.dt, args.gravity, args.radius, args.stiffness, args.substeps)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
