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

import numpy as np

PROB, VX, VY = 0, 1, 2
NUM_CHANNELS = 3


def make_grid(n):
    return [[[0.0] * NUM_CHANNELS for _ in range(n)] for _ in range(n)]


def clear_grid(G, n):
    for row in G:
        for cell in row:
            for c in range(NUM_CHANNELS):
                cell[c] = 0.0


def splat_ball(grid, n, x, y, vx, vy, radius):
    """Splat one ball's likelihood as a disk of the given radius (linear
    falloff from center to edge) into the (n, n, NUM_CHANNELS) numpy grid,
    accumulating weighted vx/vy sums into channels 1-2 so they can be
    normalized into an average once every ball has been splatted."""
    r = max(radius, 1e-6)
    i_lo = max(0, math.floor(x - r))
    i_hi = min(n - 1, math.ceil(x + r))
    j_lo = max(0, math.floor(y - r))
    j_hi = min(n - 1, math.ceil(y + r))
    if i_lo > i_hi or j_lo > j_hi:
        return
    ii = np.arange(i_lo, i_hi + 1).reshape(-1, 1)
    jj = np.arange(j_lo, j_hi + 1).reshape(1, -1)
    d = np.hypot(ii - x, jj - y)
    w = np.where(d <= r, 1.0 - d / r, 0.0)
    grid[i_lo:i_hi + 1, j_lo:j_hi + 1, PROB] += w
    grid[i_lo:i_hi + 1, j_lo:j_hi + 1, VX] += w * vx
    grid[i_lo:i_hi + 1, j_lo:j_hi + 1, VY] += w * vy


def splat_all(G, n, balls, radius):
    grid = np.zeros((n, n, NUM_CHANNELS))
    for state in balls:
        splat_ball(grid, n, state["x"], state["y"], state["vx"], state["vy"], radius)
    prob = grid[:, :, PROB]
    with np.errstate(invalid="ignore"):
        grid[:, :, VX] = np.divide(grid[:, :, VX], prob, out=np.zeros_like(prob), where=prob > 0.0)
        grid[:, :, VY] = np.divide(grid[:, :, VY], prob, out=np.zeros_like(prob), where=prob > 0.0)
    # saturate after using the raw weight sum as the averaging denominator
    # above -- smooth (C-infinity), asymptotes to 1 instead of growing
    # unbounded under many overlapping balls
    grid[:, :, PROB] = 1.0 - np.exp(-prob)
    for i in range(n):
        for j in range(n):
            cell = G[i][j]
            cell[PROB] = float(grid[i, j, PROB])
            cell[VX] = float(grid[i, j, VX])
            cell[VY] = float(grid[i, j, VY])


def penalty_force(penetration, stiffness):
    """C1-smooth repulsive force: zero (and zero-slope) at penetration <= 0,
    growing as the square of penetration depth once bodies overlap. No jump
    in value or derivative at the contact boundary. Vectorized: penetration
    may be a numpy array."""
    return np.where(penetration > 0.0, stiffness * penetration * penetration, 0.0)


def wall_force(xs, ys, n, radius, stiffness):
    """Continuous force pushing each ball back once it overlaps a wall, in
    place of the old hard position-reflect. Walls are treated as immovable,
    so all of the force/energy goes into the ball. xs/ys are numpy arrays,
    one entry per ball."""
    lo, hi = 0.0, n - 1.0
    left_x = (xs - lo) < radius
    right_x = ~left_x & ((hi - xs) < radius)
    fx = np.where(left_x, penalty_force(radius - (xs - lo), stiffness), 0.0)
    fx -= np.where(right_x, penalty_force(radius - (hi - xs), stiffness), 0.0)
    left_y = (ys - lo) < radius
    right_y = ~left_y & ((hi - ys) < radius)
    fy = np.where(left_y, penalty_force(radius - (ys - lo), stiffness), 0.0)
    fy -= np.where(right_y, penalty_force(radius - (hi - ys), stiffness), 0.0)
    return fx, fy


def ball_pair_forces(xs, ys, radius, stiffness):
    """Continuous repulsive force between every pair of overlapping balls,
    directed along their center line. Returns (fx, fy), each an (m, m)
    matrix where entry [i, j] is the force applied to ball j by ball i
    (antisymmetric: entry [j, i] == -entry [i, j], diagonal is zero)."""
    dx = xs.reshape(1, -1) - xs.reshape(-1, 1)
    dy = ys.reshape(1, -1) - ys.reshape(-1, 1)
    dist = np.hypot(dx, dy)
    close = dist < 1e-9
    safe_dist = np.where(close, 1.0, dist)
    nx = np.where(close, 1.0, dx / safe_dist)
    ny = np.where(close, 0.0, dy / safe_dist)
    dist = np.where(close, 0.0, dist)
    f = penalty_force(2 * radius - dist, stiffness)
    np.fill_diagonal(f, 0.0)
    return f * nx, f * ny


def compute_forces(balls, n, gravity, radius, stiffness):
    """Net force (== acceleration, unit mass) on each ball this step:
    gravity + wall contact + pairwise ball contact."""
    m = len(balls)
    if m == 0:
        return []
    xs = np.array([b["x"] for b in balls])
    ys = np.array([b["y"] for b in balls])
    wfx, wfy = wall_force(xs, ys, n, radius, stiffness)
    base_x = gravity + wfx
    base_y = wfy
    if m > 1:
        pfx, pfy = ball_pair_forces(xs, ys, radius, stiffness)
        # this system is numerically chaotic (stiff contacts), so matching
        # the original loop's exact left-to-right float accumulation order
        # (base term first, then each pairwise contribution in ascending
        # ball-index order) matters, not just the summation's math value --
        # cumsum (unlike .sum, which uses pairwise summation for long rows)
        # preserves that order
        chain_x = np.concatenate([base_x.reshape(-1, 1), -pfx], axis=1)
        chain_y = np.concatenate([base_y.reshape(-1, 1), -pfy], axis=1)
        fx = np.cumsum(chain_x, axis=1)[:, -1]
        fy = np.cumsum(chain_y, axis=1)[:, -1]
    else:
        fx, fy = base_x, base_y
    return np.stack([fx, fy], axis=1).tolist()


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
