#!/usr/bin/env python3
"""Two-ball bounce simulation. All state needed to compute the next frame
lives in the grid G. G[i][j] is a 11-channel cell:
  0: combined likelihood a ball occupies (i, j) -- what gets rendered
  1-5: ball A's (likelihood, x, y, vx, vy) contribution to this cell
  6-10: ball B's (likelihood, x, y, vx, vy) contribution to this cell
Each ball's likelihood is splatted as a disk of the given radius (so the
rendered blob is actually the ball's size, not just a point), and its exact
(x, y, vx, vy) is stamped into every cell it touches -- so a ball's full
state can be read back exactly from its own channels, with no drift near
walls where the disk gets clipped asymmetrically. No per-frame state is
kept outside the grid."""
import argparse
import math
import sys
import time

PROB = 0
A_PROB, A_X, A_Y, A_VX, A_VY = 1, 2, 3, 4, 5
B_PROB, B_X, B_Y, B_VX, B_VY = 6, 7, 8, 9, 10
NUM_CHANNELS = 11


def make_grid(n):
    return [[[0.0] * NUM_CHANNELS for _ in range(n)] for _ in range(n)]


def clear_grid(G, n):
    for row in G:
        for cell in row:
            for c in range(NUM_CHANNELS):
                cell[c] = 0.0


def splat_ball(G, n, state, radius, prob_ch, x_ch, y_ch, vx_ch, vy_ch):
    """Splat one ball's likelihood as a disk of the given radius (linear
    falloff from center to edge) into its own channels, add it into the
    combined display channel, and stamp its exact state into every cell it
    touches."""
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
            G[i][j][prob_ch] += w
            G[i][j][x_ch] = x
            G[i][j][y_ch] = y
            G[i][j][vx_ch] = vx
            G[i][j][vy_ch] = vy
            G[i][j][PROB] += w


def splat_all(G, n, a, b, radius):
    clear_grid(G, n)
    splat_ball(G, n, a, radius, A_PROB, A_X, A_Y, A_VX, A_VY)
    splat_ball(G, n, b, radius, B_PROB, B_X, B_Y, B_VX, B_VY)


def read_ball(G, n, prob_ch, x_ch, y_ch, vx_ch, vy_ch):
    """Read a ball's exact state back from the first touched cell -- every
    cell the ball's splat wrote to carries the same (x, y, vx, vy)."""
    for row in G:
        for cell in row:
            if cell[prob_ch] > 0.0:
                return {"x": cell[x_ch], "y": cell[y_ch], "vx": cell[vx_ch], "vy": cell[vy_ch]}
    return {"x": 0.0, "y": 0.0, "vx": 0.0, "vy": 0.0}


def move_and_bounce_off_walls(state, n, dt, gravity):
    state["vx"] += gravity * dt
    state["x"] += state["vx"] * dt
    state["y"] += state["vy"] * dt
    lo, hi = 0.0, n - 1.0
    if state["x"] < lo:
        state["x"] = lo + (lo - state["x"])
        state["vx"] = -state["vx"]
    elif state["x"] > hi:
        state["x"] = hi - (state["x"] - hi)
        state["vx"] = -state["vx"]
    if state["y"] < lo:
        state["y"] = lo + (lo - state["y"])
        state["vy"] = -state["vy"]
    elif state["y"] > hi:
        state["y"] = hi - (state["y"] - hi)
        state["vy"] = -state["vy"]


def resolve_collision(b1, b2, radius):
    """Equal-mass elastic collision between two circular balls of the given
    radius: swap the velocity component along the line connecting them, and
    push them apart so they stop overlapping."""
    dx = b2["x"] - b1["x"]
    dy = b2["y"] - b1["y"]
    dist = math.hypot(dx, dy)
    min_dist = 2 * radius
    if dist >= min_dist:
        return
    if dist < 1e-9:
        nx, ny = 1.0, 0.0
        dist = 0.0
    else:
        nx, ny = dx / dist, dy / dist

    v1n = b1["vx"] * nx + b1["vy"] * ny
    v2n = b2["vx"] * nx + b2["vy"] * ny
    if v1n - v2n > 0:  # balls approaching along the normal
        b1["vx"] += (v2n - v1n) * nx
        b1["vy"] += (v2n - v1n) * ny
        b2["vx"] += (v1n - v2n) * nx
        b2["vy"] += (v1n - v2n) * ny

    overlap = min_dist - dist
    b1["x"] -= nx * overlap / 2
    b1["y"] -= ny * overlap / 2
    b2["x"] += nx * overlap / 2
    b2["y"] += ny * overlap / 2


def step(G, n, dt, gravity, radius):
    a = read_ball(G, n, A_PROB, A_X, A_Y, A_VX, A_VY)
    b = read_ball(G, n, B_PROB, B_X, B_Y, B_VX, B_VY)
    move_and_bounce_off_walls(a, n, dt, gravity)
    move_and_bounce_off_walls(b, n, dt, gravity)
    resolve_collision(a, b, radius)
    splat_all(G, n, a, b, radius)


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", type=int, default=15, help="grid size")
    ap.add_argument("--dt", type=float, default=0.15, help="time step")
    ap.add_argument("--fps", type=float, default=12.0, help="frames per second")
    ap.add_argument("--vy", type=float, default=2.3, help="initial y speed (mirrored between the two balls)")
    ap.add_argument("--gravity", type=float, default=9.0, help="acceleration toward bottom (row n-1)")
    ap.add_argument("--radius", type=float, default=0.75, help="ball radius, in grid cells (visual size and hitbox)")
    args = ap.parse_args()

    n = args.n
    a = {"x": 0.0, "y": (n - 1) * 0.3, "vx": 0.0, "vy": args.vy}
    b = {"x": 0.0, "y": (n - 1) * 0.7, "vx": 0.0, "vy": -args.vy}
    G = make_grid(n)
    splat_all(G, n, a, b, args.radius)
    frame_delay = 1.0 / args.fps

    try:
        while True:
            render(G, n)
            time.sleep(frame_delay)
            step(G, n, args.dt, args.gravity, args.radius)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
