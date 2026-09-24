import numpy as np

CENTER = 500.0


def accel(pos, eps, g=1.0):
    d = pos[None, :, :] - pos[:, None, :]
    r2 = (d ** 2).sum(-1) + eps ** 2
    inv = r2 ** -1.5
    np.fill_diagonal(inv, 0.0)
    return g * (d * inv[..., None]).sum(1)


def energy(pos, vel, eps, g=1.0):
    d = pos[None, :, :] - pos[:, None, :]
    r = np.sqrt((d ** 2).sum(-1) + eps ** 2)
    iu = np.triu_indices(len(pos), 1)
    return 0.5 * (vel ** 2).sum() - g * (1.0 / r[iu]).sum()


def init_bodies(n_bodies, rng, spread=5.0, speed=0.5):
    pos = CENTER + rng.uniform(-spread, spread, (n_bodies, 2))
    vel = rng.normal(0.0, speed, (n_bodies, 2))
    vel -= vel.mean(0)
    return pos, vel


def rollout(pos, vel, steps, dt=0.1, substeps=4, eps=0.5, g=1.0):
    h = dt / substeps
    ps, vs = [pos.copy()], [vel.copy()]
    pos, vel = pos.copy(), vel.copy()
    a = accel(pos, eps, g)
    for _ in range(steps):
        for _ in range(substeps):
            vel += 0.5 * h * a
            pos += h * vel
            a = accel(pos, eps, g)
            vel += 0.5 * h * a
        ps.append(pos.copy())
        vs.append(vel.copy())
    return np.stack(ps), np.stack(vs)


def make_dataset(num, ball_range, steps, seed, **kw):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(num):
        n = int(rng.integers(ball_range[0], ball_range[1] + 1))
        p, v = init_bodies(n, rng)
        out.append(rollout(p, v, steps, **kw))
    return out
