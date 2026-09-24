"""Sub-cell refinement of frame-1 token positions by fitting the two frames'
exact splats (bounce.splat_ball semantics, PROB saturation undone), with no
simulator constants: velocities come from the frames' own VX/VY channels and
the frame-0 position is p1 - dt * (v0 + v1) / 2. Cuts init position error
from 0.244 to ~0.06 cells on seeds 4738-4785 (docs/debugging/
experiment-log.md). Numpy on CPU, ~12 ms per frame for 4 balls in "joint"
mode; not differentiable (init preprocessing only). Falls back to the
input positions if the fit fails or moves a token more than 1.0 cell."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

_LAM = 3e-3
_LEVELS = ((0.7, 0.1), (0.1, 0.02), (0.02, 0.004))


def _desat(prob):
    return -np.log(np.clip(1.0 - prob, 1e-9, 1.0))


def _read_vel(w, vx, vy, pos, win=1.0):
    n = w.shape[0]
    ii = np.arange(n, dtype=np.float64).reshape(-1, 1)
    jj = np.arange(n, dtype=np.float64).reshape(1, -1)
    out = np.zeros((len(pos), 2))
    ok = np.zeros(len(pos), bool)
    for k, c in enumerate(pos):
        m = ((ii - c[0]) ** 2 + (jj - c[1]) ** 2 <= win ** 2) * w
        t = m.sum()
        if t > 1e-6:
            out[k] = [(m * vx).sum() / t, (m * vy).sum() / t]
            ok[k] = True
    return out, ok


def _splat(n, x, y, r):
    ii = np.arange(n, dtype=np.float64).reshape(-1, 1)
    jj = np.arange(n, dtype=np.float64).reshape(1, -1)
    d = np.hypot(ii - x, jj - y)
    return np.where(d <= r, 1.0 - d / r, 0.0)


def _fit(frames, p, shifts, vels, use, r, use_vel, sweeps=2):
    """frames: list of (w, sx, sy) with w the desaturated weight and s* = w*vel
    sums; shifts[f]: (H,K,2) hypotheses, offset of each ball's frame-f position from p;
    vels[f]: (K,2) ball velocities in frame f; use[f]: (K,) bool."""
    n = frames[0][0].shape[0]
    K = len(p)
    p = p.copy()
    p_init = p.copy()
    H = shifts[0].shape[0]
    hsel = np.zeros(K, int)
    sh = lambda f, o: shifts[f][hsel[o], o]

    def render(f, q):
        W = np.zeros((n, n)); SX = np.zeros((n, n)); SY = np.zeros((n, n))
        for o in range(K):
            if not use[f][o]:
                continue
            w = _splat(n, q[o, 0] + sh(f, o)[0], q[o, 1] + sh(f, o)[1], r)
            W += w; SX += w * vels[f][o, 0]; SY += w * vels[f][o, 1]
        return W, SX, SY

    def total_cost(q):
        c = 0.0
        for f in range(len(frames)):
            W, SX, SY = render(f, q)
            c += ((W - frames[f][0]) ** 2).sum()
            if use_vel:
                c += ((SX - frames[f][1]) ** 2).sum() + ((SY - frames[f][2]) ** 2).sum()
        return c

    def ball_cost(k, h, half, step):
        ofs = np.arange(-half, half + 1e-9, step)
        cost = 0.0
        for f in range(len(frames)):
            if not use[f][k]:
                continue
            others = np.zeros((3, n, n))
            for o in range(K):
                if o == k or not use[f][o]:
                    continue
                w = _splat(n, p[o, 0] + sh(f, o)[0], p[o, 1] + sh(f, o)[1], r)
                others[0] += w; others[1] += w * vels[f][o, 0]; others[2] += w * vels[f][o, 1]
            cx = p[k, 0] + shifts[f][h, k, 0]; cy = p[k, 1] + shifts[f][h, k, 1]
            i0, i1 = max(0, int(round(cx)) - 3), min(n, int(round(cx)) + 4)
            j0, j1 = max(0, int(round(cy)) - 3), min(n, int(round(cy)) + 4)
            if i0 >= i1 or j0 >= j1:
                continue
            wi = np.arange(i0, i1, dtype=np.float64).reshape(1, 1, -1, 1)
            wj = np.arange(j0, j1, dtype=np.float64).reshape(1, 1, 1, -1)
            px = (cx + ofs).reshape(-1, 1, 1, 1); py = (cy + ofs).reshape(1, -1, 1, 1)
            pred = np.where(np.hypot(wi - px, wj - py) <= r, 1.0 - np.hypot(wi - px, wj - py) / r, 0.0)
            terms = [(pred, frames[f][0] - others[0])]
            if use_vel:
                terms += [(pred * vels[f][k, 0], frames[f][1] - others[1]),
                          (pred * vels[f][k, 1], frames[f][2] - others[2])]
            for pr, resid in terms:
                rw = resid[i0:i1, j0:j1]
                cost = cost + ((pr - rw[None, None]) ** 2).sum((2, 3)) + (resid ** 2).sum() - (rw ** 2).sum()
        if not np.isscalar(cost) and _LAM:
            gx = (p[k, 0] + ofs - p_init[k, 0]).reshape(-1, 1); gy = (p[k, 1] + ofs - p_init[k, 1]).reshape(1, -1)
            cost = cost + _LAM * (gx ** 2 + gy ** 2)
        return cost, ofs

    for _ in range(sweeps):
        for k in range(K):
            best = None
            for h in range(H):
                pk = p[k].copy()
                cost, ofs = ball_cost(k, h, *_LEVELS[0])
                if np.isscalar(cost):
                    continue
                a_, b_ = np.unravel_index(cost.argmin(), cost.shape)
                if best is None or cost.min() < best[0]:
                    best = (cost.min(), h, pk + [ofs[a_], ofs[b_]])
            if best is None:
                continue
            hsel[k] = best[1]; p[k] = best[2]
            for half, step in _LEVELS[1:]:
                cost, ofs = ball_cost(k, hsel[k], half, step)
                a_, b_ = np.unravel_index(cost.argmin(), cost.shape)
                p[k] = p[k] + [ofs[a_], ofs[b_]]
    return p, total_cost(p)


def refine_positions(frame0, frame1, pos1, vel1, radius=0.75, dt=0.15, mode="joint", hyp=True):
    """Sub-cell refinement of frame-1 token positions. mode: 'single' (frame-1
    exact-splat fit), 'joint' (two-frame fit, frame-0 ball at p1 - dt*(v0+v1)/2),
    'joint_vel' (joint + VX/VY-channel residuals)."""
    N = int(pos1.shape[0])
    if N == 0:
        return pos1.clone()
    dtype = pos1.dtype
    try:
        f0 = frame0.detach().cpu().double().numpy(); f1 = frame1.detach().cpu().double().numpy()
        p1 = pos1.detach().cpu().double().numpy().copy()
        v1 = vel1.detach().cpu().double().numpy().copy()
        n = f1.shape[1]
        w0, w1 = _desat(f0[0]), _desat(f1[0])
        fr1 = (w1, w1 * f1[1], w1 * f1[2])
        fr0 = (w0, w0 * f0[1], w0 * f0[2])
        v1r, ok1 = _read_vel(w1, f1[1], f1[2], p1)
        v1 = np.where(ok1[:, None], v1r, v1)
        zeros = np.zeros((N, 2))
        allT = np.ones(N, bool)
        if mode == "single":
            p, _ = _fit([fr1], p1, [zeros[None]], [v1], [allT], radius, False)
        else:
            use_vel = mode == "joint_vel"
            kappas = (0.5, 0.0, 1.0) if hyp else (0.5,)
            v0 = v1.copy(); ok0 = np.zeros(N, bool)
            p = p1.copy()
            for it in range(2):
                if it:
                    v1r, ok1 = _read_vel(w1, f1[1], f1[2], p)
                    v1 = np.where(ok1[:, None], v1r, v1)
                q = p - v1 * dt
                if it == 0:
                    try:
                        from model.token_detect import find_token_positions
                        det = find_token_positions(frame0[0].detach().cpu(), radius, 0.1).double().numpy()
                    except Exception:
                        det = np.zeros((0, 2))
                    if len(det) == N:
                        d = np.linalg.norm(q[:, None] - det[None], axis=-1)
                        r_, c_ = linear_sum_assignment(d)
                        qa = q.copy(); qa[r_] = det[c_]
                        keep = np.linalg.norm(qa - q, axis=1) <= 1.5
                        q = np.where(keep[:, None], qa, q)
                v0r, ok0 = _read_vel(w0, f0[1], f0[2], q)
                v0 = np.where(ok0[:, None], v0r, v1)
                sh0 = np.stack([-dt * (v1 + kp * (v0 - v1)) for kp in kappas])
                p, _ = _fit([fr1, fr0], p, [zeros[None].repeat(len(kappas), 0), sh0], [v1, v0], [allT, ok0], radius, use_vel)
        move = np.linalg.norm(p - p1, axis=1)
        good = np.isfinite(p).all(1) & (move <= 1.0)
        out = np.where(good[:, None], p, p1)
        return torch.from_numpy(out).to(dtype=dtype, device=pos1.device)
    except Exception:
        return pos1.clone()
