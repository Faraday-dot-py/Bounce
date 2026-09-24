"""Ball detection that resolves overlapping balls: a greedy residual fit of
the frame's desaturated PROB (-log(1 - PROB)) with exact splats, adding one
ball per iteration until a new ball no longer reduces the squared error by
more than `tau`, then a prune/merge pass. On seeds 4738-4785 the count is
right in 100% of frames vs 85-92% for find_token_positions (which merges
balls within ~2 cells); no false splits on isolated balls (docs/debugging/
experiment-log.md). Velocities are solved by linear least squares on VX/VY
given the fitted positions, so overlapping balls get their own velocity.
Numpy/scipy on CPU, ~19 ms per frame (4 balls), not differentiable
(init preprocessing only). Scenes with more than `max_tokens` seed
detections skip the fit (cost grows steeply with ball count) and keep the
plain detections."""
import numpy as np
import torch
from scipy.optimize import least_squares

from model.token_detect import find_token_positions, read_token_velocities
from model.token_refine import refine_positions

_EPS = 1e-9


def _desat(prob):
    return -np.log(np.clip(1.0 - prob, _EPS, 1.0))


def _splats(n, pos, r):
    ii = np.arange(n, dtype=np.float64).reshape(1, -1, 1)
    jj = np.arange(n, dtype=np.float64).reshape(1, 1, -1)
    dx = ii - pos[:, 0].reshape(-1, 1, 1)
    dy = jj - pos[:, 1].reshape(-1, 1, 1)
    d = np.hypot(dx, dy)
    inside = d <= r
    return np.where(inside, 1.0 - d / r, 0.0), dx, dy, d, inside


def _fit(w, pos, r, iters=30):
    n = w.shape[0]
    K = len(pos)

    def res(p):
        s = _splats(n, p.reshape(K, 2), r)[0]
        return (s.sum(0) - w).ravel()

    def jac(p):
        _, dx, dy, d, inside = _splats(n, p.reshape(K, 2), r)
        dd = np.where(d > 1e-9, d, 1.0)
        gx = np.where(inside & (d > 1e-9), dx / (dd * r), 0.0)
        gy = np.where(inside & (d > 1e-9), dy / (dd * r), 0.0)
        J = np.zeros((n * n, 2 * K))
        J[:, 0::2] = gx.reshape(K, -1).T
        J[:, 1::2] = gy.reshape(K, -1).T
        return J

    sol = least_squares(res, pos.ravel(), jac=jac, max_nfev=iters, xtol=1e-8, ftol=1e-10, gtol=1e-10)
    p = sol.x.reshape(K, 2)
    return p, float((sol.fun ** 2).sum())


_OFFS = np.arange(-1.0, 1.0 + 1e-9, 0.1)
_OX, _OY = np.meshgrid(_OFFS, _OFFS, indexing="ij")
_OX = _OX.ravel().reshape(-1, 1, 1)
_OY = _OY.ravel().reshape(-1, 1, 1)
_HALF = 3


def _grid_place(resid, center, r):
    """Best sub-cell (step 0.1) position within +-1 cell of `center` for one ball fit to `resid`."""
    n = resid.shape[0]
    i0, j0 = int(round(center[0])), int(round(center[1]))
    ii = np.arange(i0 - _HALF, i0 + _HALF + 1)
    jj = np.arange(j0 - _HALF, j0 + _HALF + 1)
    pad = np.zeros((n + 2 * _HALF + 2, n + 2 * _HALF + 2))
    pad[_HALF + 1:_HALF + 1 + n, _HALF + 1:_HALF + 1 + n] = resid
    win = pad[i0 + 1:i0 + 1 + len(ii), j0 + 1:j0 + 1 + len(jj)]
    cx, cy = center[0] + _OX, center[1] + _OY
    d = np.hypot(ii.reshape(1, -1, 1) - cx, jj.reshape(1, 1, -1) - cy)
    sp = np.where(d <= r, 1.0 - d / r, 0.0)
    score = (sp ** 2).sum((1, 2)) - 2 * (sp * win[None]).sum((1, 2))
    m = int(np.argmin(score))
    return np.array([center[0] + _OX.ravel()[m], center[1] + _OY.ravel()[m]])


def _peak_candidates(resid, r, k=3, sep=1.0):
    n = resid.shape[0]
    rr = resid.copy()
    out = []
    gi = np.arange(n).reshape(-1, 1)
    gj = np.arange(n).reshape(1, -1)
    for _ in range(k):
        idx = np.unravel_index(np.argmax(rr), rr.shape)
        if rr[idx] <= 0:
            break
        out.append(_grid_place(resid, np.array(idx, dtype=np.float64), r))
        rr[(gi - idx[0]) ** 2 + (gj - idx[1]) ** 2 <= sep ** 2] = 0
    return out


def _polish(w, pos, r, sweeps=1):
    n = w.shape[0]
    pos = pos.copy()
    for _ in range(sweeps):
        for k in range(len(pos)):
            others = np.delete(pos, k, axis=0)
            resid = w - (_splats(n, others, r)[0].sum(0) if len(others) else 0.0)
            pos[k] = _grid_place(resid, pos[k], r)
    return _fit(w, pos, r)


def _merge_pass(w, pos, sse, r, tau, reach=3.0):
    n = w.shape[0]
    changed = True
    while changed and len(pos) > 1:
        changed = False
        for j in range(len(pos)):
            p, s_ = _polish(w, np.delete(pos, j, axis=0), r)
            if s_ <= sse + tau:
                pos, sse = p, s_
                changed = True
                break
        if changed:
            continue
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                if np.linalg.norm(pos[i] - pos[j]) > reach:
                    continue
                rest = np.delete(pos, [i, j], axis=0)
                resid = w - (_splats(n, rest, r)[0].sum(0) if len(rest) else 0.0)
                best = None
                for c in _peak_candidates(resid, r, k=2):
                    p, s_ = _fit(w, np.vstack([rest, c[None]]), r)
                    if best is None or s_ < best[1]:
                        best = (p, s_)
                if best is not None and best[1] <= sse + tau:
                    pos, sse = best
                    changed = True
                    break
            if changed:
                break
    return pos, sse


def greedy_fit(w, r=0.75, tau=1e-3, max_balls=8, seeds=None, resid_floor=0.01):
    n = w.shape[0]
    pos = np.zeros((0, 2)) if seeds is None else np.array(seeds, dtype=np.float64).reshape(-1, 2)
    sse = float((w ** 2).sum())
    if len(pos):
        pos, sse = _polish(w, pos, r)
    while len(pos) < max_balls:
        resid = w - (_splats(n, pos, r)[0].sum(0) if len(pos) else 0.0)
        if resid.max() < resid_floor:
            break
        best = None
        for c in _peak_candidates(resid, r):
            p, s = _polish(w, np.vstack([pos, c[None]]), r)
            if best is None or s < best[1]:
                best = (p, s)
        if best is None or sse - best[1] < tau:
            break
        pos, sse = best
        if len(pos) > 1:
            pos, sse = _merge_pass(w, pos, sse, r, tau)
    return pos, sse


def _velocities(frame, pos, w, r):
    n = w.shape[0]
    if len(pos) == 0:
        return np.zeros((0, 2))
    S = _splats(n, pos, r)[0].reshape(len(pos), -1).T
    out = np.zeros((len(pos), 2))
    for c in range(2):
        sv = (w * frame[1 + c]).ravel()
        out[:, c] = np.linalg.lstsq(S, sv, rcond=None)[0]
    return out


def _safe_velocities(f1, pos, w, radius, max_disagreement=4.0, max_speed=20.0):
    """`_velocities`, except where the least-squares solve is ill-conditioned
    (stacked or nearly coincident balls make the splat columns collinear and
    the solution explode: errors of 20+ cells/s on settled and clustered
    scenes, and 1e8 training losses). A token whose solved velocity is
    non-finite, faster than `max_speed`, or more than `max_disagreement` from
    the windowed VX/VY readout falls back to the readout."""
    solved = _velocities(f1, pos, w, radius)
    if len(pos) == 0:
        return solved
    readout = read_token_velocities(torch.from_numpy(f1).float(), torch.from_numpy(pos).float()).double().numpy()
    bad = (~np.isfinite(solved).all(axis=1) | (np.linalg.norm(solved, axis=1) > max_speed)
           | (np.linalg.norm(solved - readout, axis=1) > max_disagreement))
    return np.where(bad[:, None], readout, solved)


def _seeds(prob, radius):
    return find_token_positions(torch.from_numpy(prob.astype(np.float32)), radius, 0.1).double().numpy()


def detect_balls(frame0, frame1, radius=0.75, dt=0.15, tau=1e-3, refine=True, return_all=False, max_tokens=12):
    f1 = frame1.detach().cpu().double().numpy()
    w1 = _desat(f1[0])
    seeds = _seeds(f1[0], radius)
    if len(seeds) > max_tokens:
        P = torch.from_numpy(seeds).float()
        V = torch.from_numpy(_safe_velocities(f1, seeds, w1, radius)).float()
        if refine and len(seeds):
            P = refine_positions(frame0.float(), frame1.float(), P, V, radius=radius, dt=dt)
            V = torch.from_numpy(_safe_velocities(f1, P.double().numpy(), w1, radius)).float()
        return (P, V, np.inf) if return_all else (P, V)
    pos, sse = np.zeros((0, 2)), np.inf
    if len(seeds):
        pos, sse = _polish(w1, seeds, radius)
    if not sse < 1e-6:
        pos, sse = greedy_fit(w1, radius, tau=tau, seeds=seeds if len(seeds) else None)
    n = w1.shape[0]
    if not np.isfinite(pos).all() or (pos < -1.0).any() or (pos > n).any():
        pos = seeds
        sse = np.inf
    vel = _safe_velocities(f1, pos, w1, radius)
    P = torch.from_numpy(pos).float()
    V = torch.from_numpy(vel).float()
    if refine and len(pos):
        P = refine_positions(frame0.float(), frame1.float(), P, V, radius=radius, dt=dt)
        V = torch.from_numpy(_safe_velocities(f1, P.double().numpy(), w1, radius)).float()
    return (P, V, sse) if return_all else (P, V)
