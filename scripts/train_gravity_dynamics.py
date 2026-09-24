"""Generality test: train the unmodified TokenFreeDynamics on a softened
2D N-body gravity sim from (pos, vel) trajectories (no grid, no walls in
use: box n=1000, bodies near the centre). Reports free-rollout position
error and energy drift vs a constant-velocity baseline."""
import argparse
import json

import numpy as np
import torch

from model.token_free import TokenFreeDynamics
from scripts import gravity_sim as gs


def unroll(dyn, pos, vel, k, dt):
    hidden = torch.zeros(pos.shape[0], dyn.hidden_dim)
    ps, vs = [], []
    for _ in range(k):
        dp, dv, hidden = dyn(pos, vel, hidden)
        pos = pos + vel * dt + dp
        vel = vel + dv
        ps.append(pos)
        vs.append(vel)
    return torch.stack(ps), torch.stack(vs)


def evaluate(dyn, data, horizon, dt, eps):
    err = np.zeros(horizon)
    cv_err = np.zeros(horizon)
    e_model, e_true = np.zeros(horizon), np.zeros(horizon)
    with torch.no_grad():
        for P, V in data:
            p0 = torch.tensor(P[0], dtype=torch.float32)
            v0 = torch.tensor(V[0], dtype=torch.float32)
            ps, vs = unroll(dyn, p0, v0, horizon, dt)
            for t in range(horizon):
                err[t] += np.linalg.norm(ps[t].numpy() - P[t + 1], axis=1).mean()
                cv_err[t] += np.linalg.norm(P[0] + V[0] * dt * (t + 1) - P[t + 1], axis=1).mean()
                e_model[t] += gs.energy(ps[t].numpy().astype(np.float64), vs[t].numpy().astype(np.float64), eps)
                e_true[t] += gs.energy(P[t + 1], V[t + 1], eps)
    n = len(data)
    return {"err": (err / n).tolist(), "const_vel_err": (cv_err / n).tolist(),
            "energy_model": (e_model / n).tolist(), "energy_true": (e_true / n).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--min-bodies", type=int, default=3)
    ap.add_argument("--max-bodies", type=int, default=8)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k-start", type=int, default=4)
    ap.add_argument("--k-end", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--neighbor-radius", type=float, default=100.0)
    ap.add_argument("--no-pair-impulse", action="store_true")
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--out", type=str, default="results/gravity_test.json")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/gravity_dynamics.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    kw = dict(dt=args.dt, eps=args.eps)
    rng_n = (args.min_bodies, args.max_bodies)
    train = gs.make_dataset(args.train, rng_n, args.steps, args.seed, **kw)
    dyn = TokenFreeDynamics(n=1000, neighbor_radius=args.neighbor_radius, pair_impulse=not args.no_pair_impulse)
    opt = torch.optim.Adam(dyn.parameters(), lr=args.lr)
    for it in range(args.iters):
        k = int(round(args.k_start + (args.k_end - args.k_start) * it / max(1, args.iters - 1)))
        loss = 0.0
        for _ in range(args.batch):
            P, V = train[rng.integers(len(train))]
            t0 = int(rng.integers(0, args.steps - k + 1))
            p0 = torch.tensor(P[t0], dtype=torch.float32)
            v0 = torch.tensor(V[t0], dtype=torch.float32)
            ps, vs = unroll(dyn, p0, v0, k, args.dt)
            tp = torch.tensor(P[t0 + 1:t0 + k + 1], dtype=torch.float32)
            tv = torch.tensor(V[t0 + 1:t0 + k + 1], dtype=torch.float32)
            loss = loss + ((ps - tp) ** 2).mean() + 0.1 * ((vs - tv) ** 2).mean()
        loss = loss / args.batch
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dyn.parameters(), 1.0)
        opt.step()
        if it % 200 == 0:
            print(f"it {it} k {k} loss {loss.item():.6f}", flush=True)
    torch.save(dyn.state_dict(), args.checkpoint)
    res = {}
    for seed in (9000, 12000):
        data = gs.make_dataset(48, rng_n, args.steps, seed, **kw)
        res[str(seed)] = evaluate(dyn, data, 20, args.dt, args.eps)
        r = res[str(seed)]
        print(seed, "err@5/10/20", [round(r["err"][i], 4) for i in (4, 9, 19)],
              "constvel", [round(r["const_vel_err"][i], 4) for i in (4, 9, 19)],
              "E true/model @20", round(r["energy_true"][19], 4), round(r["energy_model"][19], 4), flush=True)
    res["args"] = vars(args)
    with open(args.out, "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
