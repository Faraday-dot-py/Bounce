# Token free-rollout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an observation-free `free_rollout` mode to `TokenModel` (tokens evolve in state space after `init_tokens`; the frame is never read back), train it, and score it against v9.

**Architecture:** New `TokenFreeDynamics` (radius-graph attention + GRU + zero-init delta head, plus wall-distance node features). `TokenModel(free_rollout=True)` gains `step_free`. `token_train.py --free-rollout` unrolls `step_free` with state loss. `scripts/eval_free_rollout.py` scores position error, identity swaps and OOD.

**Tech Stack:** PyTorch, pytest, existing `bounce.py` simulator, Polaris (SLURM) for training.

**Spec:** `docs/superpowers/specs/2026-09-23-token-free-rollout-design.md`

## Global Constraints

- v9 and v17 code paths stay byte-for-byte unchanged (`free_rollout` defaults to `False`).
- Seed 4738 for everything.
- Grid is `n x n`, walls at cell coordinates 0 and `n-1`; x is the row axis and gravity acts along +x (`bounce.compute_forces`).
- `radius=0.75`, `dt=0.15`, `neighbor_radius=4.0`, `hidden_dim=32`.
- No inline multi-line Python in Bash; scripts go in files.
- Polaris for training, not TIDE. Full-state checkpoints every epoch.
- Match surrounding code style; comments only for non-obvious WHY.

## Review Focus

- Zero tokens at init (`init_tokens` finds no balls): `step_free`, the loss and eval must not crash or produce NaN.
- One token (no graph edges): dynamics still returns finite deltas with a grad_fn.
- Token exactly on a wall (`x == 0`, `x == n-1`) and slightly outside: wall features saturate at 1, no NaN.
- Ball count differs from ground-truth ball count (detector merges two balls at init): eval must report it, not crash on shape mismatch.
- Rollout leaves the grid: `rasterize_tokens` still returns a finite grid.

---

### Task 1: TokenFreeDynamics

**Files:**
- Create: `model/token_free.py`
- Test: `tests/test_token_free.py`

**Interfaces:**
- Produces: `TokenFreeDynamics(n, hidden_dim=32, neighbor_radius=4.0, wall_range=3.0)`; `.hidden_dim`; `.forward(positions (N,2), velocities (N,2), hidden (N,H)) -> (delta_pos (N,2), delta_vel (N,2), new_hidden (N,H))`; `wall_features(positions, n, wall_range) -> (N,4)` module-level function.

- [ ] **Step 1: Write the failing tests**

```python
import torch

from model.token_free import TokenFreeDynamics, wall_features


def test_wall_features_zero_in_interior_and_saturate_at_wall():
    n, W = 20, 3.0
    pos = torch.tensor([[10.0, 10.0], [0.0, 10.0], [-1.0, 19.0], [19.0, 0.0]])
    f = wall_features(pos, n, W)
    assert f.shape == (4, 4)
    assert torch.all(f[0] == 0)
    assert f[1, 0] == 1.0 and f[1, 1] == 0.0
    assert f[2, 0] == 1.0 and f[2, 1] == 1.0
    assert f[3, 1] == 1.0 and f[3, 2] == 1.0
    assert torch.isfinite(f).all()


def test_zero_init_is_identity_delta():
    torch.manual_seed(4738)
    dyn = TokenFreeDynamics(n=20)
    pos = torch.rand(3, 2) * 10 + 5
    vel = torch.randn(3, 2)
    hidden = torch.zeros(3, dyn.hidden_dim)
    dp, dv, nh = dyn(pos, vel, hidden)
    assert torch.all(dp == 0) and torch.all(dv == 0)
    assert nh.shape == hidden.shape


def test_interior_translation_invariance():
    torch.manual_seed(4738)
    dyn = TokenFreeDynamics(n=40)
    torch.nn.init.normal_(dyn.delta_head.weight, std=0.1)
    pos = torch.tensor([[15.0, 15.0], [16.5, 15.5], [17.0, 19.0]])
    vel = torch.randn(3, 2)
    hidden = torch.randn(3, dyn.hidden_dim)
    a = dyn(pos, vel, hidden)
    b = dyn(pos + torch.tensor([3.0, -2.0]), vel, hidden)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-6)


def test_wall_proximity_changes_output():
    torch.manual_seed(4738)
    dyn = TokenFreeDynamics(n=20)
    torch.nn.init.normal_(dyn.delta_head.weight, std=0.1)
    vel = torch.zeros(1, 2)
    hidden = torch.zeros(1, dyn.hidden_dim)
    mid = dyn(torch.tensor([[10.0, 10.0]]), vel, hidden)[0]
    wall = dyn(torch.tensor([[0.5, 10.0]]), vel, hidden)[0]
    assert not torch.allclose(mid, wall)


def test_zero_and_one_token_keep_grad_fn():
    dyn = TokenFreeDynamics(n=20)
    for count in (0, 1):
        pos = torch.full((count, 2), 10.0)
        vel = torch.zeros(count, 2)
        hidden = torch.zeros(count, dyn.hidden_dim)
        dp, dv, nh = dyn(pos, vel, hidden)
        assert dp.shape == (count, 2)
        assert (dp.sum() + dv.sum() + nh.sum()).requires_grad
        assert torch.isfinite(nh).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. python3 -m pytest tests/test_token_free.py -q`
Expected: FAIL, `ModuleNotFoundError: model.token_free`

- [ ] **Step 3: Implement**

```python
import torch
import torch.nn as nn

from model.token_graph import build_radius_graph


def wall_features(positions, n, wall_range):
    """(N, 4) proximity to each wall, 0 beyond `wall_range` cells and 1 at
    or past the wall itself. Zero in the interior, so interior translation
    invariance is preserved; the wall itself is the one absolute-position
    fact the dynamics needs once observation correction is gone."""
    x, y = positions[:, 0], positions[:, 1]
    d = torch.stack([x, (n - 1) - x, y, (n - 1) - y], dim=1)
    return (wall_range - d.clamp(0.0, wall_range)) / wall_range


class TokenFreeDynamics(nn.Module):
    """TokenDynamics (radius-graph attention + GRU + zero-init delta head)
    with wall-proximity node features -- see
    docs/superpowers/specs/2026-09-23-token-free-rollout-design.md."""

    def __init__(self, n, hidden_dim=32, neighbor_radius=4.0, wall_range=3.0):
        super().__init__()
        self.n = n
        self.hidden_dim = hidden_dim
        self.neighbor_radius = neighbor_radius
        self.wall_range = wall_range
        node_dim = 2 + 4 + hidden_dim
        edge_dim = 2
        self.query = nn.Linear(node_dim, hidden_dim)
        self.key = nn.Linear(node_dim + edge_dim, hidden_dim)
        self.value = nn.Linear(node_dim + edge_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.delta_head = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, positions, velocities, hidden):
        n = positions.shape[0]
        node_state = torch.cat(
            [velocities, wall_features(positions, self.n, self.wall_range), hidden], dim=-1
        )
        q = self.query(node_state)

        edge_index = build_radius_graph(positions, self.neighbor_radius)
        self_loops = torch.arange(n, device=positions.device)
        self_loops = torch.stack([self_loops, self_loops], dim=0)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        attn_out = torch.zeros(n, self.hidden_dim, device=positions.device, dtype=positions.dtype)
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            rel_pos = positions[src] - positions[dst]
            edge_input = torch.cat([node_state[src], rel_pos], dim=-1)
            k = self.key(edge_input)
            v = self.value(edge_input)
            scores = (q[dst] * k).sum(dim=-1) / (self.hidden_dim ** 0.5)
            weights = torch.exp(scores - scores.max())
            denom = torch.zeros(n, device=positions.device, dtype=positions.dtype)
            denom = denom.index_add(0, dst, weights)
            weights = weights / denom[dst].clamp(min=1e-6)
            attn_out = attn_out.index_add(0, dst, weights.unsqueeze(-1) * v)

        new_hidden = self.gru(attn_out, hidden)
        delta = self.delta_head(new_hidden)
        return delta[:, :2], delta[:, 2:], new_hidden
```

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=. python3 -m pytest tests/test_token_free.py -q`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add model/token_free.py tests/test_token_free.py
git commit -m "feat: add TokenFreeDynamics with wall-proximity features"
```

---

### Task 2: TokenModel free_rollout mode

**Files:**
- Modify: `model/token_model.py` (`__init__` signature at line 55, dynamics selection at ~line 132, new method after `step`)
- Test: `tests/test_token_model.py` (append)

**Interfaces:**
- Consumes: `TokenFreeDynamics(n, hidden_dim, neighbor_radius)` from Task 1; `rasterize_tokens(positions, velocities, n, radius)`.
- Produces: `TokenModel(..., free_rollout=False)`; `TokenModel.step_free(positions, velocities, hidden) -> (final_pos, final_vel, new_hidden, next_grid)`. `init_tokens` unchanged.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_token_model.py`)

```python
import pytest


def test_free_rollout_step_free_coasts_at_init_and_ignores_frames():
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, free_rollout=True)
    positions = torch.tensor([[5.0, 5.0], [12.0, 8.0]])
    velocities = torch.tensor([[2.0, -1.0], [0.0, 1.0]])
    hidden = torch.zeros(2, model.dynamics.hidden_dim)
    pos, vel, new_hidden, grid = model.step_free(positions, velocities, hidden)
    assert torch.allclose(pos, positions + velocities * dt, atol=1e-5)
    assert torch.allclose(vel, velocities, atol=1e-5)
    assert grid.shape == (3, n, n)
    assert torch.allclose(grid, rasterize_tokens(pos, vel, n, radius))


def test_free_rollout_token_count_is_fixed_over_long_rollout():
    torch.manual_seed(4738)
    model = TokenModel(n=20, radius=0.75, dt=0.15, free_rollout=True)
    torch.nn.init.normal_(model.dynamics.delta_head.weight, std=0.1)
    positions = torch.rand(5, 2) * 15 + 2
    velocities = torch.randn(5, 2)
    hidden = torch.zeros(5, model.dynamics.hidden_dim)
    for _ in range(50):
        positions, velocities, hidden, grid = model.step_free(positions, velocities, hidden)
        assert positions.shape == (5, 2)
        assert torch.isfinite(grid).all()


def test_free_rollout_zero_tokens():
    model = TokenModel(n=20, radius=0.75, dt=0.15, free_rollout=True)
    positions = torch.zeros(0, 2)
    velocities = torch.zeros(0, 2)
    hidden = torch.zeros(0, model.dynamics.hidden_dim)
    pos, vel, new_hidden, grid = model.step_free(positions, velocities, hidden)
    assert pos.shape == (0, 2)
    assert grid.shape == (3, 20, 20)


@pytest.mark.parametrize("kwargs", [
    {"track_query": True}, {"territory_masking": True}, {"velocity_weight": 0.5},
])
def test_free_rollout_incompatible_options_raise(kwargs):
    with pytest.raises(ValueError):
        TokenModel(n=20, radius=0.75, dt=0.15, free_rollout=True, **kwargs)


def test_step_free_requires_free_rollout():
    model = TokenModel(n=20, radius=0.75, dt=0.15)
    with pytest.raises(RuntimeError):
        model.step_free(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, model.dynamics.hidden_dim))
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. python3 -m pytest tests/test_token_model.py -q -k "free_rollout or step_free"`
Expected: FAIL, `unexpected keyword argument 'free_rollout'`

- [ ] **Step 3: Implement**

In `model/token_model.py`: add `from model.token_free import TokenFreeDynamics` beside the other imports; add `track_query=False, free_rollout=False` to the signature (after `track_query`); after the existing `track_query` incompatibility block add:

```python
        self.free_rollout = free_rollout
        if free_rollout and (track_query or territory_masking or velocity_weight > 0.0):
            raise ValueError(
                "free_rollout is incompatible with track_query/territory_masking/velocity_weight "
                "-- it never reads an observed frame"
            )
```

Change the dynamics selection to:

```python
        if free_rollout:
            self.dynamics = TokenFreeDynamics(n=n, hidden_dim=hidden_dim, neighbor_radius=neighbor_radius)
        elif track_query:
            ...existing...
        else:
            ...existing...
```

Add after `step`:

```python
    def step_free(self, positions, velocities, hidden):
        """Observation-free step: no frame is read, so a token can never
        lose its ball to a faded self-rendered observation -- see
        docs/superpowers/specs/2026-09-23-token-free-rollout-design.md.
        The rasterized grid is output only."""
        if not self.free_rollout:
            raise RuntimeError("step_free requires TokenModel(free_rollout=True)")
        delta_pos, delta_vel, new_hidden = self.dynamics(positions, velocities, hidden)
        final_pos = positions + velocities * self.dt + delta_pos
        final_vel = velocities + delta_vel
        next_grid = rasterize_tokens(final_pos, final_vel, self.n, self.radius)
        return final_pos, final_vel, new_hidden, next_grid
```

- [ ] **Step 4: Run full token tests**

Run: `PYTHONPATH=. python3 -m pytest tests -q`
Expected: all pass (149 existing + new)

- [ ] **Step 5: Commit**

```bash
git add model/token_model.py tests/test_token_model.py
git commit -m "feat: add TokenModel free_rollout mode (observation-free step_free)"
```

---

### Task 3: Free-rollout training

**Files:**
- Modify: `model/token_train.py` (`token_rollout_loss`, `train`, `main`)
- Test: `tests/test_token_train.py` (append)

**Interfaces:**
- Consumes: `TokenModel.step_free`, `init_tokens`, `match_tokens_to_state`, `token_state_loss`, `boundary_loss`, `token_grid_loss`.
- Produces: `token_rollout_loss(..., free=False, max_steps=None)` (when `free`, ignores `sampling_p`, no observed frame; `max_steps` caps the unroll); `horizon_for_epoch(epoch, ramp_epochs, start, full) -> int`; CLI `--free-rollout`, `--horizon-ramp` (default 0 = off), `--horizon-start` (default 4); `<checkpoint>.full` full-state file (model, optimizer, epoch) saved every epoch alongside the existing state_dict file.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_token_train.py`; reuse that file's existing helper for building a small sample if present, otherwise build from `model.token_dataset.generate_dataset_samples`)

```python
def _sample(horizon=6, seed=4738):
    from model.token_dataset import BounceTokenSequenceDataset
    ds = BounceTokenSequenceDataset(num_samples=1, n=20, ball_range=(3, 3), seed=seed, horizon=horizon)
    return ds[0]


def test_horizon_for_epoch():
    from model.token_train import horizon_for_epoch
    assert horizon_for_epoch(0, 10, 4, 24) == 4
    assert horizon_for_epoch(10, 10, 4, 24) == 24
    assert horizon_for_epoch(5, 10, 4, 24) == 14
    assert horizon_for_epoch(3, 0, 4, 24) == 24


def test_free_rollout_loss_finite_and_backprops():
    torch.manual_seed(4738)
    model = TokenModel(n=20, radius=0.75, dt=0.15, free_rollout=True)
    grid_seq, state_seq = _sample()
    weights = torch.tensor([1.0, 0.1, 0.1])
    loss = token_rollout_loss(model, grid_seq, state_seq, 6, 0.0, weights, free=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_free_rollout_loss_max_steps_caps_unroll():
    torch.manual_seed(4738)
    model = TokenModel(n=20, radius=0.75, dt=0.15, free_rollout=True)
    torch.nn.init.normal_(model.dynamics.delta_head.weight, std=0.1)
    grid_seq, state_seq = _sample(horizon=8)
    weights = torch.tensor([1.0, 0.1, 0.1])
    short = token_rollout_loss(model, grid_seq, state_seq, 8, 0.0, weights, free=True, max_steps=2)
    full = token_rollout_loss(model, grid_seq, state_seq, 8, 0.0, weights, free=True)
    assert not torch.isclose(short, full)


def test_free_rollout_loss_zero_tokens_is_zero(monkeypatch):
    model = TokenModel(n=20, radius=0.75, dt=0.15, free_rollout=True)
    grid_seq, state_seq = _sample()
    grid_seq = torch.zeros_like(grid_seq)
    weights = torch.tensor([1.0, 0.1, 0.1])
    loss = token_rollout_loss(model, grid_seq, state_seq, 6, 0.0, weights, free=True)
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. python3 -m pytest tests/test_token_train.py -q -k "free or horizon_for"`
Expected: FAIL (unexpected keyword `free` / missing `horizon_for_epoch`)

- [ ] **Step 3: Implement**

Add above `token_rollout_loss`:

```python
def horizon_for_epoch(epoch, ramp_epochs, start, full):
    if ramp_epochs <= 0:
        return full
    frac = min(1.0, epoch / ramp_epochs)
    return int(round(start + frac * (full - start)))
```

Add `free=False, max_steps=None` to `token_rollout_loss`'s signature. After `num_steps = max(horizon - 1, 1)` add `if max_steps is not None: num_steps = min(num_steps, max_steps)`. In the loop, replace the `model.step` call with:

```python
        if free:
            positions, velocities, hidden, pred_grid = model.step_free(positions, velocities, hidden)
        else:
            positions, velocities, hidden, pred_grid, prev_obs_pos = model.step(
                positions, velocities, hidden, observed_frame, prev_obs_pos
            )
```

and at the loop's tail replace the two self-feed lines with `if not free:` guarding them. `source_grid` in the free branch is `grid_seq[step + 1]` (ground-truth frame the step nominally started from, used only for the occupancy mask in `token_grid_loss`); set `source_grid = grid_seq[step + 1] if free else observed_frame`.

In `train`: pass `free_rollout=args.free_rollout` to `TokenModel`; per epoch compute `epoch_horizon = horizon_for_epoch(epoch, args.horizon_ramp, args.horizon_start, args.horizon)` and pass `free=args.free_rollout, max_steps=epoch_horizon - 1 if args.free_rollout else None`. After the existing `torch.save(model.state_dict(), args.checkpoint)` add:

```python
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch},
                   args.checkpoint + ".full")
```

In `main`: add `--free-rollout` (store_true), `--horizon-ramp` (int, 0), `--horizon-start` (int, 4), each with a one-line help.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=. python3 -m pytest tests -q`
Expected: all pass

- [ ] **Step 5: Smoke train (CPU, tiny)**

Run: `PYTHONPATH=. python3 -m model.token_train --free-rollout --num-samples 8 --horizon 6 --epochs 2 --ramp-epochs 1 --horizon-ramp 2 --horizon-start 3 --checkpoint /tmp/free_smoke.pt --log-every 0`
Expected: two epoch lines, finite loss, `/tmp/free_smoke.pt` and `/tmp/free_smoke.pt.full` written.

- [ ] **Step 6: Commit**

```bash
git add model/token_train.py tests/test_token_train.py
git commit -m "feat: free-rollout training branch with horizon ramp and full-state checkpoints"
```

---

### Task 4: Evaluation script

**Files:**
- Create: `scripts/eval_free_rollout.py`
- Test: `tests/test_eval_free_rollout.py`

**Interfaces:**
- Consumes: `scripts.diagnose_token_dropout.simulate(n, num_balls, seed, num_steps, gravity=9.0)` returning `(frames, states)` with `states[t]` a dict of lists `x, y`; `TokenModel.init_tokens`, `step_free`, `step`; `match_tokens_to_state`.
- Produces: `position_errors(pred_positions (T,N,2), token_to_ball (N,), gt_x (T,B), gt_y (T,B)) -> (T,) mean L2 error per step`; `count_identity_swaps(pred_positions, gt_x, gt_y, token_to_ball) -> int` (tokens whose nearest ground-truth ball at the final step differs from `token_to_ball`); CLI with `--checkpoint`, `--mode {free,v9,v9-noobs,v17}`, `--num-seeds 48`, `--num-steps`, `--n`, `--num-balls`, `--out` (JSON with per-step mean error at requested checkpoints, swaps, token-count mismatches, own-peak<0.05 fraction).

- [ ] **Step 1: Write failing tests** (metric helpers only; the CLI is exercised in Step 5)

```python
import numpy as np
import torch

from scripts.eval_free_rollout import count_identity_swaps, position_errors


def test_position_errors_zero_for_perfect_track():
    T, B = 4, 3
    gt_x = torch.rand(T, B) * 10
    gt_y = torch.rand(T, B) * 10
    pred = torch.stack([gt_x, gt_y], dim=-1)
    err = position_errors(pred, torch.arange(B), gt_x, gt_y)
    assert err.shape == (T,) and torch.allclose(err, torch.zeros(T), atol=1e-6)


def test_position_errors_respects_token_to_ball_permutation():
    gt_x = torch.tensor([[0.0, 5.0]])
    gt_y = torch.tensor([[0.0, 5.0]])
    pred = torch.tensor([[[5.0, 5.0], [0.0, 0.0]]])
    err = position_errors(pred, torch.tensor([1, 0]), gt_x, gt_y)
    assert torch.allclose(err, torch.zeros(1), atol=1e-6)


def test_count_identity_swaps():
    gt_x = torch.tensor([[0.0, 5.0], [0.0, 5.0]])
    gt_y = torch.tensor([[0.0, 5.0], [0.0, 5.0]])
    pred_ok = torch.tensor([[[0.0, 0.0], [5.0, 5.0]], [[0.1, 0.0], [5.0, 5.1]]])
    assert count_identity_swaps(pred_ok, gt_x, gt_y, torch.tensor([0, 1])) == 0
    pred_swapped = torch.tensor([[[0.0, 0.0], [5.0, 5.0]], [[5.0, 5.0], [0.0, 0.0]]])
    assert count_identity_swaps(pred_swapped, gt_x, gt_y, torch.tensor([0, 1])) == 2


def test_helpers_handle_zero_tokens():
    pred = torch.zeros(3, 0, 2)
    err = position_errors(pred, torch.zeros(0, dtype=torch.long), torch.zeros(3, 2), torch.zeros(3, 2))
    assert err.shape == (3,) and torch.isfinite(err).all()
    assert count_identity_swaps(pred, torch.zeros(3, 2), torch.zeros(3, 2), torch.zeros(0, dtype=torch.long)) == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH=. python3 -m pytest tests/test_eval_free_rollout.py -q`
Expected: FAIL, module not found

- [ ] **Step 3: Implement `scripts/eval_free_rollout.py`**

Structure (write it out in full; no placeholders):
- `position_errors`: `gt = stack([gt_x, gt_y], -1)[:, token_to_ball]` shape (T,N,2); return `torch.norm(pred - gt, dim=-1).mean(dim=1)` per step, zeros `(T,)` when N == 0.
- `count_identity_swaps`: final-step `pred[-1]` (N,2) vs final gt (B,2); `nearest = cdist.argmin(1)`; return `int((nearest != token_to_ball).sum())`; 0 when N == 0.
- `load_model(checkpoint, mode, n, hidden_dim, neighbor_radius)` building `TokenModel(free_rollout=mode=="free", track_query=mode=="v17", observation_weight=0.0 if mode=="v9-noobs" else 0.5, ...)`, loading state_dict (accept a `.full` dict by reading `["model"]`), `eval()`.
- `rollout(model, mode, frames, num_steps)`: `init_tokens(g0, g1)`; loop `num_steps-1` times using `step_free` for `free`, else `step(..., observed, prev_obs_pos)` self-fed with the model's own grid (matching `diagnose_token_dropout.py`); returns positions `(T,N,2)` for frame indices 1..num_steps and the final own-peak fractions via `peak_prob_at` imported from `scripts.diagnose_token_dropout`.
- `main`: for seeds `base_seed..base_seed+num_seeds`: `simulate`, `match_tokens_to_state(positions_at_frame1, state1)`, compute errors, swaps, `token_count != ball_count` mismatches; aggregate mean error at steps in `--report-steps` (default `5,10,20`, clipped to `num_steps`), total swaps, mismatch seeds, fraction of tokens with own peak < 0.05; print a table and write `--out` JSON.

- [ ] **Step 4: Run helper tests**

Run: `PYTHONPATH=. python3 -m pytest tests/test_eval_free_rollout.py -q`
Expected: 4 passed

- [ ] **Step 5: Smoke the CLI on v9**

Run: `PYTHONPATH=. python3 scripts/eval_free_rollout.py --checkpoint checkpoints/token_model_h12_v9.pt --mode v9 --num-seeds 4 --num-steps 20 --out /tmp/v9_smoke.json`
Expected: a table with finite errors, JSON written. (If the v9 checkpoint is absent locally, download it from Polaris first: `mcp__polaris__polaris_download`.)

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_free_rollout.py tests/test_eval_free_rollout.py
git commit -m "feat: add eval_free_rollout scoring (position error, identity swaps, OOD)"
```

---

### Task 5: Train v18 on Polaris and score against v9

**Files:**
- Create: `scripts/polaris_generate_token_dataset_v2.sh`, `scripts/polaris_train_token_v18.sh`
- Modify: `docs/debugging/experiment-log.md` (append result)

**Interfaces:**
- Consumes: Tasks 1-4; Polaris MCP (`mcp__polaris__polaris_upload/submit_job/job_status/job_logs/download`).
- Produces: `checkpoints/token_model_h24_v18.pt` (+ `.full`), `results/eval_v9.json`, `results/eval_v18.json`, comparison table in the experiment log.

- [ ] **Step 1: Dataset script.** Copy `polaris_generate_token_dataset_v1.sh`, change to `--horizon 24 --out checkpoints/token_dataset_10000_h24_seed4738.pt`.

- [ ] **Step 2: Train script.** Copy `polaris_train_token_v17.sh`; remove `--track-query`; add `--free-rollout --horizon 24 --horizon-start 4 --horizon-ramp 15`; `--dataset-cache checkpoints/token_dataset_10000_h24_seed4738.pt`; `--epochs 50 --ramp-epochs 1`; `--boundary-weight 0.1 --grid-weight 0.1`; checkpoint `checkpoints/token_model_h24_v18.pt`; job name v18; time limit `04:00:00`; pytest line runs `tests/test_token_*.py`.

- [ ] **Step 3: Commit scripts, upload changed files** (`model/`, `scripts/`, `tests/`, `bounce.py`, requirements), submit the dataset job, wait, submit the train job; monitor logs for finite, falling loss.

- [ ] **Step 4: Download the checkpoint. Evaluate** (48 seeds, base seed 4738):
  - `--mode free` on v18 at 20, 100, 300 steps (`--num-steps 300 --report-steps 20,100,300`).
  - `--mode v9` and `--mode v9-noobs` on the v9 checkpoint, same settings.
  - OOD: `--n 50 --num-balls 100 --num-steps 300` for v18 and v9.
  - Legacy dropout count for v9 via `scripts/diagnose_token_dropout.py` (already logged: 3/48).

- [ ] **Step 5: Render a rollout video and diagnostic grid for v18 and v9** (`scripts/render_token_rollout_video.py`, `scripts/render_token_diagnostic_grid.py`, saved under `videos/`); dispatch an unbiased subagent with `docs/debugging/frame-artifact-review-prompt.md` on the v18 grid; deliver the video with SendUserFile.

- [ ] **Step 6: Log and commit.** Append the v9-vs-v18 table (error at 20/100/300, swaps, count mismatches, OOD) and the subagent's read to `docs/debugging/experiment-log.md`; commit results with the key metric in the message.
