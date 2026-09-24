# Token Track-Query Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `TrackQueryDynamics` architecture to the token-per-ball model that resolves token identity implicitly through learned self- and cross-attention, replacing the hard `occluding_mask`/`centroid_near` gate-and-blend path, and validate whether it beats v9's 3/48 dropout baseline.

**Architecture:** New module `model/token_track_query.py` with a two-stage-attention `TrackQueryDynamics` (self-attention among tokens via the existing radius-graph shape, then cross-attention from each token to its own bounded local window of the observed frame), wired into `TokenModel.step` as a conditional branch selected by `track_query=True`/`--track-query`, with v9's exact code path left completely untouched as the default. New diagnostic script visualizes both attention stages per step. New Polaris training script runs the v9 recipe with `--track-query`.

**Tech Stack:** Python 3.12, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-09-23-token-track-query-design.md`

## Global Constraints

- `track_query=False` (default) must leave every existing code path byte-for-byte unchanged — `TokenDynamics`, `occluding_mask`, `centroid_near`, `observation_weight`/`velocity_weight` blending stay untouched (design doc's "Scope" section).
- No new loss terms — `token_state_loss`, `token_grid_loss`, `boundary_loss` apply identically to both paths; `window_collapse_loss` stays available at its existing default weight (0) and is not wired specially into the track-query path.
- Three project-wide tenets apply to `TrackQueryDynamics` exactly as they do to `TokenDynamics`: arbitrary `n` (no hardcoded horizon), arbitrary `n_ball` (no fixed-size state vector, must generalize across token counts), no baked-in physics (no gravity/collision equations).
- Train-small/tile-large: a token's update must depend only on local geometry (relative offsets), never absolute grid position, in both attention stages.
- `TokenModel.step`'s return signature stays the same 5-tuple (`final_pos, final_vel, new_hidden, next_grid, obs_pos`) in both branches — `token_train.py`'s `prev_obs_pos` bookkeeping needs no changes.
- Full existing suite (124 tests as of this plan) must stay green throughout.

## Review Focus

- **Single-token rollout** (no radius-graph neighbors) — self-attention must still produce a gradient-carrying output via its self-loop, exactly like `TokenDynamics` does today; a test must confirm this, not just assume the reused shape carries the property over.
- **Every expansion's cross-attention window empty** (token has drifted fully off any mass) — the design specifies an all-zero/uniform fallback readout with `obs_pos` unchanged from the input position (no mask-then-unmasked-retry, since there is no mask); a test must confirm this exact fallback, not a silent NaN or exception.
- **Zero tokens** — `TrackQueryDynamics.forward` and `TokenModel.step`'s track-query branch must both return correctly-shaped empty tensors, matching `token_losses.py`/`token_model.py`'s existing empty-token-set convention, not crash on an empty radius graph or an empty cross-attention loop.
- **Absolute position leakage** — neither attention stage may let absolute position reach a linear layer; cross-attention's per-cell feature is `[PROB, VX, VY, dx, dy]` (relative to the token's own position) and self-attention's edge feature is a relative offset only, mirroring `TokenDynamics`'s existing translation-invariance test but extended to cover the grid-reading stage too (shifting both positions and the observed frame together must not change the output).
- **Reordering the input token set** — since nothing in the module may key off a token's index (arbitrary-`n_ball` tenet), permuting the input tensors must permute the output identically, with no cross-talk between a token and its new neighbors in the permuted ordering beyond what the radius graph and cross-attention already define per-token.

---

## File Structure

- Create: `model/token_track_query.py` — `TrackQueryDynamics` (self-attention stage, cross-attention stage, GRU update).
- Modify: `model/token_model.py` — `track_query` constructor flag; conditional branch in `TokenModel.step`.
- Modify: `model/token_train.py` — `--track-query` CLI flag.
- Modify: `scripts/render_token_diagnostic_grid.py`, `scripts/render_token_rollout_video.py`, `scripts/diagnose_token_dropout.py` — `--track-query` CLI flag threaded to `TokenModel(...)`.
- Create: `scripts/visualize_track_query_attention.py` — per-step, per-token self-/cross-attention weight dump.
- Create: `scripts/polaris_train_token_v17.sh` — v9 recipe (`checkpoints/token_dataset_10000_h12_seed4738.pt`, epochs 3, ramp-epochs 2) plus `--track-query`.
- Test: `tests/test_token_track_query.py` — shape/gradient, permutation, translation-invariance, empty-token-set, single-token, empty-window-fallback tests.
- Test: `tests/test_token_model.py` — integration-level `TokenModel.step(track_query=True)` tests.

## Interfaces (final, for reference across tasks)

```python
# model/token_track_query.py
class TrackQueryDynamics(torch.nn.Module):
    def __init__(self, hidden_dim=32, neighbor_radius=3.0, radius=0.75,
                 margin=1.0, max_expansions=6):
        ...

    def forward(self, positions, velocities, hidden, observed_frame):
        """positions: (N, 2). velocities: (N, 2). hidden: (N, hidden_dim).
        observed_frame: (3, n, n) in bounce.py channel order (PROB, VX, VY),
        same time instant as `positions` (see TokenModel.step's docstring
        for the timing convention this mirrors).
        Returns (delta_pos (N, 2), delta_vel (N, 2), new_hidden (N, hidden_dim),
        obs_pos (N, 2)) -- obs_pos is a diagnostic/bookkeeping byproduct
        (this step's cross-attention weighted centroid, or the unchanged
        input position where every window expansion was empty), never
        used for a hard blend."""
```

```python
# model/token_model.py TokenModel.__init__ (new parameter)
def __init__(self, n, radius, dt, hidden_dim=32, neighbor_radius=3.0,
             detect_threshold=0.1, observation_weight=0.5, detect_margin=1.0,
             max_init_speed=20.0, velocity_weight=0.0, max_expansions=6,
             territory_masking=False, track_query=False):
    """track_query=True selects TrackQueryDynamics in place of
    TokenDynamics and makes TokenModel.step skip the
    occluding_mask/centroid_near/blending block entirely. Default False
    leaves every existing code path byte-for-byte unchanged."""
```

---

### Task 1: `TrackQueryDynamics` self-attention stage

**Files:**
- Create: `model/token_track_query.py`
- Test: `tests/test_token_track_query.py`

**Interfaces:**
- Consumes: `model.token_graph.build_radius_graph(positions, neighbor_radius)` (existing, unchanged).
- Produces: `TrackQueryDynamics._self_attention(positions, velocities, hidden) -> attn_out_self (N, hidden_dim)` for Task 3 to consume.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_token_track_query.py`:

```python
import torch

from model.token_track_query import TrackQueryDynamics


def test_self_attention_shape():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=3.0)
    positions = torch.rand(5, 2) * 20
    velocities = torch.randn(5, 2)
    hidden = torch.zeros(5, 8)
    out = model._self_attention(positions, velocities, hidden)
    assert out.shape == (5, 8)


def test_self_attention_single_token_gets_gradient_via_self_loop():
    # A single token has no radius-graph neighbors -- the self-loop
    # (same convention as TokenDynamics) must still give it a nonzero
    # softmax group so gradient reaches self_query/self_key.
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=3.0)
    positions = torch.tensor([[5.0, 5.0]], requires_grad=True)
    velocities = torch.tensor([[1.0, -0.5]])
    hidden = torch.randn(1, 8)
    out = model._self_attention(positions, velocities, hidden)
    out.sum().backward()
    assert model.self_query.weight.grad is not None
    assert torch.any(model.self_query.weight.grad != 0.0)


def test_self_attention_isolated_token_unaffected_by_distant_tokens():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=3.0)
    hidden = torch.randn(3, 8)
    positions_a = torch.tensor([[5.0, 5.0], [5.5, 5.0], [50.0, 50.0]])
    positions_b = torch.tensor([[5.0, 5.0], [5.5, 5.0], [90.0, 90.0]])
    velocities = torch.zeros(3, 2)

    out_a = model._self_attention(positions_a, velocities, hidden)
    out_b = model._self_attention(positions_b, velocities, hidden)

    assert torch.allclose(out_a[0], out_b[0], atol=1e-6)
    assert torch.allclose(out_a[1], out_b[1], atol=1e-6)


def test_self_attention_translation_invariant():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=3.0)
    positions = torch.tensor([[5.0, 5.0], [6.0, 5.5], [5.2, 7.0]])
    velocities = torch.tensor([[1.0, -0.5], [-0.3, 0.8], [0.0, 2.0]])
    hidden = torch.randn(3, 8)

    base = model._self_attention(positions, velocities, hidden)
    shifted = model._self_attention(positions + 500.0, velocities, hidden)

    assert torch.allclose(base, shifted, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python3 -m pytest tests/test_token_track_query.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'model.token_track_query'`

- [ ] **Step 3: Write the implementation**

Create `model/token_track_query.py`:

```python
import math

import torch
import torch.nn as nn

from model.token_graph import build_radius_graph


class TrackQueryDynamics(nn.Module):
    """Drop-in alternative to model.token_net.TokenDynamics that resolves
    token identity through two stages of learned attention instead of a
    hard occlusion gate + spatial mask -- see
    docs/superpowers/specs/2026-09-23-token-track-query-design.md.

    Stage 1 (self-attention among tokens) reuses TokenDynamics's radius-
    graph GAT shape verbatim, including its self-loop convention (a
    token with exactly one neighbor otherwise forms a softmax group of
    size 1 with zero gradient to query/key).

    Stage 2 (cross-attention to the observed scene, model/token_track_query.py's
    _cross_attention_one) replaces occluding_mask + centroid_near
    entirely in this path: no hard gate, no territory mask, just a
    learned softmax read over the same bounded local window
    centroid_near already searches.

    Both stages depend only on relative offsets, never absolute
    position -- required for train-small/tile-large transfer."""

    def __init__(self, hidden_dim=32, neighbor_radius=3.0, radius=0.75,
                 margin=1.0, max_expansions=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.neighbor_radius = neighbor_radius
        self.radius = radius
        self.margin = margin
        self.max_expansions = max_expansions

        self_node_dim = 2 + hidden_dim  # velocity + hidden -- no absolute position
        self_edge_dim = 2  # relative position offset (dx, dy)
        self.self_query = nn.Linear(self_node_dim, hidden_dim)
        self.self_key = nn.Linear(self_node_dim + self_edge_dim, hidden_dim)
        self.self_value = nn.Linear(self_node_dim + self_edge_dim, hidden_dim)

        cross_cell_dim = 5  # PROB, VX, VY, dx, dy
        self.cross_query = nn.Linear(hidden_dim, hidden_dim)
        self.cross_key = nn.Linear(cross_cell_dim, hidden_dim)
        self.cross_value = nn.Linear(cross_cell_dim, hidden_dim)

        self.gru = nn.GRUCell(2 * hidden_dim, hidden_dim)
        self.delta_head = nn.Linear(hidden_dim, 4)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def _self_attention(self, positions, velocities, hidden):
        n = positions.shape[0]
        node_state = torch.cat([velocities, hidden], dim=-1)
        q = self.self_query(node_state)

        edge_index = build_radius_graph(positions, self.neighbor_radius)
        self_loops = torch.arange(n, device=positions.device)
        self_loops = torch.stack([self_loops, self_loops], dim=0)
        edge_index = torch.cat([edge_index, self_loops], dim=1)
        attn_out = torch.zeros(n, self.hidden_dim, device=positions.device, dtype=positions.dtype)
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            rel_pos = positions[src] - positions[dst]
            edge_input = torch.cat([node_state[src], rel_pos], dim=-1)
            k = self.self_key(edge_input)
            v = self.self_value(edge_input)
            scores = (q[dst] * k).sum(dim=-1) / (self.hidden_dim ** 0.5)
            weights = torch.exp(scores - scores.max())
            denom = torch.zeros(n, device=positions.device, dtype=positions.dtype)
            denom = denom.index_add(0, dst, weights)
            weights = weights / denom[dst].clamp(min=1e-6)
            attn_out = attn_out.index_add(0, dst, weights.unsqueeze(-1) * v)
        return attn_out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python3 -m pytest tests/test_token_track_query.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add model/token_track_query.py tests/test_token_track_query.py
git commit -m "feat: add TrackQueryDynamics self-attention stage"
```

---

### Task 2: `TrackQueryDynamics` cross-attention stage

**Files:**
- Modify: `model/token_track_query.py`
- Test: `tests/test_token_track_query.py`

**Interfaces:**
- Consumes: nothing new from Task 1 besides the class itself (this stage is called independently in its own tests, wired together with self-attention in Task 3).
- Produces: `TrackQueryDynamics._cross_attention_one(prob, vx, vy, position, query_vec) -> (readout (hidden_dim,), obs_pos (2,))` for Task 3 to call once per token.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_token_track_query.py`:

```python
def _grid_with_ball(n, x, y, radius=0.75):
    from model.token_rasterize import rasterize_tokens
    positions = torch.tensor([[x, y]])
    velocities = torch.tensor([[1.5, -0.5]])
    grid = rasterize_tokens(positions, velocities, n, radius)
    return grid[0], grid[1], grid[2]


def test_cross_attention_finds_mass_and_returns_centroid_near_position():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, radius=0.75, margin=1.0, max_expansions=6)
    prob, vx, vy = _grid_with_ball(20, 10.0, 10.0)
    query_vec = torch.randn(8)
    readout, obs_pos = model._cross_attention_one(prob, vx, vy, torch.tensor([10.0, 10.0]), query_vec)
    assert readout.shape == (8,)
    assert torch.allclose(obs_pos, torch.tensor([10.0, 10.0]), atol=0.2)


def test_cross_attention_empty_window_falls_back_to_zero_readout_and_unchanged_position():
    # No mass anywhere near this position, even after widening -- must
    # fall back to an all-zero readout and obs_pos == position, matching
    # centroid_near's own give-up convention (no mask-then-unmasked-retry
    # here, since there is no mask to begin with).
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, radius=0.75, margin=1.0, max_expansions=2)
    n = 20
    prob = torch.zeros(n, n)
    vx = torch.zeros(n, n)
    vy = torch.zeros(n, n)
    position = torch.tensor([10.0, 10.0])
    query_vec = torch.randn(8)
    readout, obs_pos = model._cross_attention_one(prob, vx, vy, position, query_vec)
    assert torch.allclose(readout, torch.zeros(8))
    assert torch.allclose(obs_pos, position)


def test_cross_attention_translation_invariant():
    # Same relative window content, shifted to a different absolute
    # position via torch.roll (cyclic, but the window never reaches the
    # wrap boundary at either position) -- readout must match, obs_pos
    # must shift by the same amount.
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, radius=0.75, margin=1.0, max_expansions=6)
    prob, vx, vy = _grid_with_ball(20, 10.0, 10.0)
    shift = 3
    prob_s = torch.roll(prob, shifts=(shift, shift), dims=(0, 1))
    vx_s = torch.roll(vx, shifts=(shift, shift), dims=(0, 1))
    vy_s = torch.roll(vy, shifts=(shift, shift), dims=(0, 1))
    query_vec = torch.randn(8)

    readout, obs_pos = model._cross_attention_one(prob, vx, vy, torch.tensor([10.0, 10.0]), query_vec)
    readout_s, obs_pos_s = model._cross_attention_one(
        prob_s, vx_s, vy_s, torch.tensor([10.0 + shift, 10.0 + shift]), query_vec
    )
    assert torch.allclose(readout, readout_s, atol=1e-5)
    assert torch.allclose(obs_pos_s - obs_pos, torch.tensor([float(shift), float(shift)]), atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python3 -m pytest tests/test_token_track_query.py -v`
Expected: FAIL with `AttributeError: 'TrackQueryDynamics' object has no attribute '_cross_attention_one'`

- [ ] **Step 3: Write the implementation**

Add to `model/token_track_query.py` (as a method on `TrackQueryDynamics`, after `_self_attention`):

```python
    def _cross_attention_one(self, prob, vx, vy, position, query_vec):
        """Cross-attention for one token's bounded local window --
        replaces occluding_mask + centroid_near entirely in this path.
        Gathers the same window centroid_near searches (base half-width
        ceil(radius + margin), widening by one cell per retry up to
        max_expansions if empty), builds a per-cell feature
        [PROB, VX, VY, dx, dy] (dx/dy relative to this token's own
        position -- no absolute position enters), and reads it with a
        learned softmax attention query built from the token's
        post-self-attention hidden state. Returns (readout, obs_pos);
        obs_pos is the attention-weighted centroid, kept only as a
        diagnostic/bookkeeping byproduct (never a hard blend). If every
        expansion's window is empty, falls back to an all-zero readout
        and obs_pos == position -- there is no mask here, so there is no
        masked-then-unmasked retry to replicate (unlike centroid_near)."""
        n = prob.shape[0]
        cx = int(round(float(position[0].detach())))
        cy = int(round(float(position[1].detach())))
        step = max(1, int(math.ceil(self.radius)))
        base_half = int(math.ceil(self.radius + self.margin))
        for expansion in range(self.max_expansions + 1):
            half = base_half + expansion * step
            i_lo_raw, i_hi_raw = cx - half, cx + half
            j_lo_raw, j_hi_raw = cy - half, cy + half
            if i_hi_raw < 0 or i_lo_raw > n - 1 or j_hi_raw < 0 or j_lo_raw > n - 1:
                continue
            i_lo, i_hi = max(0, i_lo_raw), min(n - 1, i_hi_raw)
            j_lo, j_hi = max(0, j_lo_raw), min(n - 1, j_hi_raw)
            window_prob = prob[i_lo:i_hi + 1, j_lo:j_hi + 1]
            if float(window_prob.sum()) <= 1e-6:
                continue
            window_vx = vx[i_lo:i_hi + 1, j_lo:j_hi + 1]
            window_vy = vy[i_lo:i_hi + 1, j_lo:j_hi + 1]
            ii = torch.arange(i_lo, i_hi + 1, device=prob.device, dtype=prob.dtype).view(-1, 1)
            jj = torch.arange(j_lo, j_hi + 1, device=prob.device, dtype=prob.dtype).view(1, -1)
            dx = (ii - position[0]).expand_as(window_prob)
            dy = (jj - position[1]).expand_as(window_prob)
            cell_feat = torch.stack([window_prob, window_vx, window_vy, dx, dy], dim=-1)
            cell_feat = cell_feat.reshape(-1, 5)
            k = self.cross_key(cell_feat)
            v = self.cross_value(cell_feat)
            scores = (query_vec.unsqueeze(0) * k).sum(dim=-1) / (self.hidden_dim ** 0.5)
            weights = torch.softmax(scores, dim=0)
            readout = (weights.unsqueeze(-1) * v).sum(dim=0)
            centroid_dx = (weights * dx.reshape(-1)).sum()
            centroid_dy = (weights * dy.reshape(-1)).sum()
            obs_pos = position + torch.stack([centroid_dx, centroid_dy])
            return readout, obs_pos
        return query_vec.new_zeros(self.hidden_dim), position
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python3 -m pytest tests/test_token_track_query.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add model/token_track_query.py tests/test_token_track_query.py
git commit -m "feat: add TrackQueryDynamics cross-attention stage"
```

---

### Task 3: Full `forward()` wiring + module-level tests

**Files:**
- Modify: `model/token_track_query.py`
- Test: `tests/test_token_track_query.py`

**Interfaces:**
- Consumes: `_self_attention` (Task 1), `_cross_attention_one` (Task 2).
- Produces: `TrackQueryDynamics.forward(positions, velocities, hidden, observed_frame) -> (delta_pos, delta_vel, new_hidden, obs_pos)` for Task 4 (`TokenModel.step`) to call.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_token_track_query.py`:

```python
def _random_frame(n, num_balls, seed=4738):
    from model.token_rasterize import rasterize_tokens
    g = torch.Generator().manual_seed(seed)
    positions = torch.rand(num_balls, 2, generator=g) * (n - 4) + 2
    velocities = torch.randn(num_balls, 2, generator=g)
    return rasterize_tokens(positions, velocities, n, 0.75), positions, velocities


def test_forward_shapes():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=3.0)
    observed_frame, positions, velocities = _random_frame(20, 5)
    hidden = torch.zeros(5, 8)
    delta_pos, delta_vel, new_hidden, obs_pos = model(positions, velocities, hidden, observed_frame)
    assert delta_pos.shape == (5, 2)
    assert delta_vel.shape == (5, 2)
    assert new_hidden.shape == (5, 8)
    assert obs_pos.shape == (5, 2)


def test_delta_head_is_zero_at_init():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=3.0)
    observed_frame, positions, velocities = _random_frame(20, 4)
    hidden = torch.randn(4, 8)
    delta_pos, delta_vel, _, _ = model(positions, velocities, hidden, observed_frame)
    assert torch.allclose(delta_pos, torch.zeros_like(delta_pos))
    assert torch.allclose(delta_vel, torch.zeros_like(delta_vel))


def test_forward_gradients_flow_to_both_attention_stages():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=3.0)
    observed_frame, positions, velocities = _random_frame(20, 4)
    positions = positions.clone().requires_grad_(True)
    hidden = torch.randn(4, 8)
    delta_pos, delta_vel, new_hidden, _ = model(positions, velocities, hidden, observed_frame)
    new_hidden.sum().backward()
    assert model.self_query.weight.grad is not None
    assert torch.any(model.self_query.weight.grad != 0.0)
    assert model.cross_query.weight.grad is not None
    assert torch.any(model.cross_query.weight.grad != 0.0)


def test_forward_empty_token_set():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=3.0)
    n = 20
    observed_frame = torch.zeros(3, n, n)
    positions = torch.zeros(0, 2)
    velocities = torch.zeros(0, 2)
    hidden = torch.zeros(0, 8)
    delta_pos, delta_vel, new_hidden, obs_pos = model(positions, velocities, hidden, observed_frame)
    assert delta_pos.shape == (0, 2)
    assert delta_vel.shape == (0, 2)
    assert new_hidden.shape == (0, 8)
    assert obs_pos.shape == (0, 2)
    assert not torch.isnan(delta_pos).any()


def test_forward_permutation_equivariant():
    torch.manual_seed(4738)
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=4.0)
    observed_frame, positions, velocities = _random_frame(20, 4)
    hidden = torch.randn(4, 8)
    perm = torch.tensor([2, 0, 3, 1])

    delta_pos, delta_vel, new_hidden, obs_pos = model(positions, velocities, hidden, observed_frame)
    delta_pos_p, delta_vel_p, new_hidden_p, obs_pos_p = model(
        positions[perm], velocities[perm], hidden[perm], observed_frame
    )

    assert torch.allclose(delta_pos[perm], delta_pos_p, atol=1e-5)
    assert torch.allclose(delta_vel[perm], delta_vel_p, atol=1e-5)
    assert torch.allclose(new_hidden[perm], new_hidden_p, atol=1e-5)
    assert torch.allclose(obs_pos[perm], obs_pos_p, atol=1e-5)


def test_forward_translation_invariant():
    # Shift positions and the observed frame together (cyclic roll,
    # tokens kept well clear of the wrap boundary before and after) --
    # delta_pos/delta_vel/new_hidden must be unchanged (obs_pos is
    # absolute and is expected to shift by the same amount).
    torch.manual_seed(4738)
    from model.token_rasterize import rasterize_tokens
    model = TrackQueryDynamics(hidden_dim=8, neighbor_radius=4.0)
    n = 20
    g = torch.Generator().manual_seed(4738)
    positions = torch.rand(3, 2, generator=g) * 10 + 5.0  # kept in [5, 15] before AND after clamping below
    velocities = torch.randn(3, 2, generator=g)
    positions = torch.clamp(positions, 5.0, n - 5.0)
    observed_frame = rasterize_tokens(positions, velocities, n, 0.75)
    hidden = torch.randn(3, 8)
    shift = 2

    delta_pos, delta_vel, new_hidden, obs_pos = model(positions, velocities, hidden, observed_frame)

    shifted_frame = torch.roll(observed_frame, shifts=(shift, shift), dims=(1, 2))
    shifted_positions = positions + shift
    delta_pos_s, delta_vel_s, new_hidden_s, obs_pos_s = model(
        shifted_positions, velocities, hidden, shifted_frame
    )

    assert torch.allclose(delta_pos, delta_pos_s, atol=1e-4)
    assert torch.allclose(delta_vel, delta_vel_s, atol=1e-4)
    assert torch.allclose(new_hidden, new_hidden_s, atol=1e-4)
    assert torch.allclose(obs_pos_s - obs_pos, torch.full_like(obs_pos, float(shift)), atol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python3 -m pytest tests/test_token_track_query.py -v`
Expected: FAIL with `TypeError: 'TrackQueryDynamics' object is not callable` (no `forward` yet)

- [ ] **Step 3: Write the implementation**

Add to `model/token_track_query.py` (as a method on `TrackQueryDynamics`, after `_cross_attention_one`):

```python
    def forward(self, positions, velocities, hidden, observed_frame):
        n = positions.shape[0]
        if n == 0:
            zeros2 = positions.new_zeros((0, 2))
            zeros_h = hidden.new_zeros((0, self.hidden_dim))
            return zeros2, zeros2, zeros_h, zeros2

        attn_out_self = self._self_attention(positions, velocities, hidden)
        query_vecs = self.cross_query(attn_out_self)

        prob, vx, vy = observed_frame[0], observed_frame[1], observed_frame[2]
        readouts = []
        obs_positions = []
        for i in range(n):
            readout, obs_pos = self._cross_attention_one(prob, vx, vy, positions[i], query_vecs[i])
            readouts.append(readout)
            obs_positions.append(obs_pos)
        cross_readout = torch.stack(readouts, dim=0)
        obs_pos = torch.stack(obs_positions, dim=0)

        gru_input = torch.cat([attn_out_self, cross_readout], dim=-1)
        new_hidden = self.gru(gru_input, hidden)
        delta = self.delta_head(new_hidden)
        return delta[:, :2], delta[:, 2:], new_hidden, obs_pos
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python3 -m pytest tests/test_token_track_query.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
git add model/token_track_query.py tests/test_token_track_query.py
git commit -m "feat: wire TrackQueryDynamics.forward (self + cross attention + GRU update)"
```

---

### Task 4: `TokenModel` integration

**Files:**
- Modify: `model/token_model.py`
- Test: `tests/test_token_model.py`

**Interfaces:**
- Consumes: `TrackQueryDynamics(hidden_dim, neighbor_radius, radius, margin, max_expansions)` and `.forward(positions, velocities, hidden, observed_frame) -> (delta_pos, delta_vel, new_hidden, obs_pos)` (Task 3).
- Produces: `TokenModel(..., track_query=True)` / `TokenModel.track_query` for Task 5 (CLI wiring) to set from a flag.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_token_model.py`:

```python
def test_track_query_step_returns_same_five_tuple_shape():
    torch.manual_seed(4738)
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, hidden_dim=8, track_query=True)
    positions = torch.tensor([[10.0, 10.0], [12.0, 8.0]])
    velocities = torch.tensor([[1.0, -0.5], [0.0, 1.0]])
    hidden = torch.zeros(2, model.dynamics.hidden_dim)
    observed_frame = rasterize_tokens(positions, velocities, n, radius)

    new_pos, new_vel, new_hidden, pred_grid, obs_pos = model.step(positions, velocities, hidden, observed_frame)

    assert new_pos.shape == (2, 2)
    assert new_vel.shape == (2, 2)
    assert new_hidden.shape == (2, model.dynamics.hidden_dim)
    assert pred_grid.shape == (3, n, n)
    assert obs_pos.shape == (2, 2)
    assert not torch.isnan(new_pos).any()


def test_track_query_step_coasts_at_constant_velocity_when_delta_head_is_zero_init():
    torch.manual_seed(4738)
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, hidden_dim=8, track_query=True)
    positions = torch.tensor([[10.0, 10.0]])
    velocities = torch.tensor([[2.0, -1.0]])
    hidden = torch.zeros(1, model.dynamics.hidden_dim)
    observed_frame = rasterize_tokens(positions, velocities, n, radius)

    new_pos, new_vel, _, _, _ = model.step(positions, velocities, hidden, observed_frame)

    assert torch.allclose(new_pos, positions + velocities * dt, atol=1e-5)
    assert torch.allclose(new_vel, velocities, atol=1e-5)


def test_track_query_model_uses_track_query_dynamics():
    from model.token_track_query import TrackQueryDynamics
    model = TokenModel(n=20, radius=0.75, dt=0.15, track_query=True)
    assert isinstance(model.dynamics, TrackQueryDynamics)


def test_default_model_still_uses_token_dynamics():
    from model.token_net import TokenDynamics
    model = TokenModel(n=20, radius=0.75, dt=0.15)
    assert isinstance(model.dynamics, TokenDynamics)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python3 -m pytest tests/test_token_model.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'track_query'`

- [ ] **Step 3: Write the implementation**

In `model/token_model.py`, add the import:

```python
from model.token_track_query import TrackQueryDynamics
```

Change `TokenModel.__init__` signature and the dynamics construction (`model/token_model.py:54-111`):

```python
    def __init__(self, n, radius, dt, hidden_dim=32, neighbor_radius=3.0,
                 detect_threshold=0.1, observation_weight=0.5, detect_margin=1.0,
                 max_init_speed=20.0, velocity_weight=0.0, max_expansions=6,
                 territory_masking=False, track_query=False):
        super().__init__()
        self.n = n
        self.radius = radius
        self.dt = dt
        self.detect_threshold = detect_threshold
        self.observation_weight = observation_weight
        self.max_expansions = max_expansions
        self.territory_masking = territory_masking
        self.velocity_weight = velocity_weight
        self.detect_margin = detect_margin
        self.max_init_speed = max_init_speed
        # Opt-in architecture swap (see
        # docs/superpowers/specs/2026-09-23-token-track-query-design.md):
        # replaces occluding_mask/centroid_near/blending in `step` with
        # learned self- and cross-attention. Defaults to False so v9's
        # exact code path stays byte-for-byte unchanged and reachable as
        # the reference baseline.
        self.track_query = track_query
        if track_query:
            self.dynamics = TrackQueryDynamics(
                hidden_dim=hidden_dim, neighbor_radius=neighbor_radius,
                radius=radius, margin=detect_margin, max_expansions=max_expansions,
            )
        else:
            self.dynamics = TokenDynamics(hidden_dim=hidden_dim, neighbor_radius=neighbor_radius)
```

(Keep every existing docstring comment on the unchanged fields — this only inserts the new `track_query` field/branch after `max_init_speed` and before the existing `self.dynamics = TokenDynamics(...)` line, replacing that one line with the `if`/`else` above.)

Change `TokenModel.step` (`model/token_model.py:169-226`) to branch at the top:

```python
    def step(self, positions, velocities, hidden, observed_frame, prev_obs_pos=None):
        """...""" # (keep the existing docstring unchanged)
        if self.track_query:
            delta_pos, delta_vel, new_hidden, obs_pos = self.dynamics(
                positions, velocities, hidden, observed_frame
            )
            final_pos = positions + velocities * self.dt + delta_pos
            final_vel = velocities + delta_vel
            next_grid = rasterize_tokens(final_pos, final_vel, self.n, self.radius)
            return final_pos, final_vel, new_hidden, next_grid, obs_pos

        # Existing v9 path -- byte-for-byte unchanged below.
        occluding = occluding_mask(positions, self._gate_radius())
        corrected_pos = positions.clone()
        corrected_vel = velocities.clone()
        obs_pos = positions.clone()
        w = self.observation_weight
        vw = self.velocity_weight
        for i in range(positions.shape[0]):
            if occluding[i] or w == 0.0:
                continue
            op = centroid_near(observed_frame[0], positions[i], self.radius,
                                margin=self.detect_margin, max_expansions=self.max_expansions,
                                all_positions=positions if self.territory_masking else None,
                                self_idx=i if self.territory_masking else None)
            obs_pos[i] = op
            corrected_pos[i] = (1 - w) * positions[i] + w * op
            if prev_obs_pos is not None and vw > 0.0:
                obs_vel = (op - prev_obs_pos[i]) / self.dt
                corrected_vel[i] = (1 - vw) * velocities[i] + vw * obs_vel

        delta_pos, delta_vel, new_hidden = self.dynamics(corrected_pos, corrected_vel, hidden)
        final_pos = corrected_pos + corrected_vel * self.dt + delta_pos
        final_vel = corrected_vel + delta_vel

        next_grid = rasterize_tokens(final_pos, final_vel, self.n, self.radius)
        return final_pos, final_vel, new_hidden, next_grid, obs_pos
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python3 -m pytest tests/test_token_model.py tests/test_token_track_query.py -v`
Expected: PASS (all tests, including the 4 new ones)

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `PYTHONPATH=. python3 -m pytest tests/ -q`
Expected: PASS, 141 tests (124 existing + 13 from `test_token_track_query.py` + 4 new in `test_token_model.py`)

- [ ] **Step 6: Commit**

```bash
git add model/token_model.py tests/test_token_model.py
git commit -m "feat: wire track_query into TokenModel.step"
```

---

### Task 5: CLI flag wiring (`token_train.py` + render/eval/diagnostic scripts)

**Files:**
- Modify: `model/token_train.py`
- Modify: `scripts/render_token_diagnostic_grid.py`
- Modify: `scripts/render_token_rollout_video.py`
- Modify: `scripts/diagnose_token_dropout.py`

**Interfaces:**
- Consumes: `TokenModel(..., track_query=bool)` (Task 4).
- Produces: `--track-query` flag available on all four entry points, matching the existing `--territory-masking` flag's wiring pattern.

- [ ] **Step 1: Wire `model/token_train.py`**

In `train()` (`model/token_train.py:108-111`), add `track_query=args.track_query` to the `TokenModel(...)` call:

```python
    model = TokenModel(n=args.n, radius=0.75, dt=0.15, hidden_dim=args.hidden_dim,
                        neighbor_radius=args.neighbor_radius,
                        velocity_weight=args.velocity_weight,
                        territory_masking=args.territory_masking,
                        track_query=args.track_query).to(device)
```

In `main()`, add the flag next to `--territory-masking` (`model/token_train.py:170-176`):

```python
    ap.add_argument("--territory-masking", action="store_true")
    # Opt-in architecture swap (see
    # docs/superpowers/specs/2026-09-23-token-track-query-design.md):
    # replaces TokenModel's hard occlusion gate + centroid_near blend
    # with learned self-/cross-attention. Off by default so existing
    # recipes/checkpoints are byte-for-byte unaffected.
    ap.add_argument("--track-query", action="store_true")
```

- [ ] **Step 2: Wire `scripts/render_token_diagnostic_grid.py`**

Add `--track-query` next to its existing `--territory-masking` (around line 66), and `track_query=args.track_query` to its `TokenModel(...)` call (around line 77), following the exact same pattern as `--territory-masking` in that file.

- [ ] **Step 3: Wire `scripts/render_token_rollout_video.py`**

This file's `TokenModel(...)` construction is inside a helper function, not `main()` directly (`territory_masking=False` is a keyword parameter of that helper, threaded from `args.territory_masking` in `main()`). Add `track_query=False` as a parameter on that same helper, add `--track-query` next to `--territory-masking` in `main()` (around line 75), and pass `track_query=args.track_query` at the helper call site, mirroring exactly how `territory_masking` already flows through this file.

- [ ] **Step 4: Wire `scripts/diagnose_token_dropout.py`**

In `load_model()` (`scripts/diagnose_token_dropout.py:28-34`), add `track_query=False` as a parameter and pass it through to `TokenModel(...)`:

```python
def load_model(checkpoint_path, n, hidden_dim, neighbor_radius, velocity_weight=0.0,
                territory_masking=False, track_query=False):
    model = TokenModel(n=n, radius=0.75, dt=0.15, hidden_dim=hidden_dim, neighbor_radius=neighbor_radius,
                        velocity_weight=velocity_weight, territory_masking=territory_masking,
                        track_query=track_query)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model
```

Add `--track-query` next to `--territory-masking` in `main()` (`scripts/diagnose_token_dropout.py:134-136`):

```python
    ap.add_argument("--track-query", action="store_true",
                     help="evaluate the checkpoint with track-query attention -- must match "
                          "how it was trained (--track-query in token_train.py)")
```

Pass it at the `load_model(...)` call site (`scripts/diagnose_token_dropout.py:143-144`):

```python
    model = load_model(args.checkpoint, args.n, args.hidden_dim, args.neighbor_radius, args.velocity_weight,
                        territory_masking=args.territory_masking, track_query=args.track_query)
```

Leave the `window_total_at`/`occluding_mask` trace instrumentation in `main()`'s per-step loop (lines 182-205) unchanged: it remains a valid diagnostic read of the observed-frame window mass regardless of which architecture produced `positions`, even though a track-query model's own `step()` never calls `centroid_near` or `occluding_mask` internally.

- [ ] **Step 5: Verify no syntax errors and existing behavior is unchanged**

Run: `PYTHONPATH=. python3 -m pytest tests/ -q`
Expected: PASS, same count as Task 4's Step 5 (these are argparse/call-site-only changes with no new tests — `TokenModel`'s own behavior is already covered by Task 4's tests)

Run: `PYTHONPATH=. python3 -m model.token_train --help` and confirm `--track-query` appears in the output.

- [ ] **Step 6: Commit**

```bash
git add model/token_train.py scripts/render_token_diagnostic_grid.py scripts/render_token_rollout_video.py scripts/diagnose_token_dropout.py
git commit -m "feat: add --track-query CLI flag to token train/render/diagnostic scripts"
```

---

### Task 6: Attention-weight visualization diagnostic

**Files:**
- Create: `scripts/visualize_track_query_attention.py`

**Interfaces:**
- Consumes: `TokenModel(..., track_query=True)`, `model.token_graph.build_radius_graph`, `TrackQueryDynamics._self_attention`/`_cross_attention_one` internals (re-derives the same attention weights `forward` computes internally, for inspection — does not modify `TrackQueryDynamics` to expose them, since adding weight-return plumbing to the hot forward path is unnecessary for a diagnostic-only script that just needs to look at a checkpoint's learned parameters against inputs it can freely inspect).
- Produces: a standalone script, not consumed by any other task.

- [ ] **Step 1: Write the script**

Create `scripts/visualize_track_query_attention.py`:

```python
"""Dumps per-step, per-token self-/cross-attention weights for a
track-query TokenModel rollout, aligned against diagnose_token_dropout.py's
per-step trace output (window total, occlusion state would be reported
by that script; occlusion/window-total are gate-and-mask concepts that
don't apply to this architecture, so this script reports the two things
that DO apply here: which neighbors a token's self-attention weighted
most heavily, and how concentrated its cross-attention was over its own
local window). Built alongside the architecture (see
docs/superpowers/specs/2026-09-23-token-track-query-design.md's
Diagnostics section) so a still-dropping-out seed can be inspected for
WHAT changed in kind (e.g. attention diffusing across two tokens during
a close approach), not just whether the aggregate dropout count moved.

Usage:
    PYTHONPATH=. python3 scripts/visualize_track_query_attention.py \
        --checkpoint checkpoints/token_model_h12_v17.pt \
        --seed 4738 --num-steps 20
"""
import argparse
import random

import numpy as np
import torch

import bounce
from model.dataset import make_scenario_uniform
from model.token_model import TokenModel
from model.token_graph import build_radius_graph


def simulate(n, num_balls, seed, num_steps, dt=0.15, gravity=9.0, radius=0.75,
             stiffness=400.0, substeps=8, vy=2.3):
    rng = random.Random(seed)
    balls = make_scenario_uniform(num_balls, n, vy, rng)
    G = bounce.make_grid(n)
    bounce.splat_all(G, n, balls, radius)
    frames = [np.array(G, dtype=np.float32)]
    for _ in range(num_steps):
        bounce.step(G, n, balls, dt, gravity, radius, stiffness, substeps)
        frames.append(np.array(G, dtype=np.float32))
    return frames


def self_attention_weights(dynamics, positions, velocities, hidden):
    """Recomputes _self_attention's per-edge softmax weights (not just
    its pooled output) for inspection -- mirrors
    TrackQueryDynamics._self_attention exactly, since that method itself
    only returns the pooled sum, not the weights."""
    n = positions.shape[0]
    node_state = torch.cat([velocities, hidden], dim=-1)
    q = dynamics.self_query(node_state)
    edge_index = build_radius_graph(positions, dynamics.neighbor_radius)
    self_loops = torch.arange(n)
    self_loops = torch.stack([self_loops, self_loops], dim=0)
    edge_index = torch.cat([edge_index, self_loops], dim=1)
    if edge_index.shape[1] == 0:
        return {}
    src, dst = edge_index[0], edge_index[1]
    rel_pos = positions[src] - positions[dst]
    edge_input = torch.cat([node_state[src], rel_pos], dim=-1)
    k = dynamics.self_key(edge_input)
    scores = (q[dst] * k).sum(dim=-1) / (dynamics.hidden_dim ** 0.5)
    weights = torch.exp(scores - scores.max())
    denom = torch.zeros(n)
    denom = denom.index_add(0, dst, weights)
    weights = weights / denom[dst].clamp(min=1e-6)
    by_dst = {}
    for e in range(edge_index.shape[1]):
        d, s, w = int(dst[e]), int(src[e]), float(weights[e])
        by_dst.setdefault(d, []).append((s, w))
    return by_dst


def cross_attention_concentration(dynamics, positions, hidden, observed_frame):
    """Runs _cross_attention_one per token and reports each one's max
    softmax weight (close to 1/window_size = diffuse/uncertain, close to
    1.0 = confidently locked onto one cell) -- the cross-attention analog
    of asking whether a token's read was decisive or smeared."""
    query_vecs = dynamics.cross_query(dynamics._self_attention(
        positions, torch.zeros_like(positions), hidden
    ))
    prob, vx, vy = observed_frame[0], observed_frame[1], observed_frame[2]
    results = []
    for i in range(positions.shape[0]):
        n = prob.shape[0]
        cx = int(round(float(positions[i, 0])))
        cy = int(round(float(positions[i, 1])))
        half = int(np.ceil(dynamics.radius + dynamics.margin))
        i_lo, i_hi = max(0, cx - half), min(n - 1, cx + half)
        j_lo, j_hi = max(0, cy - half), min(n - 1, cy + half)
        window_prob = prob[i_lo:i_hi + 1, j_lo:j_hi + 1]
        window_vx = vx[i_lo:i_hi + 1, j_lo:j_hi + 1]
        window_vy = vy[i_lo:i_hi + 1, j_lo:j_hi + 1]
        ii = torch.arange(i_lo, i_hi + 1, dtype=prob.dtype).view(-1, 1)
        jj = torch.arange(j_lo, j_hi + 1, dtype=prob.dtype).view(1, -1)
        dx = (ii - positions[i, 0]).expand_as(window_prob)
        dy = (jj - positions[i, 1]).expand_as(window_prob)
        cell_feat = torch.stack([window_prob, window_vx, window_vy, dx, dy], dim=-1).reshape(-1, 5)
        k = dynamics.cross_key(cell_feat)
        scores = (query_vecs[i].unsqueeze(0) * k).sum(dim=-1) / (dynamics.hidden_dim ** 0.5)
        weights = torch.softmax(scores, dim=0)
        results.append(float(weights.max()))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--num-balls", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=20)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--neighbor-radius", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--gravity", type=float, default=9.0)
    args = ap.parse_args()

    model = TokenModel(n=args.n, radius=0.75, dt=0.15, hidden_dim=args.hidden_dim,
                        neighbor_radius=args.neighbor_radius, track_query=True)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    frames = simulate(args.n, args.num_balls, args.seed, args.num_steps, gravity=args.gravity)
    g0 = torch.from_numpy(frames[0].transpose(2, 0, 1))
    g1 = torch.from_numpy(frames[1].transpose(2, 0, 1))

    with torch.no_grad():
        positions, velocities, hidden = model.init_tokens(g0, g1)
        observed = g1
        for step in range(args.num_steps - 1):
            self_weights = self_attention_weights(model.dynamics, positions, velocities, hidden)
            cross_conc = cross_attention_concentration(model.dynamics, positions, hidden, observed)
            print(f"--- step {step} ---")
            for t in range(positions.shape[0]):
                neighbors = sorted(self_weights.get(t, []), key=lambda p: -p[1])
                neighbor_str = ", ".join(f"tok{s}:{w:.2f}" for s, w in neighbors if s != t)
                print(f"  token {t}: self_attn->[{neighbor_str}] cross_attn_max={cross_conc[t]:.3f}")
            positions, velocities, hidden, pred_grid, _ = model.step(positions, velocities, hidden, observed)
            observed = pred_grid


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the script against an untrained model**

Run:
```bash
PYTHONPATH=. python3 -c "
import torch
from model.token_model import TokenModel
torch.manual_seed(4738)
model = TokenModel(n=20, radius=0.75, dt=0.15, track_query=True)
torch.save(model.state_dict(), '/tmp/smoke_track_query.pt')
"
PYTHONPATH=. python3 scripts/visualize_track_query_attention.py --checkpoint /tmp/smoke_track_query.pt --num-steps 3
```
Expected: prints `--- step 0 ---`, `--- step 1 ---` blocks with per-token `self_attn->[...]` and `cross_attn_max=...` lines, no exception.

Never use inline Python via `python3 -c` for anything beyond this one-line smoke-checkpoint save — write any further ad hoc scripts to a file first, per project convention.

- [ ] **Step 3: Commit**

```bash
git add scripts/visualize_track_query_attention.py
git commit -m "feat: add track-query self-/cross-attention visualization script"
```

---

### Task 7: Polaris training script (v17)

**Files:**
- Create: `scripts/polaris_train_token_v17.sh`

**Interfaces:**
- Consumes: `--track-query` (Task 5), the existing cached dataset `checkpoints/token_dataset_10000_h12_seed4738.pt` (same one v15/v16 used).
- Produces: `checkpoints/token_model_h12_v17.pt` after a Polaris run, the input to the validation plan's dropout/visual pipeline (steps 3-5 of the design doc's Validation plan, done manually after this task, not automated by this plan).

- [ ] **Step 1: Write the script**

Create `scripts/polaris_train_token_v17.sh`, following `scripts/polaris_train_token_v15.sh`'s exact recipe (same dataset cache, epochs, hyperparameters) with `--territory-masking` replaced by `--track-query`:

```bash
#!/bin/bash
#SBATCH --job-name=bounce-token-model-v17
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=bounce-token-model-v17-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"

echo "[$(date -Iseconds)] starting pip install"
pip install -r requirements.txt
echo "[$(date -Iseconds)] pip install done"

mkdir -p checkpoints

echo "[$(date -Iseconds)] running tests before real training run"
python -m pytest tests/test_token_*.py -q
echo "[$(date -Iseconds)] tests passed"

# v17: identical recipe to v9/v15 (state-loss-only ablation) plus
# --track-query -- the track-query attention architecture
# (docs/superpowers/specs/2026-09-23-token-track-query-design.md),
# replacing v15's territory-masking approach entirely (a structurally
# different fix for the same identity-dropout problem). Primary success
# criterion: dropout count (scripts/diagnose_token_dropout.py, 48 seeds)
# below v9's unmasked baseline of 3/48.
echo "[$(date -Iseconds)] starting training"
python -m model.token_train \
  --dataset-cache checkpoints/token_dataset_10000_h12_seed4738.pt \
  --n 20 --horizon 12 --epochs 3 \
  --hidden-dim 32 --neighbor-radius 4.0 \
  --bg-weight 0.05 --peak-weight 0.5 --boundary-weight 0.0 \
  --state-weight 1.0 --grid-weight 0.0 \
  --track-query \
  --lr 1e-3 --ramp-epochs 2 \
  --seed 4738 --log-every 200 \
  --checkpoint checkpoints/token_model_h12_v17.pt
echo "[$(date -Iseconds)] training done"
```

- [ ] **Step 2: Confirm the script is syntactically valid**

Run: `bash -n scripts/polaris_train_token_v17.sh`
Expected: no output (syntax OK)

- [ ] **Step 3: Commit**

```bash
git add scripts/polaris_train_token_v17.sh
git commit -m "feat: add v17 Polaris training script (track-query attention)"
```

---

## After this plan

Submit `scripts/polaris_train_token_v17.sh` to Polaris, then run the design doc's Validation plan steps 3-5: `scripts/diagnose_token_dropout.py` (48 seeds, `--track-query`) against v9's 3/48 baseline, diagnostic grid + rollout video + unbiased fresh-subagent review, and `scripts/visualize_track_query_attention.py` on any seed that still drops out. Log the result in `docs/debugging/experiment-log.md` following this project's existing entry format. This is deliberately out of scope for this plan (a training run and its analysis are not a code task with a test cycle).
