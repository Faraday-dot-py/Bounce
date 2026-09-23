# Token Territory Masking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the token-per-ball model's tokens from stealing a neighboring token's rasterized mass during self-fed rollout, by giving each token an exclusive "territory" (Voronoi partition over currently-tracked positions) that its observation window can read from.

**Architecture:** Add a pure `territory_mask` helper in `model/token_detect.py`, thread optional `all_positions`/`self_idx` parameters through `centroid_near` and `window_collapse_loss` so their window sums exclude any cell closer to a different token, wire `TokenModel.step` to pass its own `positions` tensor through, then widen `centroid_near`'s give-up search now that widening can no longer cross into a neighbor's territory. Retrain with the unchanged v9 recipe (single-variable A/B) and run the standard dropout/visual verification pipeline.

**Tech Stack:** Python 3.12, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-09-23-token-territory-masking-design.md`

## Global Constraints

- Default behavior (no `all_positions`/`self_idx` passed) must be byte-for-byte identical to current `centroid_near`/`window_collapse_loss` output — existing call sites (`find_token_positions`, any test that doesn't pass the new args) must not change.
- No new physical constants — territory is a plain nearest-tracked-position partition, not derived from `radius`.
- Do not change dataset size or epoch count in the retrain step (Task 6) — single-variable A/B against v9, per `docs/debugging/experiment-log.md`'s systematic-debugging discipline.
- Full existing suite (106 tests as of this plan) must stay green throughout.

## Review Focus

- A single-token rollout (no other tracked tokens) must get unmasked behavior — `territory_mask` must not accidentally zero out a token's own valid window when there's nothing to partition against.
- Two tokens at the exact same position (already-collapsed degenerate case) must not make `territory_mask` return all-False for both — the tie-break rule needs a concrete test, not just a docstring claim.
- `window_collapse_loss`'s territory-masked window sum must still produce a *lower* penalty than before for a token near a healthy neighbor (previously it could "borrow" the neighbor's mass to look fine; after masking it can't) — a test should confirm the penalty rises, not silently stays the same due to a wiring bug.
- The give-up bailout (`centroid_near` returning `position` unchanged) must still trigger when a token's territory-restricted window is genuinely empty everywhere, even after `max_expansions` is widened — an infinite/runaway expansion loop is a real risk when raising this constant.
- `TokenModel.step`'s occluding-token branch must remain fully bypassed by this change — occluding tokens already skip the observation branch before territory masking would ever apply; a test should confirm occluding-token behavior is unchanged, not just that non-occluding behavior is new.

---

## File Structure

- Modify: `model/token_detect.py` — add `territory_mask`; add `all_positions`/`self_idx` params to `centroid_near`.
- Modify: `model/token_losses.py` — add `all_positions`/`self_idx` params to `window_collapse_loss`.
- Modify: `model/token_model.py` — pass `all_positions=positions, self_idx=i` at both `centroid_near` call sites in `TokenModel.step`; raise `TokenDynamics`-unrelated `max_expansions` default via a new `TokenModel.__init__` parameter threaded to `centroid_near`.
- Modify: `model/token_train.py` — pass `all_positions`/`self_idx` through to `window_collapse_loss` at its call site (only exercised when `--collapse-weight > 0`, off by default).
- Test: `tests/test_token_detect.py` — territory mask + masked `centroid_near` tests.
- Test: `tests/test_token_losses.py` — masked `window_collapse_loss` tests.
- Test: `tests/test_token_model.py` — integration-level two-token `TokenModel.step` test.
- Create: `scripts/polaris_train_token_v14.sh` — v9 recipe, unchanged hyperparameters, new checkpoint name.

## Interfaces (final, for reference across tasks)

```python
# model/token_detect.py
def territory_mask(ii, jj, positions, self_idx):
    """ii, jj: (h, w) float tensors of cell coordinates (as centroid_near
    already builds via torch.arange(...).view(-1,1) / .view(1,-1), then
    broadcast to a common (h, w) shape by the caller before this is called).
    positions: (k, 2) tensor of all currently-tracked token positions.
    self_idx: int, row of `positions` this window belongs to.
    Returns: (h, w) bool tensor, True where the cell is at least as close
    to positions[self_idx] as to every other row (ties favor self_idx)."""

def centroid_near(prob, position, radius, margin=1.0, max_expansions=3,
                   all_positions=None, self_idx=None):
    """Unchanged return type/semantics. New behavior only when both
    all_positions and self_idx are not None: window mass outside this
    token's territory is excluded before computing total/centroid."""

# model/token_losses.py
def window_collapse_loss(prob, positions, radius, margin=1.0, floor=0.3,
                          all_positions=None, self_idx_offset=0):
    """Unchanged signature/semantics when all_positions is None (default).
    When all_positions is given, each token i's window sum is masked to
    territory_mask(..., all_positions, self_idx_offset + i) before
    comparing against floor. In practice callers pass all_positions=positions
    (the same tensor as `positions`) and leave self_idx_offset=0."""
```

---

### Task 1: `territory_mask` helper

**Files:**
- Modify: `model/token_detect.py`
- Test: `tests/test_token_detect.py`

**Interfaces:**
- Consumes: nothing new (pure tensor function).
- Produces: `territory_mask(ii, jj, positions, self_idx)` for Task 2 to call from inside `centroid_near`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_token_detect.py`:

```python
from model.token_detect import territory_mask


def test_territory_mask_splits_window_at_midline():
    # Two tokens 4 cells apart on the x-axis; a window spanning both must
    # be split down the middle, each token owning its own half.
    ii = torch.arange(0, 10, dtype=torch.float32).view(-1, 1).expand(10, 1)
    jj = torch.zeros(10, 1)
    positions = torch.tensor([[3.0, 0.0], [7.0, 0.0]])
    mask0 = territory_mask(ii, jj, positions, self_idx=0)
    mask1 = territory_mask(ii, jj, positions, self_idx=1)
    # cell x=4 is closer to token 0 (dist 1) than token 1 (dist 3)
    assert bool(mask0[4, 0])
    assert not bool(mask1[4, 0])
    # cell x=6 is closer to token 1 (dist 1) than token 0 (dist 3)
    assert bool(mask1[6, 0])
    assert not bool(mask0[6, 0])


def test_territory_mask_no_op_for_single_token():
    ii = torch.arange(0, 5, dtype=torch.float32).view(-1, 1).expand(5, 1)
    jj = torch.zeros(5, 1)
    positions = torch.tensor([[2.0, 0.0]])
    mask = territory_mask(ii, jj, positions, self_idx=0)
    assert bool(mask.all())


def test_territory_mask_ties_favor_self():
    # Cell exactly equidistant from both tokens (x=5, tokens at x=3 and
    # x=7) must belong to whichever token's mask is being computed.
    ii = torch.tensor([[5.0]])
    jj = torch.tensor([[0.0]])
    positions = torch.tensor([[3.0, 0.0], [7.0, 0.0]])
    mask0 = territory_mask(ii, jj, positions, self_idx=0)
    mask1 = territory_mask(ii, jj, positions, self_idx=1)
    assert bool(mask0[0, 0])
    assert bool(mask1[0, 0])


def test_territory_mask_handles_coincident_tokens():
    # Two tokens at the exact same position -- degenerate but must not
    # produce an all-False mask for either.
    ii = torch.tensor([[5.0]])
    jj = torch.tensor([[5.0]])
    positions = torch.tensor([[5.0, 5.0], [5.0, 5.0]])
    mask0 = territory_mask(ii, jj, positions, self_idx=0)
    mask1 = territory_mask(ii, jj, positions, self_idx=1)
    assert bool(mask0[0, 0])
    assert bool(mask1[0, 0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_token_detect.py -k territory_mask -v`
Expected: FAIL with `ImportError: cannot import name 'territory_mask'`

- [ ] **Step 3: Implement `territory_mask`**

Add to `model/token_detect.py` (near the top, above `centroid_near`):

```python
def territory_mask(ii, jj, positions, self_idx):
    """Boolean mask, same shape as ii/jj (a window's per-cell coordinate
    grids, broadcastable to (h, w)): True where a cell is at least as
    close to positions[self_idx] as to every other row of positions --
    a Voronoi partition over currently-tracked token positions. Ties
    favor self_idx (a cell exactly equidistant from two tokens belongs
    to both of their masks, never to neither), so the total masked area
    across all tokens never has a gap. Distance is plain Euclidean in
    grid-cell space; this is not derived from `radius` or any other
    physical constant -- it only asks "which token is nearest," using
    whatever positions are currently tracked.

    See docs/superpowers/specs/2026-09-23-token-territory-masking-design.md."""
    self_pos = positions[self_idx]
    self_dist_sq = (ii - self_pos[0]) ** 2 + (jj - self_pos[1]) ** 2
    mask = torch.ones_like(ii, dtype=torch.bool)
    for k in range(positions.shape[0]):
        if k == self_idx:
            continue
        other = positions[k]
        other_dist_sq = (ii - other[0]) ** 2 + (jj - other[1]) ** 2
        mask &= self_dist_sq <= other_dist_sq
    return mask
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_token_detect.py -k territory_mask -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add model/token_detect.py tests/test_token_detect.py
git commit -m "feat: add territory_mask helper for per-token window exclusivity"
```

---

### Task 2: Wire territory masking into `centroid_near`

**Files:**
- Modify: `model/token_detect.py`
- Test: `tests/test_token_detect.py`

**Interfaces:**
- Consumes: `territory_mask(ii, jj, positions, self_idx)` from Task 1.
- Produces: `centroid_near(prob, position, radius, margin=1.0, max_expansions=3, all_positions=None, self_idx=None)` for Task 4 (`TokenModel.step` wiring) and Task 6 (give-up widening) to build on.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_token_detect.py`:

```python
def test_centroid_near_ignores_neighbour_mass_when_territory_given():
    # Two balls close enough that an unmasked window would pull in both;
    # with all_positions/self_idx set, each token's centroid must land
    # near its own ball only.
    balls = [
        {"x": 8.0, "y": 10.0, "vx": 0.0, "vy": 0.0},
        {"x": 12.0, "y": 10.0, "vx": 0.0, "vy": 0.0},
    ]
    prob = _prob_channel(balls, 20, 1.5)
    positions = torch.tensor([[8.0, 10.0], [12.0, 10.0]])

    result0 = centroid_near(prob, positions[0], radius=1.5, margin=3.0,
                             all_positions=positions, self_idx=0)
    result1 = centroid_near(prob, positions[1], radius=1.5, margin=3.0,
                             all_positions=positions, self_idx=1)
    assert torch.allclose(result0, torch.tensor([8.0, 10.0]), atol=0.3)
    assert torch.allclose(result1, torch.tensor([12.0, 10.0]), atol=0.3)


def test_centroid_near_default_unaffected_by_territory_params_when_absent():
    balls = [{"x": 10.5, "y": 7.5, "vx": 0.0, "vy": 0.0}]
    prob = _prob_channel(balls, 20, 0.75)
    pos = torch.tensor([10.0, 7.0])
    baseline = centroid_near(prob, pos, radius=0.75)
    with_none = centroid_near(prob, pos, radius=0.75, all_positions=None, self_idx=None)
    assert torch.equal(baseline, with_none)


def test_centroid_near_still_gives_up_when_own_territory_is_empty():
    # A token whose territory contains no mass at all must still fall
    # through to the give-up return, same as the unmasked case.
    prob = torch.zeros((20, 20))
    positions = torch.tensor([[5.0, 5.0], [15.0, 15.0]])
    result = centroid_near(prob, positions[0], radius=0.75,
                            all_positions=positions, self_idx=0)
    assert torch.equal(result, positions[0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_token_detect.py -k "territory or ignores_neighbour" -v`
Expected: FAIL — `centroid_near() got an unexpected keyword argument 'all_positions'`

- [ ] **Step 3: Implement masking in `centroid_near`**

Modify `model/token_detect.py`'s `centroid_near` signature and body:

```python
def centroid_near(prob, position, radius, margin=1.0, max_expansions=3,
                   all_positions=None, self_idx=None):
    """... (existing docstring) ...

    When `all_positions` (all currently-tracked token positions) and
    `self_idx` (this token's row in that tensor) are both given, window
    mass outside this token's territory (see `territory_mask`) is
    excluded before computing the centroid -- a token's window can never
    read a cell that rightfully belongs to a different tracked token.
    Both default to None, which reproduces the prior unmasked behavior
    exactly (only `find_token_positions`, called before any persistent
    tokens exist, relies on this default)."""
    n = prob.shape[0]
    cx = int(round(float(position[0].detach())))
    cy = int(round(float(position[1].detach())))
    step = max(1, int(math.ceil(radius)))
    base_half = int(math.ceil(radius + margin))
    for expansion in range(max_expansions + 1):
        half = base_half + expansion * step
        i_lo_raw, i_hi_raw = cx - half, cx + half
        j_lo_raw, j_hi_raw = cy - half, cy + half
        if i_hi_raw < 0 or i_lo_raw > n - 1 or j_hi_raw < 0 or j_lo_raw > n - 1:
            continue
        i_lo, i_hi = max(0, i_lo_raw), min(n - 1, i_hi_raw)
        j_lo, j_hi = max(0, j_lo_raw), min(n - 1, j_hi_raw)
        window = prob[i_lo:i_hi + 1, j_lo:j_hi + 1]

        ii = torch.arange(i_lo, i_hi + 1, device=prob.device, dtype=prob.dtype).view(-1, 1)
        jj = torch.arange(j_lo, j_hi + 1, device=prob.device, dtype=prob.dtype).view(1, -1)

        if all_positions is not None and self_idx is not None:
            ii_grid = ii.expand(window.shape[0], window.shape[1])
            jj_grid = jj.expand(window.shape[0], window.shape[1])
            mask = territory_mask(ii_grid, jj_grid, all_positions, self_idx)
            window = window * mask.to(window.dtype)

        total = window.sum()
        if total <= 1e-6:
            continue

        x = (window * ii).sum() / total
        y = (window * jj).sum() / total
        return torch.stack([x, y])
    return position
```

Note this replaces the existing `total <= 1e-6: continue` window/ii/jj block — the `ii`/`jj` construction moves above the `total` check so masking can be applied before summing.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_token_detect.py -v`
Expected: PASS (all tests in the file, including the original 9 plus the 7 new ones)

- [ ] **Step 5: Commit**

```bash
git add model/token_detect.py tests/test_token_detect.py
git commit -m "feat: territory-mask centroid_near's window when tracked positions are given"
```

---

### Task 3: Wire territory masking into `window_collapse_loss`

**Files:**
- Modify: `model/token_losses.py`
- Test: `tests/test_token_losses.py`

**Interfaces:**
- Consumes: `territory_mask` from `model/token_detect.py` (Task 1).
- Produces: `window_collapse_loss(prob, positions, radius, margin=1.0, floor=0.3, all_positions=None, self_idx_offset=0)` for Task 5 (`token_train.py` wiring).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_token_losses.py`:

```python
def test_window_collapse_loss_territory_masks_out_neighbour_mass():
    # Token 0 sits right next to token 1's real mass but has none of its
    # own -- unmasked, its window would see token 1's mass and score
    # fine; territory-masked, it must still be penalized.
    n = 20
    prob = torch.zeros(n, n)
    prob[12, 10] = 0.9  # token 1's real mass
    positions = torch.tensor([[8.0, 10.0], [12.0, 10.0]])

    loss_unmasked = window_collapse_loss(prob, positions[:1], radius=1.5, margin=3.0, floor=0.3)
    loss_masked = window_collapse_loss(prob, positions[:1], radius=1.5, margin=3.0, floor=0.3,
                                        all_positions=positions, self_idx_offset=0)
    assert loss_unmasked == 0.0  # unmasked window at radius+margin=4.5 reaches x=12
    assert loss_masked > 0.0


def test_window_collapse_loss_territory_default_unaffected():
    n = 10
    prob = torch.zeros(n, n)
    prob[5, 5] = 0.8
    positions = torch.tensor([[5.0, 5.0]])
    baseline = window_collapse_loss(prob, positions, radius=0.75, floor=0.3)
    with_none = window_collapse_loss(prob, positions, radius=0.75, floor=0.3,
                                      all_positions=None)
    assert torch.allclose(baseline, with_none)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_token_losses.py -k territory -v`
Expected: FAIL — `window_collapse_loss() got an unexpected keyword argument 'all_positions'`

- [ ] **Step 3: Implement masking in `window_collapse_loss`**

Modify `model/token_losses.py`:

```python
from model.token_detect import territory_mask


def window_collapse_loss(prob, positions, radius, margin=1.0, floor=0.3,
                          all_positions=None, self_idx_offset=0):
    """... (existing docstring) ...

    When `all_positions` is given (the full tracked-token tensor this
    call's `positions` is drawn from, or identical to `positions` itself
    when every token is being penalized in one call), each token i's
    window sum excludes mass outside its territory (see
    model.token_detect.territory_mask) before comparing against `floor`
    -- otherwise a token near a healthy neighbor could get credit for
    mass that was never its own, reintroducing the incentive this loss
    exists to remove (see docs/superpowers/specs/2026-09-23-token-territory-masking-design.md).
    `self_idx_offset` lets a caller penalize a subset of `positions` that
    starts partway through `all_positions` (0 in every current call
    site)."""
    if positions.shape[0] == 0:
        return positions.new_zeros(())
    n = prob.shape[0]
    half = int(math.ceil(radius + margin))
    totals = []
    for i in range(positions.shape[0]):
        cx = int(round(float(positions[i, 0].detach())))
        cy = int(round(float(positions[i, 1].detach())))
        i_lo, i_hi = max(0, cx - half), min(n - 1, cx + half)
        j_lo, j_hi = max(0, cy - half), min(n - 1, cy + half)
        if i_lo > i_hi or j_lo > j_hi:
            totals.append(prob.new_zeros(()))
            continue
        window = prob[i_lo:i_hi + 1, j_lo:j_hi + 1]
        if all_positions is not None:
            ii = torch.arange(i_lo, i_hi + 1, device=prob.device, dtype=prob.dtype).view(-1, 1)
            jj = torch.arange(j_lo, j_hi + 1, device=prob.device, dtype=prob.dtype).view(1, -1)
            ii_grid = ii.expand(window.shape[0], window.shape[1])
            jj_grid = jj.expand(window.shape[0], window.shape[1])
            mask = territory_mask(ii_grid, jj_grid, all_positions, self_idx_offset + i)
            window = window * mask.to(window.dtype)
        totals.append(window.sum())
    total = torch.stack(totals)
    deficit = torch.relu(floor - total)
    return (deficit ** 2).mean()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_token_losses.py -v`
Expected: PASS (all tests in the file, including the original plus the 2 new ones)

- [ ] **Step 5: Commit**

```bash
git add model/token_losses.py tests/test_token_losses.py
git commit -m "feat: territory-mask window_collapse_loss's window sum"
```

---

### Task 4: Wire `TokenModel.step` to pass tracked positions through

**Files:**
- Modify: `model/token_model.py`
- Test: `tests/test_token_model.py`

**Interfaces:**
- Consumes: `centroid_near(..., all_positions=None, self_idx=None)` from Task 2.
- Produces: `TokenModel.step` now territory-masks every non-occluding token's observation read; no signature change to `TokenModel.step` itself.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_token_model.py`:

```python
def test_step_observation_does_not_steal_neighbours_mass():
    # Two non-occluding tokens: TokenModel's actual occlusion threshold is
    # 2*_gate_radius() = 2*(radius + detect_margin) = 2*(0.75+1.0) = 3.5,
    # not 2*radius -- these two are 4.0 apart, clear of that gate. But
    # centroid_near's un-widened window (base_half=ceil(radius+margin)=2,
    # +1/expansion up to max_expansions=3 -> reaches half=5) still spans
    # far enough to overlap without territory masking. Token 0's tracked
    # position has drifted off its own (now-empty) ball; without masking
    # it would pull in token 1's real mass and "correct" onto it. With
    # masking, it must stay put (give up) instead of moving onto token 1's
    # ball.
    torch.manual_seed(4738)
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, observation_weight=1.0)
    positions = torch.tensor([[9.0, 10.0], [13.0, 10.0]])  # 4.0 apart -> not occluding (threshold 3.5)
    velocities = torch.zeros(2, 2)
    hidden = torch.zeros(2, model.dynamics.hidden_dim)
    # Ground truth: token 0's ball has actually moved away/vanished from
    # this frame; only token 1's ball is present, at its tracked position.
    observed_frame = rasterize_tokens(torch.tensor([[13.0, 10.0]]), torch.zeros(1, 2), n, radius)

    new_pos, _, _, _, obs_pos = model.step(positions, velocities, hidden, observed_frame)

    # Token 0 must NOT have been pulled toward token 1's mass.
    assert torch.norm(obs_pos[0] - positions[0]) < 0.1
    assert torch.norm(new_pos[0] - torch.tensor([13.0, 10.0])) > 1.0


def test_occluding_tokens_still_ignore_observation_with_territory_masking():
    torch.manual_seed(4738)
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, observation_weight=1.0)
    positions = torch.tensor([[10.0, 10.0], [10.6, 10.0]])
    velocities = torch.zeros(2, 2)
    hidden = torch.zeros(2, model.dynamics.hidden_dim)
    observed_frame = torch.rand(3, n, n)

    new_pos, new_vel, _, _, _ = model.step(positions, velocities, hidden, observed_frame)
    assert torch.allclose(new_pos, positions, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_token_model.py -k "steal or occluding_tokens_still" -v`
Expected: FAIL — first test fails (token 0 gets pulled onto token 1's mass, `new_pos[0]` close to `[12.0, 10.0]`, not > 1.0 away); second test may already pass (that's fine, it's a regression check).

- [ ] **Step 3: Implement wiring in `TokenModel.step`**

Modify `model/token_model.py`'s `step` method — the `for i in range(...)` loop's `centroid_near` call:

```python
        for i in range(positions.shape[0]):
            if occluding[i] or w == 0.0:
                continue
            op = centroid_near(observed_frame[0], positions[i], self.radius,
                                margin=self.detect_margin,
                                all_positions=positions, self_idx=i)
            obs_pos[i] = op
            corrected_pos[i] = (1 - w) * positions[i] + w * op
            if prev_obs_pos is not None and vw > 0.0:
                obs_vel = (op - prev_obs_pos[i]) / self.dt
                corrected_vel[i] = (1 - vw) * velocities[i] + vw * obs_vel
```

(This adds the explicit `margin=self.detect_margin` that was previously implicit via `centroid_near`'s own default — `TokenModel`'s `detect_margin` already matches `centroid_near`'s default of `1.0`, so this is a no-op for current behavior, made explicit here since we're touching the call anyway and territory masking should use the same window `centroid_near` was already using.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_token_model.py -v`
Expected: PASS (all tests, including existing + 2 new)

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest tests/ -q`
Expected: PASS, 110/110 (106 existing + 4 territory tests in Task 1 already counted... verify actual count matches: 106 + 4 (Task1) + 3 (Task2) + 2 (Task3) + 2 (Task4) = 117)

- [ ] **Step 6: Commit**

```bash
git add model/token_model.py tests/test_token_model.py
git commit -m "feat: wire TokenModel.step to territory-mask observation reads"
```

---

### Task 5: Wire `token_train.py`'s `window_collapse_loss` call to pass tracked positions

**Files:**
- Modify: `model/token_train.py`

**Interfaces:**
- Consumes: `window_collapse_loss(..., all_positions=None, self_idx_offset=0)` from Task 3.
- Produces: no change in default behavior (`--collapse-weight` defaults to 0.0, so this code path isn't exercised in the v9/v14 retrain); makes the loss consistent if collapse-weight is ever turned on again in a future A/B.

- [ ] **Step 1: Read the current call site**

Run: `grep -n "window_collapse_loss" model/token_train.py`

Confirm the call is `window_collapse_loss(pred_grid[0], positions, model.radius, margin=collapse_margin, floor=collapse_floor)` (per the read earlier in this session — line ~92). If the exact line differs, adapt Step 2 to match what's actually there.

- [ ] **Step 2: Update the call site**

Modify `model/token_train.py`:

```python
        if collapse_weight > 0.0:
            total_loss = total_loss + collapse_weight * window_collapse_loss(
                pred_grid[0], positions, model.radius, margin=collapse_margin, floor=collapse_floor,
                all_positions=positions, self_idx_offset=0,
            )
```

- [ ] **Step 3: Run the full suite**

Run: `python3 -m pytest tests/ -q`
Expected: PASS, same count as Task 4's Step 5 (this task adds no new tests — `token_train.py` has no direct unit test exercising this line per the existing test suite structure; it's covered indirectly by `tests/test_token_train.py`'s existing training-loop tests, which must still pass unchanged).

- [ ] **Step 4: Commit**

```bash
git add model/token_train.py
git commit -m "feat: pass tracked positions through to window_collapse_loss in training loop"
```

---

### Task 6: Widen `centroid_near`'s give-up search now that territory bounds it safely

**Files:**
- Modify: `model/token_model.py`
- Test: `tests/test_token_model.py`

**Interfaces:**
- Consumes: `centroid_near(..., max_expansions=3, ...)` from Task 2 (existing default parameter).
- Produces: `TokenModel.__init__` gains `max_expansions=6` parameter, threaded to both `centroid_near` call sites in `step`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_token_model.py`:

```python
def test_token_model_defaults_to_widened_give_up_search():
    model = TokenModel(n=20, radius=0.75, dt=0.15)
    assert model.max_expansions == 6


def test_step_recovers_drifted_token_with_widened_search():
    # A token has drifted 5 cells from its ball (beyond the old
    # max_expansions=3's reach at radius=0.75/margin=1.0, whose windows
    # only grow by `step`=1 cell per expansion -- 3 expansions reaches
    # ~2+3=5 cells, borderline; push to a position where the un-widened
    # search would still miss but the widened one finds it) -- with the
    # widened default it must still recover the ball via observation
    # correction rather than giving up.
    torch.manual_seed(4738)
    n, radius, dt = 20, 0.75, 0.15
    model = TokenModel(n=n, radius=radius, dt=dt, observation_weight=1.0)
    true_ball_pos = torch.tensor([[10.0, 10.0]])
    drifted_position = torch.tensor([[16.0, 10.0]])  # 6 cells off
    velocities = torch.zeros(1, 2)
    hidden = torch.zeros(1, model.dynamics.hidden_dim)
    observed_frame = rasterize_tokens(true_ball_pos, torch.zeros(1, 2), n, radius)

    _, _, _, _, obs_pos = model.step(drifted_position, velocities, hidden, observed_frame)

    assert torch.allclose(obs_pos[0], true_ball_pos[0], atol=0.5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_token_model.py -k "widened" -v`
Expected: FAIL — `AttributeError: 'TokenModel' object has no attribute 'max_expansions'` on the first test; second test fails because the un-widened search (max_expansions=3) doesn't reach 6 cells and `obs_pos[0]` stays at `drifted_position`.

- [ ] **Step 3: Implement**

Modify `model/token_model.py`'s `TokenModel.__init__` signature to add `max_expansions=6`, store as `self.max_expansions = max_expansions`, and pass it at both `centroid_near` call sites in `step` (the observation-correction call already modified in Task 4, plus any other call site — check with `grep -n "centroid_near" model/token_model.py`):

```python
    def __init__(self, n, radius, dt, hidden_dim=32, neighbor_radius=3.0,
                 detect_threshold=0.1, observation_weight=0.5, detect_margin=1.0,
                 max_init_speed=20.0, velocity_weight=0.0, max_expansions=6):
        super().__init__()
        ...
        self.max_expansions = max_expansions
```

```python
            op = centroid_near(observed_frame[0], positions[i], self.radius,
                                margin=self.detect_margin, max_expansions=self.max_expansions,
                                all_positions=positions, self_idx=i)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_token_model.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run the full suite**

Run: `python3 -m pytest tests/ -q`
Expected: PASS, all tests green (117 + 2 new = 119)

- [ ] **Step 6: Commit**

```bash
git add model/token_model.py tests/test_token_model.py
git commit -m "feat: widen TokenModel's give-up search now that territory masking bounds it"
```

---

### Task 7: Retrain (v14) and verify against v9 baseline

**Files:**
- Create: `scripts/polaris_train_token_v14.sh`

**Interfaces:**
- Consumes: `model/token_train.py`'s existing CLI (unchanged), `checkpoints/token_dataset_10000_h12_seed4738.pt` (existing cached dataset, per Global Constraints — do not regenerate or resize).
- Produces: `checkpoints/token_model_h12_v14.pt`.

- [ ] **Step 1: Create the training script**

Copy `scripts/polaris_train_token_v9.sh` to `scripts/polaris_train_token_v14.sh`, changing only the job name, output log name, checkpoint filename, and the comment (every hyperparameter value stays identical to v9 — this is the single-variable A/B the spec requires):

```bash
#!/bin/bash
#SBATCH --job-name=bounce-token-model-v14
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=bounce-token-model-v14-%j.log

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

# v14: identical recipe to v9 (state-loss-only ablation) -- the only
# change is the territory-masking fix in model/token_detect.py,
# model/token_losses.py, model/token_model.py (see
# docs/superpowers/specs/2026-09-23-token-territory-masking-design.md).
# Single-variable A/B against v9's 6/48 dropout baseline -- do not
# change epochs/dataset alongside this fix.
echo "[$(date -Iseconds)] starting training"
python -m model.token_train \
  --dataset-cache checkpoints/token_dataset_10000_h12_seed4738.pt \
  --n 20 --horizon 12 --epochs 3 \
  --hidden-dim 32 --neighbor-radius 4.0 \
  --bg-weight 0.05 --peak-weight 0.5 --boundary-weight 0.0 \
  --state-weight 1.0 --grid-weight 0.0 \
  --lr 1e-3 --ramp-epochs 2 \
  --seed 4738 --log-every 200 \
  --checkpoint checkpoints/token_model_h12_v14.pt
echo "[$(date -Iseconds)] training done"
```

- [ ] **Step 2: Commit the script**

```bash
git add scripts/polaris_train_token_v14.sh
git commit -m "feat: add v14 training script (territory-masking fix, v9 recipe unchanged)"
```

- [ ] **Step 3: Submit the job**

Use the `mcp__polaris__polaris_submit_job` tool (per project memory: Bounce ML training compute goes on Polaris, not TIDE) with `scripts/polaris_train_token_v14.sh`. Wait for completion via `mcp__polaris__polaris_job_status`.

- [ ] **Step 4: Download the checkpoint**

Use `mcp__polaris__polaris_download` to pull `checkpoints/token_model_h12_v14.pt` to the local `checkpoints/` directory.

- [ ] **Step 5: Run the dropout diagnostic**

Run: `PYTHONPATH=. python3 scripts/diagnose_token_dropout.py --checkpoint checkpoints/token_model_h12_v14.pt --num-seeds 48 --num-steps 20`

Record the dropout count. Compare against v9's baseline of 6/48 — this is the number to beat, not just "better than v13's 34/48."

- [ ] **Step 6: Render the diagnostic grid and run the standard unbiased visual review**

Run `scripts/render_token_diagnostic_grid.py` (or the project's current equivalent — check `scripts/render_token_multiseed_grid_video.py` if the diagnostic-grid script has since been superseded) against the v14 checkpoint, save to `videos/token_model_v14_diagnostic_grid.png` per the project's standing video-output convention.

Dispatch a fresh subagent with no hypothesis primed (per `[[feedback_auto_run_video_analysis_on_sim_finish]]`) to describe what it sees in the grid, specifically noting whether it observes clean vanishing, merged multi-color patches (the v13 failure shape this fix targets), or something else.

- [ ] **Step 7: Log the result**

Append an entry to `docs/debugging/experiment-log.md` (append-only, per its own header instructions) summarizing: dropout count vs. v9's 6/48, whether the v13 merged-patch failure shape is gone per the unbiased review, and the commit hash of the territory-masking implementation. Per the spec's Validation plan step 5: if dropout count still exceeds v9's baseline, do not chain another fix in the same run — flag it as a new decision point instead of guessing the next change.

- [ ] **Step 8: Commit the log entry and any generated artifacts**

```bash
git add docs/debugging/experiment-log.md videos/token_model_v14_diagnostic_grid.png
git commit -m "docs: log v14 result -- territory masking vs v9 baseline (N/48 dropout)"
```

(Replace `N` with the actual observed count before writing the commit message — this is a real result to report, not a placeholder to leave unfilled.)
