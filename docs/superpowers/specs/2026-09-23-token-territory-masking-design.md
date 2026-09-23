# Token Territory Masking — Design

Date: 2026-09-23

## Goal

Fix the token-per-ball model's identity-collapse/give-up dropout problem
structurally, without another loss-space A/B. Five straight single-variable
attempts (v9 baseline through v13, see `docs/debugging/experiment-log.md`)
have all failed to reduce dropout by changing the loss or training regime;
v13 (`window_collapse_loss`) made it categorically worse and revealed the
real mechanism: a struggling token's observation window can pull in a
*neighboring* token's real mass instead of failing independently, producing
merged multi-color patches at ball boundaries rather than clean vanishing.
Teacher-forced (single-step) dynamics error has stayed flat and small across
every checkpoint regardless of loss/epoch changes — the bug is in the
observation branch's window read under self-feed, not in `TokenDynamics`'s
predictions.

## Root cause being fixed

`centroid_near` (`model/token_detect.py`) computes each token's observation
correction from a window of `prob` mass centered on that token's *own*
predicted position, independently of where every other tracked token is.
Nothing prevents two tokens' windows from overlapping and summing the same
underlying ball's mass. Under self-feed, once a token's own position drifts
enough that its window's mass thins out, the give-up-avoidance widening
(added earlier this project to fix a different dropout mechanism) searches
further outward — and can now cross into a neighboring, healthy token's
territory, dragging that token's real mass into its own centroid. This is
confirmed causally in v13: `window_collapse_loss` penalized *having no
mass* without caring *whose* mass satisfied it, and the network learned
exactly that shortcut (identity theft), the cheapest way to escape the
penalty once its own ball is hard to track.

## Constraint carried over from the token-per-ball design

Per `docs/superpowers/specs/2026-09-22-token-per-ball-model-design.md`:
after sequence-start initialization, token identity is carried by
persistent recurrent state, not re-derived by full-grid detection each
step. This fix must preserve that — it changes what mass a token's
*existing* window read is allowed to see, not how tokens are identified or
re-detected.

## Design

### 1. Territory mask

New pure function in `model/token_detect.py`:

```python
def territory_mask(ii, jj, positions, self_idx):
    """Boolean mask, same shape as ii/jj (a window's cell-coordinate grids):
    True where a cell is at least as close to positions[self_idx] as to any
    other row of positions. Ties favor self_idx."""
```

`ii`, `jj` are the same per-cell coordinate grids `centroid_near` already
builds for its weighted-centroid sum. `positions` is the full tracked-token
tensor (`TokenModel.step`'s `positions` argument, not the corrected/final
one — the same snapshot every token's window read uses this step). Distance
is plain Euclidean in grid-cell space; no new physical constant introduced
(unlike the occlusion gate, this isn't derived from `radius` — it's a
pointwise nearest-token partition, i.e. a Voronoi diagram over currently
tracked positions).

Cost: for a window of `w` cells and `k` tokens, an O(w·k) distance
comparison per token, O(w·k²) per step. Token counts in this project are
small (a handful of balls), so this is not a performance concern.

### 2. `centroid_near` changes

Add optional parameters `all_positions=None, self_idx=None`. When both are
given, after building `window` (as today), zero out any cell where
`territory_mask` is False *before* computing `total = window.sum()` and the
weighted centroid. The give-up control flow (widen on empty, bail after
`max_expansions`) is otherwise unchanged — territory masking only changes
which mass a window can see, not the retry structure. When `all_positions`
is `None` (the default), behavior is byte-for-byte identical to today —
`find_token_positions` (init-time, called before any persistent tokens
exist) keeps calling it with no territory args.

### 3. Give-up threshold

Once territory makes cross-token contamination structurally impossible,
increase `max_expansions` (exact value determined empirically during
implementation — start by doubling it, 3 → 6, and check the calibration
script) so a token searches further before giving up on its own ball. This
was previously unsafe (direction (b) from the v9-v12 synthesis, explicitly
flagged as risking a neighbor hijack) and is now safe because the widened
search still cannot cross the territory boundary. The terminal fallback
(`return position` unchanged when every expansion comes up empty) stays —
a token whose ball has genuinely left the grid or is occluded everywhere
still needs a defined behavior.

### 4. `window_collapse_loss` changes

Same territory mask applied when summing the window in
`model/token_losses.py::window_collapse_loss`, using the token's own
tracked `positions` the same way `centroid_near` does. Without this, the
loss would penalize a token for lacking mass that was never its own to
claim (a healthy token near a neighbor would get artificially penalized by
territory it doesn't own), reintroducing pressure toward the same failure
this design fixes. `floor` may need recalibration once masking shrinks the
achievable window sum for tokens near a neighbor — check via the existing
`scripts/calibrate_window_total.py`-style probe mentioned in the function's
docstring before reusing 0.3 unchanged.

### 5. `TokenModel.step` wiring

`model/token_model.py`'s per-token loop already has `positions` (the full
tensor) and `i` (the loop index) in scope. The `centroid_near` call becomes:

```python
op = centroid_near(observed_frame[0], positions[i], self.radius,
                    all_positions=positions, self_idx=i)
```

Occluding tokens are unaffected — they already skip the observation branch
entirely via the existing gate check earlier in the loop. Territory masking
only changes behavior for the non-occluding tokens that already read an
observation this step, which is exactly the population v13 showed
committing identity theft (mass theft was observed between tokens that
were, by the existing gate's definition, not occluding each other).

## Error handling

- Single-token case (no other tracked tokens): `territory_mask` returns
  all-True (nothing to be closer to) — identical to current unmasked
  behavior.
- All tokens at the exact same position (degenerate case, e.g. two tokens
  collapsed onto one ball already): tie-break favors `self_idx`, so masking
  doesn't produce an empty mask for every token simultaneously.
- Territory masking can only shrink a window's available mass, never grow
  it — so the existing "give up on truly empty window" path still triggers
  correctly, just later (wider search) than before.

## Testing

New tests in `tests/test_token_detect.py`:

1. Two tokens straddling a shared blob: each token's `centroid_near` call
   (with `all_positions`/`self_idx` set) returns a centroid computed only
   from its own side of the midline, verified against a hand-built `prob`
   grid.
2. Single-token / `all_positions=None` calls are unaffected (regression —
   existing tests keep passing unmodified).
3. A token whose entire un-masked-but-territory-restricted window is empty
   still falls through to the give-up return, even with `all_positions` set
   (masking doesn't break the existing bailout).
4. Tie case: two tokens exactly equidistant from a cell — cell goes to
   `self_idx` in both calls (each token "wins" the tie for itself), not
   dropped from both.

New test in `tests/test_token_losses.py`: `window_collapse_loss` with
`all_positions` wired through, confirming a token's penalty is computed
only from its own territory (analogous to test 1 above).

`tests/test_token_model.py`: extend the existing `TokenModel.step` tests to
cover a two-token near-contact case, confirming the observation correction
each token receives no longer includes the other's mass (this is the
integration-level version of the unit tests above, exercising the actual
wiring change in section 5).

## Validation plan

Same staged approach as the original token-per-ball design
(`docs/superpowers/specs/2026-09-22-token-per-ball-model-design.md`'s
Validation plan), scoped to this fix:

1. Unit tests above, plus full existing suite (106/106 today) staying
   green.
2. Retrain with the v9 recipe unchanged (3 epochs, same dataset) — single
   variable, per this project's systematic-debugging discipline. Do not
   change epoch count or dataset size alongside this fix (see conversation:
   more training has previously made dropout worse under the old
   mechanism, and conflating variables would make the result
   uninterpretable).
3. Run `scripts/diagnose_token_dropout.py`'s 48-seed dropout count; compare
   against v9's baseline of 6/48 as the number to beat, not just "better
   than v13."
4. Standard diagnostic-grid + unbiased fresh-subagent visual review
   (`videos/`, per project standing practice) — specifically check whether
   the v13 merged-multi-color-patch failure shape is gone, not just whether
   the aggregate dropout count improved.
5. If dropout count still exceeds v9's baseline, do not chain another
   single-variable fix on top in the same run — surface it as a new
   decision point, same as after v10-v13.

## Open questions deferred to implementation

- Exact new `max_expansions` value (start at 6, tune against the
  calibration script if needed).
- Whether `window_collapse_loss`'s `floor=0.3` needs recalibration under
  territory masking (check empirically, adjust if the calibration probe
  shows a shifted baseline for tokens near a neighbor).
