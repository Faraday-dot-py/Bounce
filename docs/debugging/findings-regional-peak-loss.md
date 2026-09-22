# Findings: is regional/local peak-preservation a viable fix for the blur-vs-give-up tradeoff?

Investigation of Problem 1 in `docs/debugging/flownet-open-issues-v2.md`.
Investigation only -- `model/net.py`, `model/losses.py`, `model/train.py`
not edited. All regional peak-loss variants implemented standalone in a
scratchpad script (not committed, see bottom); the "base" (no-peak-term)
loss and the production "global peak" term are computed by importing and
calling the real, frozen `occupancy_weighted_mse` (`model/losses.py`)
directly, to keep numbers comparable to `findings-peak-decay-
dissolution.md` Part 5, which this extends.

## Summary of conclusion up front

**Regional/local peak-preservation is a real, measurable partial
mitigation, not a clean fix.** It reliably beats the existing global
peak term at low-to-moderate drift and low-to-moderate ball density, but
degrades toward the same "give up wins" failure as density and drift
both increase together, for a structural reason specific to
region-based formulations: any formulation that checks "is there *a*
peak somewhere in this region" rather than "is there a peak at *this
specific* location" gets masked by a neighboring confidently-predicted
ball sharing the same region -- which becomes more likely exactly as
density rises, i.e. in precisely the regime where the give-up pathology
is worst. No single fixed tile/window size is robust across both axes
simultaneously; the fix requires the region size to scale with expected
positional drift, but larger regions are also more likely to contain
multiple balls at high density, reintroducing the masking failure.
Confirms option (1) from `findings-peak-decay-dissolution.md` as
**worth adopting as an improvement over the current global peak term,
but not sufficient on its own** to resolve the underlying blur-vs-give-up
tradeoff at the densities/drift levels this architecture actually faces
past the first few autoregressive steps.

## Method

Synthetic single-step probe, generalizing the existing findings doc's
one-shot two-ball probe (Part 5). Scenario: `n=50` grid, radius=0.75,
`bounce.init_balls` (seed derived from 4738 per trial). Half the balls
(rounded up) in each scenario are designated "uncertain" (the ones a
hypothetical prior autoregressive step lost track of); the rest are
"certain" and kept at their true position in every prediction. Two
competing predictions are built per scenario:

- **give-up**: uncertain balls omitted entirely (zero PROB there),
  certain balls splatted correctly.
- **commit**: uncertain balls splatted at their true position **plus** a
  random-direction drift of `drift_px` cells (simulating "confidently
  predicting a nearby-but-wrong position"), certain balls unchanged.

Swept: **ball counts** {2, 5, 10, 20} (density), **drift** {0.5, 1, 2, 4,
8} px, 15 random trials per (balls, drift) cell = 300 trials per
formulation. `source` = `target` (matches the existing probe). Channel
weights `[1.0, 0.1, 0.1]`, `bg_weight=0.05`, `peak_weight=0.1` (all
training defaults, `model/train.py`).

**Formulations compared** (all added to the real `base` loss, i.e.
`occupancy_weighted_mse(..., peak_weight=0.0)`, exactly the way
`peak_weight * peak_loss` is added in the real loss):

- `global` -- the production term: single frame-wide max, as-is.
- `tile4` / `tile8` / `tile16` -- `F.max_pool2d`-based local max per
  tile, tile edge 4/8/16 cells on the 50x50 grid (per the task spec).
- `topk8` -- top-8 non-max-suppressed local maxima (3x3 window), MSE
  between pred's and target's **sorted magnitude lists** (no positional
  matching).
- `component` -- per-connected-component of target's occupancy mask
  (`scipy.ndimage.label`), comparing pred's max within each true
  component's footprint to target's.
- `window5` -- fixed 5x5-cell window centered on each ground-truth
  ball's *true* position (not tile-aligned, not merged across nearby
  balls), comparing pred's max in that window to target's.

"Loss-optimal" = which prediction (give-up vs. commit) scores lower
total loss; a formulation is doing its job if commit reliably beats
give-up once positional uncertainty is present.

## Results: win-rate table (commit beats give-up, out of 15 trials)

```
balls  drift    base   global   tile4   tile8  tile16   topk8  component  window5
    2    0.5   10/15    10/15   10/15   12/15   13/15   13/15      14/15    15/15
    2    1.0    5/15     6/15    5/15    7/15   12/15   14/15       7/15    14/15
    2    2.0    0/15     4/15    2/15    8/15   11/15   14/15       0/15    14/15
    2    4.0    0/15     2/15    0/15    3/15   10/15   15/15       0/15     0/15
    2    8.0    0/15     2/15    0/15    3/15    5/15   13/15       0/15     1/15
    5    0.5   11/15    11/15   11/15   12/15   13/15   15/15      13/15    15/15
    5    1.0    2/15     5/15    4/15   10/15   12/15   15/15       8/15    15/15
    5    2.0    0/15     1/15    0/15    3/15   10/15   13/15       0/15    13/15
    5    4.0    0/15     1/15    0/15    0/15    7/15   13/15       0/15     0/15
    5    8.0    0/15     1/15    0/15    0/15    5/15   13/15       0/15     2/15
   10    0.5   11/15    11/15   11/15   12/15   13/15   15/15      14/15    14/15
   10    1.0    4/15     6/15    4/15    6/15   12/15   15/15      10/15    15/15
   10    2.0    0/15     3/15    0/15    3/15    7/15   15/15       0/15    15/15
   10    4.0    0/15     1/15    0/15    0/15    9/15   15/15       0/15     2/15
   10    8.0    0/15     0/15    0/15    0/15    1/15   15/15       0/15     2/15
   20    0.5   15/15    15/15   15/15   15/15   15/15   15/15      15/15    15/15
   20    1.0    1/15     2/15    3/15    6/15    9/15    5/15       5/15    15/15
   20    2.0    0/15     0/15    0/15    0/15    3/15    0/15       0/15    14/15
   20    4.0    0/15     0/15    0/15    0/15    2/15    0/15       0/15     0/15
   20    8.0    0/15     0/15    0/15    0/15    1/15    1/15       1/15     1/15
```

Mean across all 20 (balls, drift) cells: `base` 0.197, `global` 0.270,
`tile4` 0.217, `tile8` 0.333, `tile16` 0.567, `topk8` 0.780, `component`
0.290, `window5` 0.607.

## `topk8`'s apparently strong score is not real -- it is positionally blind

`topk8` looks best in the raw table, but a direct adversarial check
falsifies it as a usable formulation: it compares only the **sorted
magnitude list** of pred's top-8 local maxima against target's, with no
positional matching at all.

```
topk8 loss, all peaks at WRONG random locations (not near any true ball): 0.00921
topk8 loss, honest give-up on half the balls:                            0.02882
```

A prediction that hallucinates the right *number* and *magnitude* of
peaks at **completely wrong locations** scores a **lower** loss than
honestly giving up -- worse than useless as a "regional preservation"
term, since it would incentivize confident hallucination anywhere in
the frame over either committing near the true position or giving up.
`topk8` is disqualified; its win-rate numbers above should be
disregarded as evidence for anything except "top-k-by-magnitude alone
is not a viable formulation without position matching."

## Why tile/component formulations degrade with density: neighbor-masking, confirmed mechanically

`tile16` and `component` both check "is there *a* peak somewhere in
this region/blob," not "is there a peak at *this* ball's position." At
higher density, a give-up ball increasingly shares its tile (or, for
`component`, its merged connected-component blob once balls' disks
overlap) with a **certain** ball that is still confidently and
correctly predicted -- which keeps that tile/component's local max high
regardless of whether the uncertain ball's own position was preserved,
masking the give-up strategy's actual failure. Mean balls-per-tile,
computed directly from grid size and tile geometry (uniform-random
placement, no need to run the model):

```
tile=16 (4x4=16 tiles):  2 balls: 0.125/tile   5 balls: 0.312/tile  10 balls: 0.625/tile  20 balls: 1.250/tile
tile=8  (7x7=49 tiles):  2 balls: 0.041/tile   5 balls: 0.102/tile  10 balls: 0.204/tile  20 balls: 0.408/tile
tile=4 (13x13=169 tiles):2 balls: 0.012/tile   5 balls: 0.030/tile  10 balls: 0.059/tile  20 balls: 0.118/tile
```

`tile16`'s mean-balls-per-tile crosses 1.0 exactly at the 20-ball
density where its win-rate collapses hardest (0.567 mean overall, but
9/15, 3/15, 2/15, 1/15 at drift >=1 for 20 balls) -- directly consistent
with the masking mechanism, not coincidence. `component` shows the same
pattern for the same reason (merged blobs at high density), and is
weaker overall than `tile16` because target-ball disks (radius 0.75)
already touch/merge at fairly modest density, so its "region" grows
uncontrolled with density in exactly the wrong direction.

## `window5` (ball-centered, not tile-aligned) avoids masking but is capped by its own radius

`window5` fixes the neighbor-masking problem for isolated balls (its
region is centered on each true ball, not a shared grid cell), and
clearly wins at drift <= 2px even at 20 balls (14-15/15). But it fails
completely once drift exceeds its half-width (2px, from a 5x5 window):
at drift 4 and 8, the committed-but-wrong peak falls **outside** the
window entirely, so window5 sees zero peak near the true position under
*either* give-up or commit -- both strategies look identical to it, and
the underlying base loss (which still prefers give-up, per the original
finding doc) decides. This is not a bug in the implementation; it is
the fundamental tension the whole investigation turns on: **a region
must be large enough to contain the model's actual (possibly quite
wrong) predicted position to reward committing over giving up, but a
larger region is also more likely to contain an unrelated neighboring
ball at realistic densities, which reintroduces masking.** No fixed
region size (tile or window) escapes this tradeoff; it would need to
scale with both expected drift (which grows with autoregressive step
count) and inverse ball density (which is a property of the scene, not
knowable per-ball at training time without per-ball ground truth
tracking).

## `peak_weight` sensitivity: turning up the term's weight helps, but plateaus below full coverage in the hardest regime

Swept `peak_weight` in {0.1, 0.5, 1.0, 3.0, 10.0} for `tile16` and
`window5` at three representative hard configs (dense and/or
high-drift):

```
balls=10 drift=4.0:  tile16   9/15 -> 11/15 (saturates by w=0.5)   window5  2/15 -> 4/15  (saturates by w=1.0)
balls=20 drift=2.0:  tile16   3/15 -> 12/15 (saturates by w=1.0)   window5 14/15 -> 15/15 (saturates by w=0.5)
balls=20 drift=4.0:  tile16   2/15 -> 9/15  (saturates by w=1.0)   window5  0/15 -> 12/15 (saturates by w=3.0)
```

Raising the weight well above the current `peak_weight=0.1` default
materially improves both formulations and both saturate at a higher
win-rate than the un-tuned default -- so weight alone was leaving real
headroom on the table in the v7 checkpoint's configuration. But even at
10x the weight, none of these three hardest configs reaches 15/15: the
ceiling in the worst case (20 balls, drift 4) is 9/15 (`tile16`) and
12/15 (`window5`), not full coverage. Weight tuning amplifies whichever
formulation is used, it does not remove the structural masking/coverage
tradeoff above.

## Answering the task's question directly

**Is there a tile size, window, or formulation where committing is
reliably cheaper than giving up, across the scenarios tested?**
No single one, across the full swept range (2-20 balls, 0.5-8px drift).
`window5` is the best-behaved of the tested formulations (highest
win-rate at low-to-moderate drift regardless of density, and the only
one immune to neighbor-masking), but its effective range is capped by
its own fixed size, and none of the tested region-based formulations
holds up simultaneously at high density *and* high drift -- which is
exactly the regime `findings-peak-decay-dissolution.md` establishes as
the normal state past the first few autoregressive steps of a real
multi-ball rollout.

**Is regional peak-preservation a real fix, a partial mitigation, or
does it not resolve the tradeoff at all?**
**Partial mitigation.** It is a clear, quantitatively demonstrated
improvement over the current global peak term at realistic near-term
drift (0.5-2px, which is where autoregressive rollout is at steps
1-3ish given the flow magnitudes measured in the earlier finding doc)
and low-to-moderate density, and is worth adopting in place of (or in
addition to) the global term, ideally as a multi-scale combination
(e.g. `window5`-style for recent/small drift plus a coarser tile for
long-range coverage) with `peak_weight` raised from the current 0.1.
But it does **not** resolve the underlying blur-vs-give-up tradeoff at
the higher drift and higher density that dominate longer rollouts and
denser scenes -- the same forced choice documented in
`findings-peak-decay-dissolution.md` reappears there, just at a later
step / higher density than with the global term alone. This does not
falsify option (1) as worthless, but it does falsify treating it as a
complete solution; per that doc's own ranking, options (2) (bound and
report the honest horizon) and eventually (3) (a non-dense-regression
output representation) remain necessary regardless of whether a
regional peak term is adopted.

## Repro (not committed to the repo)

Scratchpad scripts used for this investigation (not in `scripts/`,
session-local):

- `probe_regional_peak.py` -- main sweep: scenario construction,
  give-up vs. commit prediction builder, all 8 formulations
  (`base`/`global`/`tile4`/`tile8`/`tile16`/`topk8`/`component`/
  `window5`), win-rate table.
- `topk_sanity.py` -- adversarial check that falsifies `topk8`
  (wrong-location peaks scoring lower than honest give-up).
- `peakweight_sweep.py` -- `peak_weight` sensitivity sweep for `tile16`
  and `window5` at three hard (dense/high-drift) configs.
- `crowding_stat.py` -- mean-balls-per-tile geometry check supporting
  the neighbor-masking mechanism.

If a follow-up wants a specific formulation (`window5`-style
ball-centered local peak, or a multi-scale combination) taken forward,
it should be implemented in `model/losses.py` proper, validated with a
real retrain the same way v7's global peak term was, and re-run through
the standard diagnostic-grid + unbiased-subagent-review pipeline --
this investigation is synthetic-loss-value evidence only, not a
retrain.
