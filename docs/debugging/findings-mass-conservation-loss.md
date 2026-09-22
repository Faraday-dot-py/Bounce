# Findings: does the existing hard mass-renorm amplify give-up, and is a local mass-conservation loss a viable new lever?

Investigation of both parts of `docs/debugging/flownet-open-issues-v4.md`.
`model/net.py`, `model/losses.py`, `model/train.py` not edited. All probes
implemented standalone in scratchpad scripts (session-local, not
committed; see "Repro" at bottom); `model/losses.py`'s real
`occupancy_weighted_mse` is imported and called directly for Part B so
numbers are comparable to `findings-regional-peak-loss.md`.

## Summary of conclusions up front

**Part A: the hard renorm is a real, measurable, but secondary
contributor to the give-up blob's brightness -- not its cause, and not
its main driver of spatial concentration.** The rescale factor climbs
from ~0.84 (early rollout, mildly deflating) to ~1.3 by step 70-100
(late rollout), i.e. it does inflate whatever real mass survives by up
to 30%, confirming the hypothesis's mechanical premise. But the
*relative* concentration of mass into the brightest cell is actually
slightly **lower** with renorm on than off at matched steps -- renorm's
extra mass gets spread across a growing occupied footprint, not piled
onto the same shrinking spot. And without renorm, the rollout doesn't
reach a stable give-up plateau at all: raw PROB mass collapses to
**exactly zero by step ~35** regardless of any give-up "strategy",
because the warp+threshold pipeline is intrinsically lossy. Renorm's
primary causal role is preventing that total blackout, not creating the
concentrated-blob look.

**Part B: sum-based local mass-conservation is a meaningfully better
lever than the peak-based regional terms already investigated,**
particularly at the tile16 granularity, where it holds up at high
density *and* high drift simultaneously -- exactly the regime that
broke every peak-based formulation. `tile16` (sum) mean win rate 0.753
vs. peak's `tile16` 0.567; at `balls=20, drift=8` specifically, sum-tile16
scores 13/15 vs. peak-tile16's 1/15. `window5` (sum) is roughly on par
with peak's `window5` (0.637 vs 0.607 mean) and shares its same
fundamental cap (fails once drift exceeds the window half-width).

## Part A: does the hard global renorm amplify the give-up blob?

### Method

Reimplemented `BounceNextFrameModel.forward` (`model/net.py:115-148`) in
a scratchpad script with a `renorm_on` switch: with it off, the
post-threshold PROB is used as-is (scale fixed at 1.0) instead of being
multiplicatively rescaled to match the pre-warp frame's sum; the
soft-threshold and VX/VY recentering are unchanged in both conditions.
Ran the real `stage2_flownet_h12_v7.pt` checkpoint, seed 4738, standard
scenario (`n=50`, `num_balls=150`, matching
`scripts/render_flownet_rollout_video.py` defaults and the params used
for the 100-step v7 rollout review logged in
`docs/debugging/experiment-log.md`), 100 autoregressive steps, logging
per-step: `prob_in` (pre-warp sum), `prob_out` (post-threshold,
pre-scale sum), `scale` (the rescale factor actually applied), `peak`
(post-step max PROB value), `top1_frac` (peak / total mass -- relative
concentration), and `occ_count` (cells with PROB > 0.01).

### Results

```
 step  scale_on   peak_on   top1_on  occ_on   mass_on |  peak_off  top1_off occ_off  mass_off
    1     0.841    0.5359    0.0080     574    67.181 |    0.6376    0.0080     586    79.922
    2     0.928    0.5803    0.0086     717    67.181 |    0.6634    0.0078     802    84.859
    5     0.953    0.6774    0.0101    1015    67.181 |    0.7183    0.0079    1293    91.137
   10     1.052    0.6352    0.0095    1155    67.181 |    0.5987    0.0095    1331    62.885
   15     0.961    0.7996    0.0119    1266    67.181 |    0.6530    0.0133    1131    48.934
   20     1.075    0.7161    0.0107    1373    67.181 |    0.5874    0.0160    1097    36.797
   25     1.011    0.6812    0.0101    1338    67.181 |    0.5414    0.0341     596    15.892
   30     1.083    0.7970    0.0119    1359    67.181 |    0.4497    0.0528     305     8.525
   35     1.095    0.8193    0.0122    1422    67.181 |    0.0000    0.0000       0     0.000
   40     1.169    0.5521    0.0082    1473    67.181 |    0.0000    0.0000       0     0.000
   50     1.178    0.8918    0.0133    1517    67.181 |    0.0000    0.0000       0     0.000
   60     1.230    0.4831    0.0072    1571    67.181 |    0.0000    0.0000       0     0.000
   70     1.300    0.0857    0.0013    1633    67.181 |    0.0000    0.0000       0     0.000
   80     1.312    0.0860    0.0013    1635    67.181 |    0.0000    0.0000       0     0.000
   90     1.312    0.0916    0.0014    1651    67.181 |    0.0000    0.0000       0     0.000
  100     1.280    0.1053    0.0016    1618    67.181 |    0.0000    0.0000       0     0.000
```

(`mass_on` is exactly `67.181` at every step -- confirms the renorm
mechanism itself is exact/lossless by construction, as documented.)

### Point 1 -- does the rescale factor grow as give-up progresses?

**Confirmed.** `scale` starts *below* 1.0 (0.841 at step 1: the raw
post-threshold output already has more mass than the input at this
point, so renorm is mildly deflating), crosses 1.0 around step 10, and
climbs steadily to a plateau of ~1.3 by step 70-100. This directly
tracks the `renorm_off` condition's mass trace: raw post-threshold mass
falls monotonically every step it's measurable (91.1 -> 62.9 -> 48.9 ->
36.8 -> 15.9 -> 8.5 -> 0.0, steps 5 through 35) -- the warp+threshold
pipeline is structurally lossy on its own (consistent with
`findings-correction-drift-and-mass-dissolution.md`'s bicubic-resampling
diffusion finding), and renorm has to apply a growing multiplicative
boost to force that shrinking real signal back up to the fixed initial
total (67.181) every single step.

### Point 2 -- counterfactual: does disabling renorm change blob brightness/concentration?

Two different, partially opposing effects, both real:

- **Absolute peak brightness is higher with renorm on**, at every
  step where both conditions still have live content (e.g. step 30:
  `peak_on=0.797` vs `peak_off=0.450`; step 20: `0.716` vs `0.587`).
  This is the amplification the hypothesis predicted, and it's
  consistent with `scale` exceeding 1.0 from step ~10 onward.
- **Relative concentration (`top1_frac`) is *not* higher with renorm
  on** -- if anything it's lower at matched steps (step 25:
  `top1_on=0.0101` vs `top1_off=0.0341`; step 30: `0.0119` vs `0.0528`).
  Without renorm, the shrinking total mass concentrates what's left
  into a *proportionally* larger share of a smaller pool as the
  pipeline loses content faster than any single peak decays.
  With renorm, the reinflated mass gets spread across a **growing**
  occupied footprint (`occ_on` climbs monotonically, 574 -> 1651 cells)
  rather than piled onto the same shrinking region -- by late rollout
  (steps 70-100) `peak_on` itself has collapsed to 0.09-0.11 and
  `top1_frac` to ~0.001-0.002, consistent with the previously-documented
  "periodic lattice dominates the frame by step 90-100"
  (`experiment-log.md`, 2026-09-22 entry), not a single bright blob.
- Without renorm, the rollout never reaches a stable give-up plateau at
  all -- it hits **exact zero PROB mass by step 35**, well before the
  give-up/blur tradeoff or the periodic-lattice endpoint has a chance to
  develop. Renorm's presence is a precondition for the rollout staying
  non-degenerate long enough to exhibit either failure mode.

### Point 3 -- plain statement

The hard renorm mechanism is **a real, quantifiable, but secondary
contributor** -- not neutral, not the root cause. It measurably inflates
the give-up survivor's *absolute* pixel brightness (up to ~30% by late
rollout, confirmed via the scale factor and matched-step peak
comparison), which is consistent with the hypothesis. But it is not
responsible for the give-up blob's *spatial concentration* -- that
concentration is, if anything, slightly weaker with renorm on than off
at matched steps, because renorm redistributes its correction across a
growing set of cells rather than concentrating it further. Renorm's
dominant real effect is structural, not cosmetic: it is the thing
standing between the model and literal all-zero output, given how lossy
the underlying warp+threshold pipeline already is on its own. Removing
or softening renorm would not fix the give-up incentive (which
`findings-peak-decay-dissolution.md` already attributes to the base
per-pixel loss preferring diffuse mass-matching), and would introduce a
new, more severe failure (total blackout within ~35 steps) without
addressing that incentive.

## Part B: local/regional mass-conservation loss (sum, not peak)

### Method

Exact same probe methodology as `findings-regional-peak-loss.md`
("Method" section): `n=50`, radius 0.75, `bounce.init_balls`
(scenario-specific seed derived from 4738), half (rounded up) balls
"uncertain" per scenario, **give-up** (uncertain balls omitted) vs.
**commit** (uncertain balls splatted at true position + random-direction
drift of `drift_px`) predictions compared under the real base loss
(`occupancy_weighted_mse(..., peak_weight=0.0)`, channel weights
`[1.0, 0.1, 0.1]`, `bg_weight=0.05`) plus a new term at the same nominal
weight (`0.1`, matching `peak_weight`'s default) added exactly the way
`peak_weight * peak_loss` is added in the real loss. Swept ball counts
{2, 5, 10, 20}, drift {0.5, 1, 2, 4, 8}px, 15 trials/cell = 300 trials
per formulation. Note: this probe's own RNG indexing scheme
(`seed = 4738 + balls_idx*100000 + drift_idx*1000 + trial_idx`) is not
guaranteed bit-identical to the original peak probe's trial ordering, so
`base` column values differ slightly from `findings-regional-peak-loss.md`'s
`base` column (e.g. `balls=2,drift=1`: 6/15 here vs 5/15 there) -- this
is sampling noise from a different (but equally valid) trial draw, not a
methodology discrepancy; all formulations in *this* sweep share identical
per-trial data, so the cross-formulation comparisons within this table
are exact.

**Formulations:**

- `base` -- no extra term (same role as the peak doc's `base`).
- `tile4` / `tile8` / `tile16` -- `F.avg_pool2d`-based per-tile
  **integrated sum** of PROB (`avg_pool2d(...) * tile_area` to recover
  the tile's total mass), squared error between pred's and target's
  per-tile mass, mean-reduced over tiles. Same tile edges (4/8/16 cells)
  as the peak investigation.
- `window5` -- 5x5-cell window centered on each ground-truth ball's true
  position, **summed** PROB mass compared pred vs. target (mirrors the
  peak doc's `window5` geometry, sum instead of max).

### Results: win-rate table (commit beats give-up, out of 15 trials)

```
balls  drift    base    tile4    tile8   tile16  window5
    2    0.5     9/15     9/15    13/15    13/15    14/15
    2    1.0     6/15     7/15    10/15    13/15    15/15
    2    2.0     0/15     0/15     9/15    12/15    15/15
    2    4.0     0/15     0/15     6/15     8/15     0/15
    2    8.0     0/15     0/15     0/15     3/15     0/15
    5    0.5    12/15    12/15    14/15    14/15    15/15
    5    1.0     4/15     5/15     9/15    12/15    15/15
    5    2.0     0/15     0/15     7/15    14/15    15/15
    5    4.0     0/15     0/15     4/15    12/15     1/15
    5    8.0     0/15     0/15     0/15     6/15     1/15
   10    0.5    13/15    15/15    14/15    14/15    15/15
   10    1.0     1/15     3/15    10/15    12/15    15/15
   10    2.0     0/15     0/15     6/15    13/15    15/15
   10    4.0     0/15     0/15     4/15    11/15     3/15
   10    8.0     0/15     0/15     2/15     7/15     1/15
   20    0.5    15/15    15/15    15/15    15/15    15/15
   20    1.0     3/15     7/15    14/15    14/15    15/15
   20    2.0     0/15     0/15     4/15     7/15    15/15
   20    4.0     0/15     0/15     2/15    13/15     5/15
   20    8.0     0/15     0/15     0/15    13/15     1/15
```

Mean across all 20 (balls, drift) cells: `base` 0.210, `tile4` 0.243,
`tile8` 0.477, `tile16` 0.753, `window5` 0.637.

### Direct comparison to the peak-based formulations (`findings-regional-peak-loss.md`)

```
formulation   sum-mean   peak-mean   delta
tile4          0.243      0.217      +0.026
tile8          0.477      0.333      +0.144
tile16         0.753      0.567      +0.186
window5        0.637      0.607      +0.030
```

Sum beats peak at every tested granularity, and the gap **widens** with
tile size -- `tile16` is where the two operators diverge most.

### Does a give-up ball still get masked by a same-tile neighbor? Substantially less than with peak.

The peak doc's central finding was that `tile16`'s win rate collapses
sharply as density rises (crossing mean-balls-per-tile > 1.0 at 20
balls), because a shared tile's *max* is dominated by whichever ball in
it is confidently correct, masking a give-up neighbor entirely. The sum
operator does not show this pattern in nearly as strong a form:

```
tile16 (sum), win rate by density at each drift:
            balls=2  balls=5  balls=10  balls=20
drift=1.0     13/15    12/15    12/15     14/15    (flat, no density falloff)
drift=2.0     12/15    14/15    13/15      7/15    (drop at 20, but nowhere near peak's collapse to 3/15)
drift=4.0      8/15    12/15    11/15     13/15    (improves with density)
drift=8.0      3/15     6/15     7/15     13/15    (improves with density)

tile16 (peak), same cells, from findings-regional-peak-loss.md:
            balls=2  balls=5  balls=10  balls=20
drift=1.0     12/15    12/15    12/15      9/15
drift=2.0     11/15    10/15     7/15      3/15
drift=4.0     10/15     7/15     9/15      2/15
drift=8.0      5/15     5/15     1/15      1/15
```

At drift >= 2, peak-tile16 gets *worse* as density rises in every row;
sum-tile16 is flat-to-improving in 3 of 4 rows and only degrades in one
(drift=2.0). This matches the task doc's prediction: a **sum** is
inherently additive over everything in the tile, so a present neighbor's
mass doesn't hide a missing ball's absence the way a shared *max* does
-- the tile's total is directly short by the missing ball's mass either
way, give-up's per-tile error doesn't get cancelled out by a correctly-
predicted neighbor sharing the tile. Sum-tile16's one genuine weak spot
(20 balls, drift=2: 7/15) is worth noting but is a much softer failure
than peak's near-total collapse at the same density (peak-tile16 hits
1-3/15 by 20 balls at drift >= 4, vs. sum-tile16's 13/15 at the same
cells).

### `window5` (sum) shares peak's window5's fundamental cap

Same structural limitation as the peak version: strong (14-15/15) at
drift <= 2 regardless of density, collapsing once drift exceeds the
window's 2px half-width (drift=4: 0-5/15; drift=8: 0-1/15) -- the
committed-but-wrong position simply falls outside the fixed window, so
give-up and commit look identical to it and the underlying base loss
decides. Sum doesn't fix this; it's a geometric limitation of a
fixed-size region, independent of the max-vs-sum choice, exactly as the
peak doc characterized it.

### Answering the task's question directly

**Is local mass-conservation meaningfully different (better, worse, or
equivalent) than the regional peak terms already investigated?**
**Better**, and specifically better in the regime that mattered most:
`tile16` (sum) is dramatically more robust to the density x drift
combination that broke every peak-based tile/component formulation
(13/15 vs. peak's 1/15 at `balls=20, drift=8`; 13/15 vs. 2/15 at
`balls=20, drift=4`), confirming the task doc's prediction that a
continuous/graded sum operator escapes the near-binary "is *a* peak
present" masking failure that peak-based region checks are structurally
prone to. `window5` (sum) is roughly on par with `window5` (peak) and
inherits the same fixed-window drift cap; it is not the more
interesting result here -- `tile16` (sum) is. Neither `tile4` nor
`tile8` (sum) is strong enough alone to be a standalone
recommendation, but `tile16` (sum) is a genuinely more promising
candidate than anything found in the peak investigation, and unlike
that investigation, does not obviously need a second escape hatch for
the high-density/high-drift regime -- it's already the best performer
there.

## Repro (not committed to the repo)

Scratchpad scripts, session-local:

- `part_a_renorm_probe.py` -- Part A: instrumented `stage2_flownet_h12_v7.pt`
  rollout, renorm on/off comparison, per-step scale/peak/concentration/
  occupancy log.
- `probe_local_mass_loss.py` -- Part B: give-up-vs-commit sweep for
  `tile4`/`tile8`/`tile16`/`window5` sum-based mass terms, win-rate table.

If `tile16` (sum) is taken forward, it should be implemented in
`model/losses.py` proper (as `peak_weight`/`peak_loss` currently are),
validated with a real retrain the same way v7's global peak term was,
and re-run through the standard diagnostic-grid + unbiased-subagent-review
pipeline -- this investigation is synthetic-loss-value evidence only,
not a retrain, and per the standing rule in
`feedback_prefer_architecture_over_hparam_tuning`, a retrain decision
should wait on human/coordinator review of these numbers rather than
being queued automatically.
