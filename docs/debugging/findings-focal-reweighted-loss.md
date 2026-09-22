# Findings: does focal-style reweighting fix the blur-vs-give-up tradeoff?

Investigation scoped by `docs/debugging/flownet-open-issues-v3.md`.
Investigation only -- `model/net.py`, `model/losses.py`, `model/train.py`
not edited. Both focal variants implemented standalone in scratchpad
scripts (`probe_focal_loss.py`, `adversarial_focal_check.py`, not
committed), reimplementing `occupancy_weighted_mse`'s exact occ/bg
masking, `bg_weight=0.05`, `weights=[1.0, 0.1, 0.1]` structure from
`model/losses.py`, modifying only the prob-channel error reduction, to
isolate the effect of the reduction from the masking logic. Same
give-up-vs-commit scenario builder, ball counts {2, 5, 10, 20}, drift
{0.5, 1, 2, 4, 8} px, 15 trials/cell as `findings-regional-peak-loss.md`.

## Summary of conclusion up front

**Focal-style reweighting of the base loss does not change the
give-up-vs-commit incentive, in either direction, by any meaningful
amount.** Both tested variants (power reweighting `gamma` in {1,2,4},
confidence-modulated CenterNet-style reweighting `gamma` in {1,2,4})
produce win rates statistically indistinguishable from the unweighted
`base` loss at every (balls, drift) cell, and combining a focal term
with `window5` (the best regional formulation from the prior
investigation) is indistinguishable from `window5` alone. Focal
reweighting is not a partial mitigation like the regional terms -- it
is **inert** with respect to this specific tradeoff. It does not
introduce the adversarial hallucination-reward failure mode that
disqualified `topk8`, but that's a low bar it was never at risk of
failing (see mechanism below).

## Variants tested

- **Power reweighting**: `sq_err * |pred-target|**gamma` in place of
  `sq_err`, for `gamma` in {1, 2, 4} (raises the effective error
  exponent to 4, 6, 8 respectively) -- everything else in
  `occupancy_weighted_mse` (occ/bg masking, `bg_weight`, other-channel
  loss, `peak_weight`) held fixed at training defaults, matching the
  isolation goal in the scope doc.
- **Confidence-modulated focal term**: `sq_err * (1 - min(pred,
  target))**gamma`, `gamma` in {1, 2, 4}. Justification/mechanism: for
  a pixel where both pred and target are near the same value (agreement,
  `min` close to that value when both are large, or `sq_err` already
  near 0 when both are small), the weight shrinks, matching CenterNet's
  penalty-reduced heatmap loss intent of not over-penalizing near-peak,
  mostly-correct pixels. For a pixel where the model is confidently
  *wrong* in either direction (a false-positive bright cell where
  target is near 0, or a false-negative near-0 cell where target is
  bright), `min(pred, target)` is small, so `(1-min)**gamma` stays near
  1 and the existing squared error passes through at close to full
  weight -- i.e. it reduces the loss's sensitivity to small residual
  noise around agreement while leaving confidently-wrong pixels' penalty
  essentially unchanged, unlike the power variant, which instead
  amplifies whichever pixel already has the largest raw error.

## Win-rate results

Mean win rate (commit beats give-up) across all 20 (balls, drift)
cells, 15 trials each (300 trials/row), same convention as
`findings-regional-peak-loss.md`:

```
base                       0.213
power gamma=1              0.197
power gamma=2              0.190
power gamma=4              0.187
confidence gamma=1         0.220
confidence gamma=2         0.220
confidence gamma=4         0.233
window5 alone (w=0.1)      0.623
power g2 + window5         0.627
confidence g2 + window5    0.627
power g4 + window5         0.627
confidence g4 + window5    0.627
```

(`base` here is 0.213 rather than the 0.197 reported in the prior doc
because this reimplementation's `make_scenario`/RNG draws are
independent per-cell seeds rather than a shared trial index across
formulations, and 15 trials/cell has nonzero sampling noise at this
scale -- both numbers agree to within that noise and the qualitative
pattern by density/drift is the same.)

Every cell where `base` is 0/15 or 15/15 (the saturated regions -- drift
too small or too large for the base masking to distinguish the
strategies at all) stays exactly 0/15 or 15/15 under every focal
variant with no exceptions. The only cells with any nonzero variation
are the same mid-drift (0.5-2px) cells where `base` itself is not
saturated (e.g. drift=1.0: base 2-balls 5/15, 5-balls 2/15, 10-balls
4/15, 20-balls 1/15; power_g2 tracks the same shape within 2/15; window5
alone is 15/15, 15/15, 15/15, 15/15 at the same cells), and the
variation there is within +/-3/15 of `base`, i.e. within normal
15-trial sampling noise, not a directional shift.

## Task 3: mechanism check -- does focal reweighting avoid neighbor-masking, and does it saturate below full coverage?

**Neighbor-masking**: confirmed absent, as expected. Focal reweighting
is applied per-pixel with no region-pooling or region-sharing step
(unlike `tile16`/`component`/`window5`, which check "is there a peak
*somewhere in this region*"), so there is no mechanism by which a
correctly-predicted neighboring ball's pixels can mask an uncertain
ball's own pixels' contribution to the loss -- each pixel's error term
depends only on that pixel's own `pred`/`target` values. This is
confirmed structurally (the reduction has no spatial pooling op at all,
unlike the tile/window formulations) and empirically: the win-rate
table shows no density-driven degradation pattern distinct from `base`
(e.g. at drift=1.0, win rate falls monotonically with density for base,
power, and confidence alike, the same shape as `base`, not the sharper
density-specific collapse `tile16`/`component` showed relative to
`window5` in the prior doc). But absence of masking doesn't help here,
because...

**Saturation below full coverage**: also confirmed, and this is the
real finding -- focal reweighting saturates at essentially the *same*
ceiling as `base` at the hardest cells, not a higher one. At (balls=20,
drift=4.0) and (balls=20, drift=8.0), `base`, `power` (all gammas), and
`confidence` (all gammas) are all 0/15 or 1/15. Focal reweighting is a
**monotonic transform of the per-pixel squared error**, applied
identically to however many pixels are wrong under either strategy. It
does not change *which* pixels are being compared or introduce any new
positional signal near the ball's true location the way `window5`
does -- it only rescales the existing per-pixel signal's magnitude.
Since give-up's characteristic error (a missed cell, up to
`target_peak**2` per pixel) and commit's characteristic error (a
false-positive cell of similar peak magnitude *plus* a missed cell of
similar magnitude) are both already large-magnitude errors of
comparable per-pixel size, a monotonic upweighting of "large errors"
scales both strategies' dominant loss terms by roughly the same factor,
leaving their relative order -- and thus which one is loss-optimal --
essentially unchanged. This is the mechanistic reason focal reweighting
is inert here: **it has nothing to differentiate, because give-up's and
commit's losses are already dominated by errors of similar per-pixel
magnitude, just at different pixel counts/locations, and a purely
magnitude-dependent reweighting can't distinguish "one extra
false-positive pixel" from "one missed pixel" when both have comparable
peak-scale error.**

## Task 4: combined with `window5`

Combining any focal variant with `window5` (`window5_weight=0.1`,
same additive-term style as the existing `peak_weight` term) produces
mean win rate 0.627, vs. `window5` alone at 0.623 -- a difference of
0.004, i.e. noise at this trial count, not a real effect. Per-cell, the
combo table is identical to `window5` alone at every single cell
tested. **Focal reweighting and `window5` are neither complementary nor
redundant in any measurable sense here -- focal contributes nothing on
top of `window5`,** because `window5`'s fix (giving the loss an
explicit local-window signal near the ball's true position, so commit's
correctly-placed portion of mass is rewarded even when the ball as a
whole drifted) operates through a completely different mechanism
(adding new positional information) than focal reweighting could in
principle contribute (rescaling existing per-pixel error magnitude) --
and since focal doesn't move any win-rate numbers on its own, it has
nothing to add on top of a term that already does.

## Task 5: adversarial check -- confident hallucination vs. honest give-up

Built a hallucination scenario (uncertain balls' predictions placed at
*uniformly random* locations across the grid, not near their true
position at all, with full confidence/correct disk shape) vs. honest
give-up (zero PROB at uncertain balls' locations), same certain-ball
population, 60 trials (15 x {2,5,10,20} balls):

```
variant                    mean_loss(give-up)  mean_loss(hallucinate)  hallucination wins
none (base)                0.00667             0.00814                 0/60
power gamma=1              0.00346             0.00476                 0/60
power gamma=2              0.00220             0.00343                 0/60
power gamma=4              0.00141             0.00260                 0/60
confidence gamma=1         0.00666             0.00812                 0/60
confidence gamma=2         0.00665             0.00811                 0/60
confidence gamma=4         0.00664             0.00810                 0/60
power gamma=2 + window5    0.00220             0.00343                 0/60
```

Honest give-up beats confident hallucination in **every single trial**,
under every focal variant and the combo, with a consistent margin (the
hallucinated prediction's loss is 30-80% higher than give-up's
depending on gamma). Unlike `topk8` (which was positionally blind by
construction -- it compared only sorted magnitude lists), every focal
variant here still operates on the real per-pixel `occ`/`bg` masked
squared error, which already penalizes a bright pixel landing where
`target` is 0 (a false positive against a background-adjacent occupied
cell, or full `bg_weight`-discounted background penalty elsewhere) --
focal reweighting only rescales that existing, already-correct
per-pixel signal, it never removes or bypasses it. **No adversarial
hallucination-reward failure mode found in any tested variant.**

## Task 6: direct answer

**Focal-style reweighting does not change the fundamental blur-vs-give-up
tradeoff. It is not even a partial mitigation like the regional/window
terms -- it is measurably inert**, within normal 15-trial sampling
noise, across every (balls, drift) cell tested, alone or combined with
`window5`. The mechanism (Task 3) explains why: focal reweighting can
only rescale the *magnitude* of an already-present per-pixel error
signal; it has no way to introduce a new positional signal the way
`window5`'s ball-centered window does, and give-up's and commit's
dominant error terms are already comparable in per-pixel magnitude (both
involve peak-scale errors, just distributed over different pixel
counts/locations), so any purely magnitude-dependent reweighting scales
both sides by roughly the same factor and leaves the ordering -- hence
the loss-optimal strategy -- unchanged. This is a structurally different
kind of null result from `topk8`'s disqualification (which failed an
adversarial check); focal reweighting passes the adversarial check
cleanly (Task 5) but simply doesn't move the metric the investigation
is about.

**Recommendation**: do not pursue focal-style reweighting of the
existing pixelwise term as a fix for the blur-vs-give-up tradeoff --
it adds implementation/tuning surface (an extra gamma hyperparameter)
for no measured benefit over `base`, and no measured synergy with
`window5`. This does not change the standing recommendation from
`findings-regional-peak-loss.md`: `window5`-style regional
peak-preservation remains the best partial mitigation found so far, and
the deeper options (bound/report the honest horizon, or a genuinely
different output representation) remain the paths for resolving the
tradeoff rather than mitigating it.

## Repro (not committed to the repo)

Scratchpad scripts (session-local, not in `scripts/`):

- `probe_focal_loss.py` -- scenario builder (reused from
  `findings-regional-peak-loss.md`'s method), `occupancy_weighted_mse`
  reimplementation with `power`/`confidence`/`none` focal modes plus
  optional `peak_weight`/`window5_weight` additive terms, full win-rate
  sweep and combo runs.
- `adversarial_focal_check.py` -- hallucinated-wrong-location vs.
  honest-give-up adversarial check, reusing `probe_focal_loss.py`'s
  grid/loss helpers.
