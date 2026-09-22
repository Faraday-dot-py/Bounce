# Findings: peak-intensity decay / dissolution to flat mush (v6)

Investigation of the item promoted to active at the end of
`docs/debugging/experiment-log.md`'s 2026-09-22 entries: with the
quilting artifact and VX/VY drift both fixed, `stage2_flownet_h12_v6.pt`'s
rollout still collapses from crisp discrete balls to a featureless
blurred blob by step ~4, and to a fully flat, static two-tone band with
zero ball structure by step ~7-13, staying there for the rest of a
30-step rollout. Investigation only -- `model/net.py`, `model/losses.py`,
`model/train.py` were not edited; all probes below run a local copy of
`forward()`'s logic and `occupancy_weighted_mse` in scratch scripts
against frozen `stage2_flownet_h12_v6.pt` weights. Repro: `n=50`,
150 balls, seed=4738, `model.dataset.make_scenario_uniform` +
`bounce.make_grid`/`bounce.splat_all`, radius=0.75, matching prior
probes this session.

## Summary of conclusion up front

This is **architectural / objective-level, not a resampling-mode
problem, and not fixable by longer training horizon.** Two mechanisms
compound:

1. `occupancy_weighted_mse` (`model/losses.py`) rewards a diffuse,
   mass-matched blob over a sharp-but-slightly-mispositioned ball --
   directly demonstrated on the loss formula itself, not inferred.
   Under any real positional uncertainty (many balls, collisions,
   chaotic bounces), this makes blur the loss-minimizing strategy, not
   a training artifact.
2. The model's *own real predicted flow field* is measurably divergent
   near occupied cells from step 0 onward (mean divergence +0.35,
   growing per-step), which mechanically spreads each ball's footprint
   over a growing area every autoregressive step -- and this happens
   under **any** `grid_sample` interpolation mode, including
   `nearest`, which the previous doc's idealized rigid-shift probe
   predicted would eliminate the collapse. It doesn't, because the
   flow field itself is doing the spreading, not neighbor-averaging
   interpolation.

Mechanism 2 is best understood as the architecture's learned
implementation of mechanism 1's incentive: since the loss actively
prefers spread-but-centered mass to sharp-but-uncertain mass, gradient
descent trained the flow head to diverge (spread) rather than commit
to a precise position. These are not two independent bugs like the
quilting/drift chain earlier this session -- they are one root cause
(the loss's blur preference) expressed through one visible symptom
(divergent learned flow).

## Part 1: quantifying the collapse -- warp-only vs full forward vs flow-divergence stats

Instrumented rollout (n=50, 150 balls, seed=4738, radius=0.75) against
`stage2_flownet_h12_v6.pt`, replicating `forward()` exactly plus
per-step flow-field divergence near occupied cells (`div = d(flow_x)/dx
+ d(flow_y)/dy`, positive = locally expanding/divergent, negative =
locally converging):

```
step  peak    mass    occ_frac(>0.05)  |flow|_occ  div_mean_occ  div_min_occ
   0  0.8104  67.181  0.0900           0.5031      0.3464        -0.4543
   1  0.5316  67.181  0.1288           0.5665      0.2399        -0.4923
   2  0.3881  67.181  0.1528           0.6196      0.1832        -0.6485
   3  0.2510  67.181  0.1960           0.6683      0.1238        -0.6104
   4  0.1472  67.181  0.2528           0.7531      0.0854        -0.7031
   5  0.0940  67.181  0.2432           0.8105      0.0729        -0.9887
   6  0.0703  67.181  0.1668           0.8607      0.1060        -0.8775
   7  0.0543  67.181  0.0056           1.1867      0.4106        -0.3238
   8  0.0437  67.181  0.0000           --          --            --
  ...  (flat: peak oscillates 0.036-0.040, occ_frac==0.0 for the rest of the rollout to step 30)
```

Total PROB mass is exactly constant (67.181, the renorm working as
designed) at every step -- so the collapse is **not** mass loss, it is
mass spreading over an ever-growing area (`occ_frac` climbs from 0.09
to 0.25 through step 4-5 before the occupancy threshold itself
collapses the mask to 0 once everything is below 0.05). Flow magnitude
near occupied cells grows monotonically (0.50 -> 1.19) as does mean
divergence's *magnitude* both positive and negative -- the flow field
is not just larger, it is locally both expanding and contracting more
aggressively step over step, consistent with a network racing to
redistribute mass over the frame as fast as it can.

**Warp-only vs full-forward peak retention** (same real predicted flow
each step; "warp-only" = `grid_sample` with no correction, threshold,
or renorm applied after):

```
step  warp_only_peak  full_forward_peak  ratio (full/warp)
   1  0.7251          0.5316             0.733
   2  0.5370          0.3881             0.723
   3  0.3821          0.2510             0.657
   4  0.2518          0.1472             0.584
   5  0.1504          0.0940             0.625
   6  0.1054          0.0703             0.667
   7  0.0748          0.0543             0.726
   8  0.0536          0.0437             0.815
   9+ ratio -> ~1.0 (both sides pinned near the flat 0.036-0.040 floor)
```

Correction/threshold/renorm are **not neutral** during the active
collapse (steps 1-8): they consistently reduce peak further below what
the warp alone already produced (ratio 0.58-0.82), i.e. they make the
decay somewhat worse, not better, during exactly the window where the
ball structure is still salvageable. Once `occ_frac` hits 0 (step 8
onward) the ratio converges to ~1 because both paths are already at
the same near-uniform floor -- correction/renorm have nothing left to
act on. So: the warp (driven by divergent flow) is the dominant driver
of the early collapse, and correction/threshold/renorm are a smaller,
compounding, non-fixing second-order contributor, not a stabilizer.

## Part 2: the loss does not clearly reward peak sharpness -- it demonstrably prefers diffuse blur

Directly probed `occupancy_weighted_mse` on synthetic single-ball
targets/predictions (bypassing the model and dataset entirely, so this
is a property of the loss formula, not of training data or the flow
network):

**Variant 1** -- same total mass as the target's sharp disk (peak
0.6321, matching a single `bounce.splat_ball` disk), but spread over
increasing radius (still centered correctly):

```
radius  pred_peak  pred_mass  loss
 0.75   0.6321     0.6321     0.000000   (exact match)
 1.50   0.2467     0.6321     0.001190
 3.00   0.0674     0.6321     0.002539
 6.00   0.0168     0.6321     0.003008
10.00   0.0060     0.6321     0.003113
```

Loss grows with spread, as expected -- so the loss is not *blind* to
sharpness. But the growth saturates quickly and stays small in
absolute terms (max ~0.0031 even at radius 10, a totally featureless
blob covering most of a 50x50 grid).

**Variant 3** -- correct peak, correct radius, but shifted off-center:

```
shift_px  pred_peak  overlap_frac  loss
0.0       0.6321     1.000         0.000000
1.0       0.6321     0.000         0.003331
2.0-8.0   0.6321     0.000         0.003331   (flat -- once there is zero pixel overlap, loss saturates)
```

**Variant 4 -- the direct comparison that matters**: a
diffuse-but-mass-matched blob (radius 6, peak 0.0168, essentially
featureless) scores a **lower loss (0.003008)** than a perfectly sharp
disk that is merely shifted by 2 pixels (peak 0.6321, loss 0.003331).
**The loss actively prefers a flat, structureless blob to a sharp ball
that is even slightly mispositioned**, as soon as position uncertainty
exceeds about one ball radius. This is not a subtle gradient-sensitivity
argument -- it is a direct, exact evaluation of the loss on two concrete
candidate predictions.

Gradient check confirms the mechanism operates as expected (no
surprises): `d(loss)/d(pred)` at `pred=target` is exactly 0 (sanity),
and at `pred=0.5*target` gives a uniform negative gradient pushing
pred up everywhere in the occupied region, with no extra pressure
toward sharpening the peak specifically vs. filling in the whole
occupied footprint uniformly -- the loss's only signal is
pixel-wise squared error against the current target's occupied mask
(`target_occ | source_occ`, from `model/losses.py`), which has no
mechanism to penalize spread once mass is inside the (binary) occupied
region and heavily discounts (via `bg_weight=0.05`) mass that spills
outside it.

**Conclusion for task 2**: given any real positional uncertainty about
where a ball will be next (which is the norm past the first couple of
autoregressive steps, when many balls are interacting/colliding), a
network trained under this exact loss minimizes expected error by
predicting a diffuse, roughly-correctly-centered blob rather than
committing to a precise sharp position it might get wrong. This is the
textbook "regression-to-the-blur-under-uncertainty" pathology of
pixelwise MSE losses (the same phenomenon documented for direct pixel
regression in this session's earlier windowed-attention post-mortem,
`docs/debugging/experiment-log.md`'s 2026-09-21 entry) -- and
occupancy-weighting does not fix it, because it only privileges the
*binary* occupied/not indicator, not peak magnitude within that region.

## Part 3: nearest-neighbor resampling under the real predicted flow does not help, and does add aliasing

Frozen-weight counterfactual, same rollout, only `grid_sample`'s `mode`
argument swapped (`bilinear` / `bicubic` / `nearest`), same real
per-pixel predicted flow field each step (not an idealized rigid
shift):

```
mode      step0   step2   step4   step6   step8   step12  step20  step30
bilinear  0.8104  0.3534  0.1033  0.0510  0.0378  0.0355  0.0358  0.0357
bicubic   0.8104  0.3881  0.1472  0.0703  0.0437  0.0383  0.0376  0.0360
nearest   0.8104  0.5379  0.2351  0.0805  0.0465  0.0378  0.0377  0.0357
```

`occ_frac(>0.05)` collapses to exactly 0.0 by step 6-8 under **all
three** modes. `nearest` retains peak slightly longer through steps
2-6 (as expected -- no interpolation blur) but converges to the same
flat floor (~0.036-0.038) by step 8-12 as bilinear and bicubic. This
directly falsifies the prior finding doc's speculation (Part 2,
Recommendation 3) that nearest-neighbor "fully eliminates the
collapse" would generalize to a real predicted flow field -- it only
did so in the idealized single-rigid-shift ground-truth-flow probe.
Under the model's real, spatially-varying, divergent flow, the
resampling mode is a second-order effect; the flow field's own
divergence dominates.

A fresh unbiased subagent review of a bicubic-vs-nearest side-by-side
diagnostic grid (steps 0,1,2,4,6,8,12,16,20,30) confirms this
quantitatively and qualitatively: both rows follow the identical
trajectory (speckle -> clumping -> flat two-region bright/dark split by
step ~12 -> renewed boundary roughness by step 30), but nearest
"consistently shows blockier, jagged, stair-step edges and more
discrete/checkerboard-like patchiness" at every step from 4 onward,
with no compensating benefit -- exactly the aliasing risk the original
finding doc flagged as untested, now confirmed real with zero upside
under a real (non-idealized) flow field. **Recommendation: do not
switch to `nearest`** -- it trades identical collapse timing for a
strictly worse (jagged/aliased) failure mode.

## Part 4: horizon=12 is not the lever -- the collapse is already complete well inside the trained window

Two lines of evidence, neither requiring "just retrain longer":

1. **The observed collapse is entirely within the training horizon,
   not past it.** `stage2_flownet_h12_v6.pt` was trained with
   `horizon=12` (`model/train.py`'s `rollout_loss` explicitly computes
   and backpropagates loss at every step `k=1..horizon`, so the model
   *is* directly supervised at every one of steps 1-8 where the
   collapse to `occ_frac==0` happens). If "the loss never sees past
   the horizon" were the primary mechanism, degradation would be
   expected to start only after step 12, not to be essentially
   complete (flat, zero discrete structure) by step 7-8, four steps
   *inside* the supervised window. A longer horizon extends supervision
   further into the rollout, but every step where the collapse actually
   happens is already supervised at `horizon=12` -- so lengthening the
   horizon has no obvious mechanism to fix a failure that occurs before
   the existing horizon even ends.
2. **The loss-sensitivity result in Part 2 is horizon-independent.**
   Variant 4's finding (diffuse blur scores lower loss than a
   slightly-mispositioned sharp prediction) holds at every individual
   step `k` in isolation -- it is a property of `occupancy_weighted_mse`
   evaluated on one frame pair, not of how many steps are summed
   together. A longer horizon would mean more steps get this same
   per-step incentive to blur, not a different incentive.

**Directional local retrain** (horizon=4 vs horizon=12, both reduced
scale -- 60 samples, 3 epochs, otherwise default hyperparameters,
~4.6 minutes total CPU wall time): neither undertrained model collapses
to `occ_frac==0` at all within 20 steps (unlike the fully-trained v6),
and the two are statistically indistinguishable from each other:

```
horizon=4:  occ_frac(>0.05) per step (1..20): 0.163, 0.233, 0.249, ..., plateauing ~0.21-0.26
horizon=12: occ_frac(>0.05) per step (1..20): 0.098, 0.205, 0.238, ..., plateauing ~0.20-0.24
```

This test is confounded by both models being heavily undertrained (3
epochs vs. the real checkpoints' 80), so it cannot show where a
*fully-trained* model's collapse-onset step would land for each
horizon -- but it does show that at matched (very early) training
progress, horizon=4 and horizon=12 diffuse the occupancy mask at
statistically the same rate, with no sign that the shorter horizon
produces earlier or worse diffusion, or that the longer horizon
suppresses it. Combined with the two logical arguments above, this is
consistent with -- and does not contradict -- horizon length not being
the primary lever. **Retraining with a longer horizon is not
recommended as a primary fix**, per this session's standing rule
against defaulting to "train longer" without evidence.

## Relation between the mechanisms / what to actually fix

This is **architectural at the objective level**, compounded by one
second-order effect in the resampling path:

- **Root cause**: `occupancy_weighted_mse`'s pixelwise squared-error
  formulation, even with occupancy weighting, rewards diffuse
  mass-matched blur over confidently-sharp-but-uncertain predictions
  (Part 2). This is the same class of MSE-blur-under-uncertainty
  pathology already diagnosed for the earlier windowed-attention direct
  pixel regression architecture -- switching to a flow-warp
  architecture changed *how* the network implements blur (via divergent
  flow instead of via smeared direct-regression) but did not change
  *that* the objective rewards it.
- **Mechanism**: the flow head learns locally divergent flow near
  occupied cells (confirmed, Part 1) to spread mass into the diffuse
  shape the loss prefers; this happens regardless of `grid_sample`'s
  interpolation mode (Part 3), so the earlier "bicubic resampling
  diffusion" finding (`findings-correction-drift-and-mass-dissolution.md`
  Part 2) was real but was measuring a symptom of the same underlying
  incentive under an idealized rigid-flow probe, not an independent
  interpolation-only artifact -- under the model's own real flow, the
  interpolation-mode choice barely matters.
- **Second-order compounding factor**: correction/threshold/renorm make
  the early-rollout peak decay measurably worse, not better, while
  ball structure is still salvageable (Part 1's warp-only vs
  full-forward ratio, 0.58-0.82 through step 8) -- not the dominant
  driver, but not neutral either.
- **Ruled out**: training horizon length as the primary lever (Part 4)
  and resampling interpolation mode as a fix (Part 3).

### Recommended direction (not yet implemented, `model/net.py` untouched per constraints)

Since the root cause is objective-level, the fix has to change what the
loss rewards, not just the resampling plumbing:

1. **Add an explicit sharpness/peakiness term** to the loss (e.g. a
   penalty on `occupied-region entropy` or `-max(pred_occ)` per ball
   footprint, or a higher-order penalty than pixelwise MSE such as a
   focal-style weighting that penalizes low-confidence-but-present
   predictions more than an outright miss) -- this is the "training-time
   sharpness/peakiness regularizer" option flagged as untested in the
   prior finding doc, and this investigation's Part 2 result is direct
   evidence it is necessary, not just a nice-to-have: the existing loss
   provably prefers blur once position uncertainty exceeds about one
   ball radius, which is the normal multi-ball regime past the first few
   steps.
2. Any such change should be validated the same way the quilting/drift
   fixes were: frozen-weight loss-value counterfactual first (as done
   in Part 2 here), then a real retrain, then the standard
   diagnostic-grid + unbiased-subagent-review + per-step-stats pipeline.
3. Do not adopt `mode="nearest"` (Part 3, no benefit, real cost).
4. Horizon length is not ruled out as *a* contributing factor entirely,
   but the evidence here (Part 4) does not support it as the primary
   fix, and should not be the first thing tried.

## Repro / instrumentation scripts (not committed to the repo)

Scripts used for this investigation currently live in the session
scratchpad, not in the repo (same convention as prior findings docs):

- Instrumented v6 rollout w/ per-step warp-only vs full-forward peak
  comparison and flow-divergence stats (Part 1)
- Synthetic loss-value probe against `occupancy_weighted_mse` directly,
  bypassing model/dataset (Part 2)
- `grid_sample` mode counterfactual (bilinear/bicubic/nearest) under
  the real predicted flow field, plus a rendered side-by-side diagnostic
  grid for unbiased visual review (Part 3)
- Reduced-scale local horizon=4 vs horizon=12 training comparison (Part 4)

If a follow-up wants these as permanent repo tools, they should be
cleaned up and added under `scripts/`, following the same pattern as
`scripts/investigate_gridding_artifact.py`.

## Part 5 (2026-09-22, post-retrain): the peak term fixes the metric but exposes a deeper, harder tradeoff -- not a clean win

The peak-magnitude term recommended above was implemented
(`model/losses.py`, `peak_weight=0.1` default) and retrained end-to-end
as `checkpoints/stage2_flownet_h12_v7.pt` (job 2829, otherwise identical
hyperparameters to v6). Verified via the standard pipeline:

- Step-1 MSE-vs-copy-baseline: 0.70x (v6 was 0.71x) -- no regression.
- **The specific collapse metric this fix targeted is fixed**: `max0`
  (peak PROB intensity) now stays in the 0.54-0.85 range at every step
  0-30 (was decaying to a flat 0.036-0.040 floor by step 8 in v6).
  `nz_frac` stabilizes at a moderate 0.3-0.58 (was saturating to
  0.75-0.84 in v6). VX/VY drift remains exactly fixed (unaffected by
  this change, as expected). No quilting/blockiness regression
  (block-variance metric 0.07-0.7, comparable to v6's early steps, far
  below v6's late-step 15-35 peak).
- **But an unbiased subagent video review found a new, different failure
  mode, not obviously better**: the model still diverges from ground
  truth starting ~step 3-5, but instead of v6's uniform blur-everywhere
  collapse, v7 **concentrates into a smeared, overlapping blob in one
  screen region (bottom-left in the tested rollout) while the rest of
  the frame goes to solid black**, abandoning most of the other balls
  entirely rather than blurring them. A new **faint periodic diagonal
  ripple/moire pattern** appears in a different region from step ~23
  onward (low intensity, not present in v6 or ground truth).

### Root cause of the new failure mode: identified directly, not merely observed

A synthetic single-step probe (two-ball scenario, one ball already
mispositioned from a hypothetical prior autoregressive step) shows the
mechanism is not really about the peak term specifically -- it's an
incentive already present in the base `occupancy_weighted_mse`:

```
scenario                                       base_loss   total(+peak, w=0.1)
give up on ball2 (predict near-zero there)     0.00313     0.00313
keep guessing near ball2's last position       0.00328     0.00328
```

**Once a ball's predicted position has already drifted from its true
position (the normal state past the first few autoregressive steps,
given chaotic multi-ball collisions), committing to *any* specific
nearby position costs more loss than giving up entirely** -- a
confident-but-wrong prediction incurs a double penalty (a false-positive
bright cell at the wrong location *and* a missed true-position cell),
while predicting near-zero only incurs the missed-cell penalty once.
This was always true of the base loss; it just wasn't visible before
because the (also loss-optimal, per Part 2) alternative of blurring
everywhere let the network hedge across many plausible positions at
once, which is cheaper than either committing or giving up. **The peak
term successfully closed off the "blur everywhere" hedge (that was its
job, and it worked), which meant the network's next-cheapest option
under the same base loss was "commit confidently to what's still easy,
give up on what's uncertain"** -- not a bug introduced by the peak
term, but a second, previously-hidden failure mode of the same
underlying objective, now exposed once the first one was fixed. This
is the textbook two-sided failure mode of deterministic point-estimate
regression under genuine multimodal uncertainty (blur-as-hedge vs.
give-up-as-hedge): closing off one side does not eliminate the
uncertainty, it just changes which cheap-but-wrong strategy gradient
descent finds instead.

### Assessment: real progress, not a clean fix -- do not treat as resolved

This is **not a fixable-by-another-loss-term problem in the same style
as tonight's other fixes** (quilting, VX/VY drift) -- those were
genuine implementation bugs with a clean root cause and a fix that
left no comparably-sized side effect. This one is a fundamental
property of the modeling approach (dense per-cell regression under a
pointwise loss, predicting a single deterministic future for an
inherently chaotic multi-ball system) that reappears in a different
shape no matter which side of the blur/give-up tradeoff is
suppressed. Concretely recommended options for whoever picks this up
next, roughly in order of effort:

1. **Regional/local peak-preservation instead of a single global peak**
   (e.g. `F.max_pool2d` over tiles, matching each tile's local max
   rather than one frame-wide max) -- tested only in an isolated
   synthetic single-step probe here (not retrained), and did not show a
   clearly larger penalty for the give-up strategy than the global
   version did in that same probe, so it is not a confirmed fix, only
   an untested next thing to try.
2. **Accept and bound the honest horizon**: rather than continuing to
   chase 30-step rollout quality, characterize and report the actual
   step count at which multi-ball position uncertainty becomes
   irreducible for point-estimate regression (this session's data
   suggests single-digit steps), and treat longer rollouts as
   out-of-scope for this architecture rather than a bug to fix.
3. **A genuinely different output representation** (e.g. per-ball
   tracked state instead of a dense occupancy grid, or a probabilistic/
   multi-hypothesis output such as a mixture density or ensemble) would
   remove the forced choice between blur and give-up, but is a
   substantially larger redesign than anything else done this session
   and should be a deliberate decision, not something to start
   unprompted overnight.

**Recommendation for now**: keep `stage2_flownet_h12_v6.pt` as the
current default (it has every artifact/drift bug from this session's
earlier chain fixed, with a well-understood, honestly-reported residual
blur-collapse limitation). Treat `stage2_flownet_h12_v7.pt` as a
documented experimental variant with a different, not clearly better,
residual limitation -- useful primarily as evidence for the diagnosis
above, not as a drop-in upgrade. Both checkpoints and this finding
should be reviewed by a human before choosing a direction, since the
real options above (especially #3) are design decisions, not bug
fixes.
