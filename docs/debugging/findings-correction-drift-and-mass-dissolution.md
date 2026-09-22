# Findings: correction-head drift damping + mass drift/dissolution

Investigation of `docs/debugging/drift-investigation-brief.md`'s two
problems against `checkpoints/stage2_flownet_h12_v2.pt`
(`model/net.py`'s `BounceNextFrameModel`, post period-2/gridding fixes).
Investigation only — `model/net.py` was not edited; all counterfactuals
below run a local copy of `forward()`'s logic in scratch scripts.

## Part 1: the VX drift is an unbounded DC (spatial-mean) bias with no restoring force

### Diagnosis

Instrumented rollout (n=50, 150 balls, seed 4738, same repro as the
brief) confirms the reported trajectory and pins down the mechanism:
`correction[:,1]` (VX channel of `correction_head`'s output) is **not**
oscillating and **not** proportional to the current VX magnitude — its
per-step spatial mean stays in a fairly narrow band (~+0.11 to +0.17)
for steps 0-9 regardless of VX growing from 0.016 to >1.2 over the same
window, then flips to a sustained negative band (~-0.03 to -0.10) from
step ~12 onward as VX falls back down. Critically, `grid_sample`-warping
the VX field is close to mean-preserving (the warp redistributes VX
spatially but barely moves the frame-wide mean), so:

```
x1_mean(t+1) ~= warped_x1_mean(t) + correction[:,1]_mean(t)
```

i.e. `x[1].mean()` is, to good approximation, a **running sum of
correction[:,1]'s per-step spatial mean** — confirmed directly:
`cumsum(corr1_mean)` tracks `x[1].mean()` almost exactly step-for-step in
the instrumented rollout (e.g. step 8: cumsum=1.196 vs actual=1.225; step
12: cumsum=1.357 vs actual=1.433). `correction_head` has a nonzero VX bias
(`correction_head.bias[1] = 0.0275`, next largest after PROB's 0.120) but
the swing is much larger than the bias alone could produce, so this is a
learned, input-dependent, but persistently same-signed output, not just a
constant offset.

This is a distinct failure mode from the period-2 issue
(`findings-period2-oscillation.md`): that one was a per-step sign-flipping
instability (dominant Jacobian eigenvalue real and < -1). This one is a
**slow, one-directional accumulation** (an undamped integrator) — nothing
in `correction`'s definition (`tanh(correction_head(x)) * max_correction`)
penalizes or bounds its *cumulative* effect, only its *instantaneous*
per-cell magnitude. A flat magnitude cap (already applied) bounds how much
correction can be added on any one step, but says nothing about the sign
staying consistent for 12 steps in a row, which is exactly what produces
the horizontal-banding drift. (The eventual collapse back toward 0 by
step ~30 is not a stabilizing mechanism — it coincides with the model's
`horizon=12` training window per `findings-period2-oscillation.md`'s
same observation, i.e. the sign-flip at step ~12 looks like the model
extrapolating into unsupervised territory, not a designed damping term.)

### Damping mechanisms tested (counterfactual, scratch-only, not applied to `net.py`)

Same trained weights, same IC, only the correction term modified between
`warped` and the final output, per step:

```
mode            peak|vx_mean|   step of peak   final (step 30)
baseline        +1.4329         12              +0.0616
center          +0.1395         30 (monotonic)  +0.1395
ema_highpass    +1.0280         12              +0.0556   (decay=0.9)
state_decay     +1.3136         12              +0.0865   (0.98x/step on warped state)
```

- **`center`**: subtract `correction`'s per-channel *spatial* mean before
  adding it to `warped` (`correction - correction.mean(dim=(2,3),
  keepdim=True)`). This forces `correction` to be a pure redistribution
  term — it can still push individual pixels around, but it can no longer
  inject a net per-channel bias into the frame every step. **Peak |vx_mean|
  drops 10x (1.43 -> 0.14)**, and extending the rollout to 80 steps shows
  it plateaus at ~0.155-0.157 (not still climbing, not collapsing) — no
  more one-directional drift-then-collapse, no undamped integrator.
- `ema_highpass` (subtract a slow EMA of the per-channel mean, decay=0.9)
  only partially helps (~28% reduction) — decay=0.9 is too slow relative
  to how fast the bias accumulates per-step; a much faster EMA would
  converge toward `center` but adds a hyperparameter for no benefit over
  the exact centering above.
- `state_decay` (multiply the warped state by 0.98 before adding
  correction, i.e. decay applied to accumulated state rather than to
  correction itself) barely helps (1.43 -> 1.31) — the leak is too slow to
  counteract a bias of this size within 12 steps, and this would also
  quietly leak real mass/momentum out of legitimately-static regions.

### Recommendation (Part 1)

**Center `correction` per-channel per-step** (subtract its own spatial
mean, per channel, before adding to `warped`), i.e. in
`BounceNextFrameModel.forward`:

```python
correction = torch.tanh(self.correction_head(x)) * self.max_correction
correction = correction - correction.mean(dim=(2, 3), keepdim=True)
```

This is the "normalizing/centering correction per-channel per-step"
option named in the brief, and it's the cleanest of the three tested: no
new hyperparameter, directly targets the confirmed mechanism (an
unconstrained per-channel DC term with no restoring force), and the
counterfactual above is a clean causal ablation (same weights, same IC,
only this one line different) in the same style as the period-2
investigation. It does not fully zero out the drift (a physically real
scenario can have genuine net rightward motion, e.g. this IC's balls
launched with a rightward bias, so *some* nonzero VX mean is legitimate)
but removes the specific unconstrained-integrator pathology and keeps the
trajectory bounded indefinitely (verified to 80 steps).

Complementary, not a substitute: consider also re-training with a longer
rollout horizon once this is applied, for the same reason given in
`findings-period2-oscillation.md` — but the architectural fix should come
first.

## Part 2: bilinear grid_sample warping is a real, large, structural contributor to the dissolve/drift-apart symptom

### Test design

Per the brief's suggested clean test: a single ball simulated with real
`bounce.py` physics (ground-truth positions each step, `n=50`,
`dt=0.15`, `gravity=9.0`, `radius=0.75`, seed 4738, IC `x=15,y=25,
vx=1.2, vy=-0.6`). Three parallel renderings of the same trajectory:

- **GT**: re-splat a fresh sharp disk at the true position every step
  (`bounce.splat_all`) — the reference, no warping at all.
- **Pure-warp**: start from the step-0 splat and advect it forward using
  `F.grid_sample(mode="bilinear", padding_mode="border")` with the *true*
  per-step centroid displacement each step (ground-truth flow) — no
  model, no correction, isolates resampling diffusion alone.
- **Model**: the real `stage2_flownet_h12_v2.pt` rollout on the same IC,
  for a magnitude comparison.

### Result: pure bilinear warping with perfect flow already destroys the ball

```
step |  GT peak | warp peak | warp/GT peak | model peak
   0 |   0.6321 |    0.6321 |        1.000 |     0.6321
   2 |   0.4682 |    0.2610 |        0.557 |     0.1816
   4 |   0.2530 |    0.1725 |        0.682 |     0.0627
   6 |   0.2823 |    0.1322 |        0.468 |     0.0274
   8 |   0.4651 |    0.0943 |        0.203 |     0.0128
  10 |   0.5726 |    0.0743 |        0.130 |     0.0115
  12 |   0.5599 |    0.0645 |        0.115 |     0.0119
  16 |   0.3386 |    0.0028 |        0.008 |     0.0119
```

Peak-intensity retained relative to step 0: GT retains 91%/81%/71% at
steps 10/20/30 (the mild real decay there is genuine physics — a moving,
occasionally wall-clipped disk, not resampling). **Pure bilinear warping
with the exact true flow field and zero model error collapses the peak to
13% by step 10 and <1% by step 16** — this happens with no model in the
loop at all, purely from repeated bilinear resampling of a rigidly
translating disk. The real model's peak collapses even faster (1.8% by
step 10) — worse than pure-warp alone, so the model contributes
additional degradation on top of the structural warp diffusion, but the
warp-only baseline already accounts for the large majority of the
collapse in order of magnitude.

**Isolating the mechanism further** — swapping only the resampling mode
(`bilinear` -> `nearest`) in the same pure-warp test, same true flow, same
trajectory:

```
mode      peak, steps 0/4/8/12  (relative)
bilinear  0.632 / 0.173 / 0.094 / 0.064     (90% lost by step 12)
nearest   0.632 / 0.632 / 0.632 / 0.632     (0% lost by step 12)
```

`mode="nearest"` fully eliminates the collapse over this window (this
particular ball's sub-pixel velocity happens not to cross an integer
boundary until later) — a clean confirmation that the loss is coming
specifically from bilinear's neighbor-averaging, not from the flow field,
the border padding, or anything else in the pipeline. (Both modes
eventually go to zero later in this specific single-ball test once the
ball's trajectory interacts with wall-bounce dynamics that a single rigid
per-frame displacement can't represent — an artifact of this simplified
isolation test's rigid-shift assumption, not part of the diffusion
mechanism being measured.)

### Diagnosis

**Confirmed: (c) both.** The Part 1 VX drift is a specific, fixable bug
in `correction_head`'s missing damping term. Separately and additionally,
repeated bilinear `grid_sample` resampling is a real, structural,
architecture-independent source of numerical diffusion — demonstrated
here with zero model involvement, using the exact ground-truth flow field
for the actual physics being modeled. This matches the classic
semi-Lagrangian advection literature the brief cites: every bilinear
resample blends a cell with its neighbors, and this compounds
multiplicatively over N rollout steps regardless of how accurate the
predicted flow is. It plausibly explains why "balls drift apart and
dissolve" recurred across architecturally distinct attempts this session
(windowed-attention, first flow-warp version, this one) — any of them
that resample the state through interpolation every step inherits this,
independent of whatever else was wrong with each one specifically.

One caveat on interpretation: the model's total PROB-channel mass grew
enormously in this single-ball test (33x by step 10) even as peak
intensity collapsed — but the training distribution is 50-250 balls
(`model/train.py --min-balls 50 --max-balls 250` defaults), so a
single-ball IC is heavily out-of-distribution and this mass-explosion
number should not be read as representative of in-distribution behavior;
it's flagged here for completeness, not as further evidence for the
diffusion hypothesis (that evidence is the peak-intensity/nearest-vs-
bilinear comparison above, which does not depend on ball count).

### Recommendations (Part 2)

In order of how directly they attack the confirmed mechanism, with
tested trade-offs:

1. **Bicubic resampling instead of bilinear.** `F.grid_sample` supports
   `mode="bicubic"`. Better preserves local sharpness/curvature than
   bilinear (higher-order interpolation, less low-pass blur per resample)
   while still being differentiable and much less prone to the aliasing
   nearest-neighbor introduces under real (non-databaked) predicted flow
   fields with per-pixel-varying displacement — worth a repeat of this
   exact pure-warp test with `mode="bicubic"` before committing, since
   this doc did not test it (only bilinear vs. nearest, to cleanly bound
   the effect).
2. **Mass-conserving renormalization after each warp**: rescale the
   PROB channel so its frame-wide sum matches the pre-warp sum. This is
   cheap and directly counteracts the mass-explosion/mass-loss numbers
   seen above, but note it does *not* by itself fix peak-sharpness
   dissolution — renormalizing an already-diffused, spread-out blob back
   up in total mass does not refocus it into a sharp disk again. It's a
   reasonable complementary safeguard, not a fix for the diffusion
   mechanism itself.
3. **Nearest-neighbor** eliminates blur entirely (confirmed above) but
   trades it for spatial aliasing/jitter under a *predicted* (imperfect,
   spatially-varying, not-integer-aligned) flow field — the clean result
   here used a perfectly rigid, single global per-frame displacement,
   which is the best case for nearest; a real per-pixel predicted flow
   field would likely reintroduce jaggedness instead of smooth drift.
   Worth testing against the real model's predicted flow field (not just
   this idealized ground-truth-flow probe) before adopting.
4. **Explicit sharpening/counter-blur correction** or a training-time
   "peakiness" regularization (per the brief's literature note) is the
   most principled long-term direction if (1)-(3) aren't sufficient, but
   is the most invasive change and should be tried last, after the
   cheaper resampling-mode swap.

Per this session's standing rule, none of these should be treated as
"retrain and see" — (1) and (3) are one-line changes to `grid_sample`'s
`mode` argument that can and should be validated with the same
pure-warp-vs-model-vs-GT probe used here (no retraining needed to
measure the resampling-mode effect in isolation) before any retraining is
considered; only after confirming which resampling mode reduces the
structural diffusion should the model be fine-tuned/retrained on top of
that architectural change, exactly as was done for the gridding-artifact
fix.

## Relation between Part 1 and Part 2

Both are real and are not the same bug, but they interact exactly as the
brief speculated: with `correction_head` free to push a sustained
per-channel bias (Part 1), it's plausible the network partially learned
to counteract the very blur described in Part 2 during short-horizon
(`horizon=12`) training — pushing VX up for as long as gradient signal
rewarded it, with no mechanism to cap how long that push could persist
once rolled out past the training horizon. Fixing Part 1 (centering
correction) removes the undamped-integrator pathology but does not touch
the underlying resampling diffusion from Part 2 — that requires a
resampling-mode or mass-renormalization change independently. Both fixes
are recommended; neither substitutes for the other.

## Repro / instrumentation scripts (not committed to the repo)

Scripts used for this investigation currently live in the session
scratchpad, not in the repo (per the same convention as
`findings-period2-oscillation.md`):

- Instrumented rollout w/ per-step correction/warped mean breakdown
  (Part 1 diagnosis table)
- Damping-mechanism counterfactual harness (`center` / `ema_highpass` /
  `state_decay` modes, Part 1 table)
- Single-ball GT-vs-pure-warp-vs-model isolation probe, with a
  `bilinear`-vs-`nearest` resampling-mode ablation (Part 2)

If a follow-up wants these as permanent repo tools, they should be
cleaned up and added under `scripts/`, following the same pattern as
`scripts/investigate_gridding_artifact.py`.
