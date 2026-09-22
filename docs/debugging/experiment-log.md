# Bounce Stage-2 experiment log

Append-only journal of the Stage-2 next-frame model investigation. Each
entry is a short summary with the commit that has the full numbers/
reasoning — read the commit (`git show <hash>`) for detail, don't
duplicate it here. Newest entries at the bottom.

Use `scripts/eval_step1_baseline.py` and `scripts/render_diagnostic_grid.py`
+ `docs/debugging/frame-artifact-review-prompt.md` as the standard
verification pipeline before claiming any fix worked — aggregate loss/MSE
stats have repeatedly looked fine while the rollout was actually broken
(see 2026-09-21 entries below). Always check actual per-frame, auto-scaled
visual output.

---

## 2026-09-21 — checkerboard artifact root-caused and (partially) fixed

Windowed-attention (Swin-style) Stage-2 model's autoregressive rollout
developed a periodic checkerboard/lattice pattern that a fixed-scale
video made look like "fade to black" (the checkerboard's contrast was
hidden by fixed vmin=0/vmax=1 display). Root-caused via 3 parallel
subagents to: static window-partition bias in `RelativePositionBias`,
a padding seam (25 patches not divisible by window_size=8), loss
dilution of background error by grid size, undertrained horizon, and
all-or-nothing (not per-step) scheduled sampling.

Fixed: randomized window-partition offset, per-region loss
normalization, per-step scheduled sampling, gradient clipping.
Retrained at horizon=12 (job 2819). Result: checkerboard softened/
delayed but not eliminated — independent subagent read confirmed grid
texture already present by rollout step 1-2, fully dominant by step
16-20, just via a different visual path (soft blob before lattice
instead of immediate lattice).

Commit: `da79c43`

## 2026-09-21 — 200-epoch retrain does not fix it; epochs aren't the lever

Retrained the fixed architecture for 200 epochs instead of 80 (job
2821), hypothesizing undertraining. Step-1 MSE-vs-copy-baseline check
(`scripts/eval_step1_baseline.py`, 10 seeds) showed 200ep was *worse*
(1.18x) than 80ep (1.04x), and both were worse than a horizon=3/200ep
run from earlier (0.90x, the only config that ever beat trivial
"copy the last frame forward"). Loss curve was flat (11.99 -> 11.95).
Diagnostic grid confirmed the checkerboard/banding artifact was
unaffected by epoch count.

Conclusion: horizon length dominates step-1 accuracy, not epoch count
— averaging the loss over 12 rollout steps dilutes the single-step
gradient relative to horizon=3.

Commit: `c64adc2`

## 2026-09-21 — architecture rewrite: windowed-attention -> dilated conv + flow-warp

Convergent evidence across 4 checkpoints (different loss formulas,
horizons, epoch counts) that the checkerboard is structural to the
windowed-attention architecture itself (window partitioning + a shared,
position-invariant relative-position bias gives the network a periodic
basis to compound into under autoregressive rollout), not a
training-schedule problem. Separately, step-1 (non-compounding)
prediction was consistently blurry — occupied cells never reached true
splat magnitude despite the physics being deterministic (no
uncertainty to blur into) — a signature of direct pixel regression
under MSE rather than motion representation.

Replaced `model/net.py`: windowed self-attention -> dilated residual
conv stack (translation-equivariant everywhere, ~33px receptive field
via dilations 1-2-4-8-4-2-1, no downsampling so grid/ball-size
generalization is preserved) predicting a per-cell flow field, warping
`g_t` via `grid_sample` (advection), plus a small correction head.
Both heads zero-init so the model starts as an exact identity (`out ==
g_t`) — matching that "copy forward" was already a competitive
baseline. Deleted the now-dead windowed-attention modules and their
tests; rewrote `tests/test_net.py` for the new architecture. Old
checkpoints (`stage1.pt`, `stage2*.pt`) are incompatible.

First training run of the new architecture: job 2822
(`checkpoints/stage2_flownet_h12.pt`, horizon=12, 80 epochs). Not yet
verified as of this entry — see next entry or `git log` for the
verification result.

Commit: `f0e709b`

## 2026-09-21 — flownet verified: real single-step win, two new long-horizon bugs

`stage2_flownet_h12` (job 2822) step-1 MSE vs. copy-baseline: **0.80x**
— best of any checkpoint tried (previous best 0.90x, most were
1.0-1.35x, worse than doing nothing). Sparse ball structure now
visibly survives to rollout step ~5 (vs. dissolving to noise by step 1
under the old architecture).

But two distinct problems remain at longer horizon, both found via the
standard diagnostic-grid + unbiased-subagent pipeline plus a manual
per-step stats dump:

1. **Grid/lattice pattern re-emerges by step ~16-30**, plus edge/corner
   brightening from step 8 on. Leading hypothesis: the dilation
   schedule `(1, 2, 4, 8, 4, 2, 1)` in `model/net.py` is a textbook
   setup for the "gridding artifact" in dilated CNNs (shared common
   factor 2 across all dilations leaves periodic receptive-field gaps)
   — not yet verified.
2. **Period-2 brightness oscillation** starting ~step 14, confirmed via
   per-step mean/max stats (alternates ~5-15x in magnitude every other
   frame). Signature of an unstable/oscillating autoregressive fixed
   point, likely undamped gain in the flow+correction path. Distinct
   from problem 1, not caught by the diagnostic grid's sparse step
   sampling (it happened to sample mostly one phase).

Both under active investigation — see
`docs/debugging/flownet-open-issues.md` for the full brief handed to
investigating subagents, and `docs/debugging/findings-gridding-artifact.md`
/ `docs/debugging/findings-period2-oscillation.md` for their conclusions
once written.

## 2026-09-21 — both problems root-caused (both architectural, dilation schedule cleared) and fixed

Two parallel subagent investigations returned:

- **Period-2 oscillation**: confirmed architectural — `correction_head`
  was the only head with no bounding (unlike `flow_head`'s
  `tanh*max_flow`). Jacobian probe at a late-rollout operating point
  measured dominant eigenvalue ≈ -1.15 to -1.2 (magnitude >1, negative
  sign = textbook period-doubling instability). Ablation (zero or halve
  `correction`) eliminated the oscillation outright. Findings:
  `docs/debugging/findings-period2-oscillation.md`.
- **Gridding/lattice artifact**: leading hypothesis (dilation-schedule
  gridding, shared factor 2 across `(1,2,4,8,4,2,1)`) was **falsified**
  — theoretical receptive-field coverage has zero holes and 0.17% parity
  bias, and the rollout's actual FFT period (13-40px) is far coarser
  than the 2/4/8px scale a dilation checkerboard would produce. Real
  cause: the conv stack's implicit zero-padding gives the border ring
  anomalous flow (2.2x interior magnitude), and
  `grid_sample(padding_mode="zeros")` turns that into a guaranteed 100%
  out-of-bounds perimeter every single step (exact match: OOB fraction
  = 196/2500 perimeter pixels), forcing `correction_head` to
  reconstruct the whole border from nothing each step. Confirmed
  causally: swapping to `padding_mode="border"` (no retraining)
  measurably softens the sharp lattice-line texture. Findings:
  `docs/debugging/findings-gridding-artifact.md`.

**Fix applied** (`model/net.py`, commit `b05e1a6`): `correction` is now
`tanh(...) * max_correction` (default 0.2, mirroring the existing
`max_flow` pattern); `grid_sample` uses `padding_mode="border"`; the
conv stack (`ResidualConvBlock` + stem + flow/correction heads) uses
`padding_mode="replicate"` instead of implicit zero-padding.
`DILATIONS` left unchanged — the evidence didn't support changing it.
28/28 tests pass (added a bounded-correction test, updated the
correction-delta test for the new tanh saturation).

Retraining as `stage2_flownet_h12_v2.pt` (job 2823) to let the model
adapt to the new padding/bounding semantics — not a hyperparameter
sweep, a required adaptation step since the forward pass itself
changed. Old checkpoint (`stage2_flownet_h12.pt`) kept for
before/after comparison. Verification pending — run the standard
pipeline (`scripts/eval_step1_baseline.py`,
`scripts/render_diagnostic_grid.py` + unbiased subagent review,
`scripts/investigate_gridding_artifact.py` for the OOB/FFT metrics,
and a full per-step stats dump to catch any residual oscillation)
once job 2823 completes.

## 2026-09-21 — v2 verified: edge/gridding + period-2 fixed, two new problems (drift, dissolution)

`stage2_flownet_h12_v2` (job 2823) verified via the standard pipeline:

- Edge/gridding fix worked: border-flow ratio 0.72x interior (was
  2.20x), OOB fraction 0.038 (was 0.078), edge/interior PROB means now
  track closely at every step.
- Period-2 oscillation is gone (no more 5-15x frame-to-frame
  alternation).
- **Step-1 regressed to 1.08x** MSE-vs-copy-baseline (worse than doing
  nothing), from 0.78-0.80x pre-fix — `max_correction=0.2` likely too
  restrictive.
- **New VX-channel drift**: not oscillating, but growing monotonically
  from 0.016 to 1.43 by step 12, then collapsing back toward 0 by step
  30 — produces visible horizontal banding, confirmed by an unbiased
  subagent image review.

Video sent, `scripts/investigate_gridding_artifact.py`'s stale
`instrumented_forward` (still called the raw unbounded
`correction_head` and hardcoded `padding_mode="zeros"`) fixed to match
the real post-fix forward pass (commit `5ea4d2e`).

Comparison video convention established: rollout/comparison `.mp4`s
now go in `videos/` (repo root), not left only in `/tmp` or the repo
root directly (`scripts/render_rollout_video.py` updated, commit
`7fa684f`).

## 2026-09-21 — VX drift + mass dissolution root-caused and fixed

Dispatched a subagent (`docs/debugging/drift-investigation-brief.md`)
to dig into the VX drift and a bigger question the user raised
directly: "we need a way to keep the balls together, they seem to
drift apart no matter what we do" — true across every architecture
tried this session (windowed-attention, first flow-warp, this one).
Findings: `docs/debugging/findings-correction-drift-and-mass-dissolution.md`.

1. **VX drift = undamped integrator.** `correction[:,1]`'s per-step
   *spatial mean* stayed same-signed (~+0.11 to +0.17) for steps 0-9
   regardless of VX's growing magnitude; since bilinear/bicubic warping
   is close to mean-preserving, `x[1].mean()` is essentially a running
   sum of `correction`'s per-step mean (`cumsum` tracked the real
   trajectory almost exactly). The magnitude cap (previous fix) bounds
   any single step but does nothing about the sign staying consistent
   for 12 steps straight. Tested 3 damping mechanisms via
   counterfactual; **centering `correction` per-channel per-step**
   (subtract its own spatial mean) won cleanly — peak |vx_mean| dropped
   10x (1.43 → 0.14) and plateaued instead of drifting, verified to 80
   steps.
2. **Mass dissolution = structural, not just this bug.** Isolated test:
   a single ball, real ground-truth flow, zero model involved, pure
   `grid_sample(mode="bilinear")` warping — peak intensity collapsed to
   13% by step 10 and <1% by step 16 from resampling alone.
   `mode="nearest"` on the same test fully eliminated the collapse,
   cleanly isolating bilinear's neighbor-averaging as the mechanism.
   This is architecture-independent — plausibly why "balls drift apart
   and dissolve" recurred across every model tried this session, each
   for different specific reasons, but all resampling through
   interpolation every autoregressive step. Diagnosis: **both** the
   correction-head bug and the structural warp diffusion are real and
   compounding, neither substitutes for the other.

**Fixes applied** (`model/net.py`, commit `4ec5348`):
```python
correction = torch.tanh(self.correction_head(x)) * self.max_correction
correction = correction - correction.mean(dim=(2, 3), keepdim=True)
...
warped = F.grid_sample(g_t, sample_grid, mode="bicubic", padding_mode="border", align_corners=True)
```
`mode="bicubic"` chosen over `nearest` (which fully eliminates blur in
the idealized rigid-shift probe but risks aliasing under a real,
per-pixel, non-integer predicted flow field) after a quick local
probe: bicubic retained 0.24 peak intensity at step 10 vs. bilinear's
0.07, in the same single-ball isolation test. Centering changes
`correction_head`'s semantics — a spatially-uniform raw output now
cancels to exactly zero (redistribution only, no level shift); updated
tests accordingly (28/28 pass).

Retraining as `stage2_flownet_h12_v3.pt` (job 2824) — required
adaptation to the changed forward pass, not a hyperparameter sweep.
Verification pending once job 2824 completes.

## 2026-09-21 — v3 verified: step-1 fixed, oscillation/banding gone, VX drift persists (worse mechanism found)

`stage2_flownet_h12_v3` (job 2824) verified via the standard pipeline:

- Step-1 fixed: **0.88x** MSE-vs-copy-baseline (was 1.08x on v2, back
  to beating the trivial baseline).
- No period-2 oscillation, no strong horizontal banding.
- **But VX drift persists** — worse than the pre-retrain counterfactual
  predicted. `mean1` (VX) climbs monotonically from 0.016 to **1.30 by
  step 30** (counterfactual on the old checkpoint predicted a plateau
  around 0.14-0.16). Unbiased subagent review of the diagnostic grid
  found a new periodic diagonal ripple pattern emerging from step ~12,
  plus persistent top/bottom edge brightening.

**Root cause of the persisting drift, found via a synthetic (no-model)
probe**: `grid_sample`'s `padding_mode="border"` (needed for the
gridding-artifact fix, `findings-gridding-artifact.md`) is **not
mass-conserving under sustained one-directional flow**. Repeatedly
shifting a synthetic ramp field (no model at all) with
`padding_mode="border"`, `dx=0.6`/step, 20 steps: mean grew from 0.50
to 0.71 — clamped boundary reads duplicate high-value edge content
into the frame every step it's shifted the same direction, pumping the
mean up. `padding_mode="reflection"` showed the same drift (0.50 ->
0.71, nearly identical). `padding_mode="zeros"` doesn't pump (slight
decay, 0.50 -> 0.47 over the same test) but that's the padding mode
the gridding-artifact fix specifically moved away from, for good
reason (100% OOB perimeter every step, edge brightening).

Centering `correction` (previous fix) still helps — it removed the
*correction-head-driven* part of the drift — but this newly-found
padding-mode-driven drift is a second, independent, larger-magnitude
mechanism that the centering fix doesn't touch, and retraining alone
can't fix it either (it's a property of the warp op, not something
gradient descent on this loss can learn around when the flow field
itself needs to point consistently downward/outward under gravity).

Dispatched a subagent (`docs/debugging/drift-padding-conservation-brief.md`)
to find a fix that keeps the gridding-artifact fix intact while
removing the directional mass-pumping — e.g. explicit per-step mass
renormalization after the warp, a different boundary treatment, or
something else. Not yet resolved.

## 2026-09-21 — real drift mechanism found: Jacobian oversampling, not padding mode; renormalization fix applied

Subagent investigation (`docs/debugging/findings-padding-mass-conservation.md`)
found the brief's own leading hypothesis (`padding_mode="border"`
pumping mass) was real but **secondary** — a real-model counterfactual
(`border` -> `zeros`, frozen weights) only cut VX drift ~40% (1.302 ->
0.765), and a strictly-interior synthetic flow (padding mode never
invoked, zero OOB by construction) still pumped mass 2.4x over 15
steps. **Real mechanism: `grid_sample` does no Jacobian-determinant
correction for locally convergent flow fields** — any region where
predicted flow compresses gets structurally oversampled every
autoregressive step, independent of padding mode. This is why
`padding_mode="reflection"` also pumped nearly identically to
`"border"` (1.285 vs 1.302).

Tested 8 candidate fixes on frozen weights (renormalization variants,
`max_flow` reduction, edge-only flow damping, `"zeros"`/`"reflection"`
padding). Renormalizing PROB channel 0 only reduced VX drift ~26% (via
the PROB->VX/VY coupling from the period-2 Jacobian probe) with no MSE
regression (step-1 PROB MSE-vs-copy-baseline actually *improved*
slightly, 0.841x vs baseline's 0.881x); renormalizing all 3 channels
fully flattened the drift (1.302 -> 0.016) but was rejected as the
default since it pins VX/VY's global mean to the initial frame for the
whole rollout, not physically justified for velocity fields (though
kept as a documented fallback).

**Fix applied** (`model/net.py`, commit `284b03f`): after
`warped + correction`, rescale channel 0 (PROB) so its frame-wide sum
matches the pre-warp frame's sum:
```python
prob_in = g_t[:, 0:1].sum(dim=(2, 3), keepdim=True)
prob_out = out[:, 0:1].sum(dim=(2, 3), keepdim=True)
scale = prob_in / (prob_out + 1e-6)
out = torch.cat([out[:, 0:1] * scale, out[:, 1:]], dim=1)
```
VX/VY (channels 1/2) deliberately left unrenormalized. Verified as a
no-op whenever `correction` is already centered (centering already
makes `correction`'s per-channel sum zero, so the only thing the
renorm actually corrects is the warp's own Jacobian effect) — 28/28
tests pass unchanged. Sanity rollout confirms PROB mass held exactly
constant across 30 steps on an untrained model.

Retraining as `stage2_flownet_h12_v4.pt` (job 2825). Verification
pending once it completes.

## 2026-09-22 — v4 verified: drift fixed, mass held exact, but a new blocky/tiled artifact appeared

`stage2_flownet_h12_v4` (job 2825) verified via the standard pipeline:

- Step-1: 0.87x MSE-vs-copy-baseline — consistent with v1-v3 (0.83-0.87x
  across all four flownet checkpoints), no regression.
- **PROB mass held exactly constant**: `sum0` = 67.181 at every single
  step from 0 to 30 (direct confirmation the renormalization works
  exactly as designed).
- **VX/VY drift fixed**: VX (mean1) rises then *plateaus* around
  0.52-0.57 from step ~16 onward (was climbing unboundedly to 1.30 in
  v3); VY (mean2) similarly settles around -0.11 to -0.12. No more
  runaway one-directional drift. No period-2 oscillation.
- `max0` (peak intensity) still decays gradually from 0.81 to 0.04 over
  30 steps — this is the separate, already-documented residual-
  diffusion issue from `findings-correction-drift-and-mass-dissolution.md`
  (bicubic reduces but doesn't eliminate resampling blur); not new, not
  regressed by this fix.

**But a new artifact appeared**: an unbiased subagent review of the
diagnostic grid found sparse noise (steps 0-3) transitioning gradually
through step 5, then a **sudden onset at step 8 of a coarse,
axis-aligned blocky/tiled mosaic pattern** — large rectangular patches
of relatively uniform brightness with grid-aligned edges, "patchwork/
quilt" in character. This tiling persists and dominates through step
30, with a bright hotspot developing in the upper-left region from
step ~20 onward. Visually and structurally distinct from every prior
artifact this session (windowed-attention checkerboard, the first
flownet lattice/edge-brightening, the period-2 flicker, the diagonal
ripple/horizontal banding from the drift bug) — not yet diagnosed.

Video sent (`videos/stage2_flownet_v4_rollout_comparison.mp4`). Four
architectural fixes now landed and verified working as designed
(bounded+centered correction, border/replicate padding, bicubic
resampling, PROB mass renormalization) — the drift/oscillation family
of bugs is resolved. This new blocky-tiling artifact is a distinct,
open problem, not yet investigated.

## 2026-09-22 — quilting artifact root-caused: renorm amplifying diffusion noise, not a new bug

Dispatched a subagent to root-cause the blocky/tiled artifact (visual
review confirmed onset ~step 8-12, ~5-6px axis-aligned tiles on the
50x50 grid). All plausible new-bug candidates were ruled out with
direct evidence: OOB fraction, flow saturation, GroupNorm spatial
behavior, renorm-scale magnitude, NaN/Inf, and a plotting artifact were
each checked and cleared; a Jacobian-fold "basin of attraction" probe
found ~0 correlation with the blocky regions.

**Real mechanism**: not a new bug — the already-documented bicubic
resampling diffusion (`findings-correction-drift-and-mass-dissolution.md`
Part 2) spreads PROB mass into near-zero background across ~90% of the
grid by step 8 regardless of renorm (confirmed against ground truth,
which stays ~10% nonzero). Pre-v4 this diffusion just looked like decay
(`max0` dropping). v4's exact mass renormalization (needed for the
VX/VY drift fix) has no way to distinguish that diffused background
noise from real occupancy, so it perpetually rescales the noise back up
to the true total mass every step, turning the model's own smooth
low-frequency residual structure into a persistent, growing mosaic —
same underlying diffusion bug, newly surfaced by the renorm fix
interacting with it, not a fourth independent architectural flaw.

**Fix applied** (`model/net.py`, commit `787393b`): soft-threshold PROB
(`relu(prob - prob_threshold)`, `prob_threshold=0.015`) before computing
the renorm sum and before scaling, so sub-threshold diffusion noise is
zeroed instead of amplified. Frozen-weight counterfactual on v4 cuts
nonzero-area fraction from ~0.85-0.93 down to ~0.55-0.70 by step 8-10
(direct evidence the mechanism is real and the fix engages it) but
still shows visible blockiness at frozen weights, since v4's weights
were trained under the pre-threshold forward pass and the low-frequency
structure they emit wasn't shaped with this transform in the loop.
Test suite updated: the "exact identity at init" test now checks VX/VY
channels exactly and PROB via background-stays-zero + mass-conservation
(soft-threshold is a deliberate shrink-and-rescale, not an identity map,
for the PROB channel specifically); the correction-bounds test now
scopes its magnitude/mean assertions to VX/VY, since PROB's final value
passes through the threshold+renorm transform afterward. 29/29 tests
pass. Findings: `docs/debugging/findings-quilting-artifact.md`.

Retraining as `stage2_flownet_h12_v5.pt` (job 2826) to let the model
adapt to the new forward pass — required adaptation, not a
hyperparameter sweep, same as every other forward-pass change this
session. Verification pending.

(Job 2826 failed fast on a stale `tests/test_dataset.py`/`model/dataset.py`
mismatch on the remote Polaris copy, unrelated to this change — full
`model/`+`tests/` sync fixed it, resubmitted as job 2827.)

## 2026-09-22 — v5 verified: quilting fully fixed, but exposed a worse VX/VY drift regression

`stage2_flownet_h12_v5` (job 2827) verified via the standard pipeline:

- Step-1: **0.68x** MSE-vs-copy-baseline — best of any checkpoint this
  session (previous best 0.83-0.87x across v1-v4), no regression.
- **Quilting artifact fully eliminated**: unbiased subagent visual
  review of a v4-vs-v5 side-by-side diagnostic grid found *zero*
  blocky/tiled texture at any step in v5 (vs. v4's persistent coarse
  patchwork from step ~8 onward) — instead v5 converges to a smooth,
  gradually-flattening bright/dark boundary.
- **But VX drift came back much worse**: a full 30-step stats dump
  showed VX (mean1) climbing to a **1.37 peak** (vs. v4's 0.5 plateau)
  before slowly decaying — worse than the VX drift problem that was
  supposedly fixed twice already this session. Root-caused via
  subagent (`docs/debugging/findings-vx-drift-regression-v5.md`): not
  a new bug — v4's own diffuse-PROB quilting bug had been
  *accidentally* flattening this same pre-existing `grid_sample`
  Jacobian-non-conservation drift (already documented, deliberately
  left unfixed for VX/VY in
  `findings-padding-mass-conservation.md`) by spreading the field to
  near-uniform noise, leaving nothing for convergent flow to
  concentrate. Fixing the quilting bug correctly removed that
  accidental brake, so the always-latent VX/VY pump ran unchecked.
  Confirmed: 100% of the per-step VX/VY mean growth is attributable to
  the warp step itself in both v4 and v5 (`correction`'s contribution
  is exactly 0.0 every step, both checkpoints); v5's flow field is if
  anything *smaller* than v4's, ruling out "the model learned more
  aggressive flow".

**Fix applied** (`model/net.py`, commit `4769282`): additively
recenter VX/VY's frame-wide mean to match the pre-warp frame's mean,
every step (signed-quantity analog of PROB's multiplicative renorm).
Frozen-weight counterfactual on v5 exactly pins VX/VY to their initial
values through 30 steps with no step-1 MSE cost. This is the second
time leaving VX/VY unrenormalized has failed (first as unaddressed
drift, now as a regression from an otherwise-correct fix) — the
previously-documented "not physically justified" tradeoff (can't
represent a genuine secular mean trend across the rollout) is now
accepted given zero measured cost on every other tracked metric.
29/29 tests pass (updated to check VX/VY exact identity in the
zero-flow case, since recentering is a no-op there).

Retraining as `stage2_flownet_h12_v6.pt` (job 2828) to adapt to both
fixes together. Verification pending — standard pipeline plus the
same 30-step per-channel stats dump used to catch this regression,
since the diagnostic grid + MSE checks alone did not surface it (only
the full stats dump comparing v4 vs v5 side by side did).

## 2026-09-22 — v6 verified: quilting and VX/VY drift both resolved, no regressions; adopted as current default

`stage2_flownet_h12_v6` (job 2828) verified via the full pipeline
(step-1 MSE, 30-step per-channel stats dump, diagnostic grid + unbiased
subagent visual review):

- Step-1: **0.71x** MSE-vs-copy-baseline (v5 was 0.68x, v4 0.84x) — no
  regression, still the best or near-best of any checkpoint this
  session.
- **VX/VY drift fully fixed and stable across an actual retrain (not
  just the frozen-weight counterfactual)**: `mean1`/`mean2` are pinned
  exactly at their initial values (+0.0162 / -0.0098) at *every* step
  0-30, confirming the recentering fix generalizes through training,
  not just as a frozen-weight patch.
- **Quilting artifact confirmed still absent**: unbiased subagent
  review of a v5-vs-v6 side-by-side diagnostic grid found no
  checkerboard/stripe/rectangular-tile pattern in either row.
- One new, minor, distinct texture difference noted by the same
  review: v6's bright/dark rollout boundary is more jagged/
  saw-toothed than v5's smoother one, and v6 reaches its final
  near-uniform state a few steps faster. Not a periodic tiling
  artifact (explicitly ruled out by the reviewer) and not a regression
  of anything previously fixed — most likely a downstream texture
  consequence of the VX/VY-pinning fix interacting with the
  already-documented residual bicubic-diffusion/peak-decay issue from
  `findings-correction-drift-and-mass-dissolution.md` (both v5 and v6
  ultimately collapse toward a similar bright-bottom/dark-top band at
  late steps) rather than a new bug. Flagged, not yet investigated
  further — the diffusion/peak-decay issue was already a known,
  documented, lower-priority open item before tonight's quilting work
  started.

**Conclusion**: the quilting bug reported at the top of this log is
resolved, with its one second-order side effect (the VX/VY drift
regression it exposed) also resolved, verified through an actual
end-to-end retrain with no MSE, oscillation, or gridding-lattice
regressions. `stage2_flownet_h12_v6.pt` is adopted as the current
default checkpoint going forward. `model/net.py`'s docstring documents
all six architectural fixes landed this session (windowed-attention
replacement, bounded+centered correction, border/replicate padding,
bicubic resampling, PROB mass renorm + soft-threshold, VX/VY mean
recentering).

The remaining known, lower-priority open item is the residual
peak-intensity decay / bicubic diffusion (`max0` dropping over a
rollout, `findings-correction-drift-and-mass-dissolution.md` Part 2) —
already documented, not a new regression, candidate fixes (nearest-
neighbor resampling under the model's real predicted flow, or a
training-time sharpness/peakiness regularizer) not yet tried.

A fresh unbiased subagent video review (frame-by-frame, not just the
sparse diagnostic-grid columns) of `videos/stage2_flownet_v6_rollout_comparison.mp4`
makes clear this "lower-priority" item is now the dominant remaining
problem in practice: with quilting and VX/VY drift both fixed, v6's
rollout still visibly collapses from crisp discrete balls to a
featureless blurred blob by step ~4, and to a fully flat, static
two-tone band with *zero* ball structure by step ~7-13, staying there
for the rest of the rollout — while ground truth keeps evolving with
sharp, discrete, physically structured balls the whole time. This
matches this session's opening user-raised concern ("we need a way to
keep the balls together, they seem to drift apart... dissolve") more
directly than any of the artifact-specific bugs fixed so far — it was
simply harder to see clearly underneath the checkerboard/gridding/
oscillation/quilting/drift artifacts that got fixed first. Promoting
this to the active investigation, next in this log.

## 2026-09-22 — peak-decay root-caused: objective-level (loss prefers blur), not resampling/horizon; fix attempt exposes a deeper tradeoff

Subagent investigation (`docs/debugging/findings-peak-decay-dissolution.md`)
root-caused the collapse to `occupancy_weighted_mse` itself: evaluated
directly on synthetic targets, the loss provably prefers a diffuse,
mass-matched blob over a sharp disk merely shifted ~2px, since it has
no mechanism to penalize spread once mass is inside the (binary)
occupied mask. The model's flow head learned locally divergent flow as
its implementation of this incentive; this happens under any
`grid_sample` mode (bilinear/bicubic/nearest all tested, all converge
to the same collapse), so it isn't a resampling artifact, and the
collapse completes well inside the trained `horizon=12` window, so
longer-horizon training isn't the lever either (both directly tested
and ruled out, per this session's standing rule against defaulting to
"train longer").

**Fix attempted** (`model/losses.py`, `model/train.py`, commit
`827d5b2`): added a peak-magnitude term to `occupancy_weighted_mse`
(`peak_weight=0.1` default, new `--peak-weight` train arg) penalizing
the predicted frame's global PROB peak diverging from the target's.
Verified on the exact synthetic scenario from the investigation that
this flips the loss's preference from blob-wins to sharp-wins (even
`peak_weight=0.02` already flips it). New regression test encodes both
the bug and the fix as one assertion pair. 30/30 tests pass.

Retrained as `stage2_flownet_h12_v7.pt` (job 2829). Verified via the
standard pipeline: step-1 MSE unaffected (0.70x vs v6's 0.71x); the
targeted metric is genuinely fixed (`max0` now stays 0.54-0.85 across
all 30 steps, vs. v6's collapse to a flat 0.036-0.040 floor by step 8);
VX/VY drift and quilting both remain fixed. **But an unbiased video
review found a new, not-clearly-better failure mode**: instead of
blurring uniformly everywhere (v6), v7 concentrates into a smeared blob
in one screen region while abandoning most other balls to solid black,
plus a new faint periodic diagonal ripple from step ~23 on. A follow-up
synthetic probe (`findings-peak-decay-dissolution.md` Part 5) shows
this is not a bug introduced by the peak term but a second,
previously-hidden failure mode of the *same* base loss: once a ball's
predicted position has already drifted (the normal state past the
first few autoregressive steps), committing to any specific nearby
position costs more loss than giving up on it entirely (a
confident-but-wrong prediction double-penalizes: false-positive bright
cell + missed true-position cell). Closing off the "blur as hedge"
strategy (which worked, as designed) simply revealed "give up as
hedge" as gradient descent's next-cheapest option under the same
underlying objective — the textbook two-sided failure mode of
deterministic point-estimate regression under genuine multimodal
(chaotic multi-ball) uncertainty.

**Conclusion: this is not resolved, and is not a same-night-fixable bug
like the quilting/drift chain.** It's a fundamental property of dense
per-cell regression under a pointwise loss for an inherently chaotic
multi-ball system, not a clean implementation defect. Recommended next
steps (not attempted, deliberately left for human judgment given the
scope): regional/local peak-preservation instead of one global peak
(untested even synthetically here beyond an inconclusive isolated
probe); explicitly bounding and reporting the honest rollout horizon
this architecture can support rather than continuing to chase 30-step
quality; or a genuinely different output representation (per-ball
tracked state, or a probabilistic/multi-hypothesis output) — a
substantially larger redesign than anything else done this session.

**Checkpoint recommendation**: `stage2_flownet_h12_v6.pt` remains the
current default — every artifact/drift bug from this session's chain
(windowed-attention checkerboard, edge/gridding lattice, period-2
oscillation, quilting/tiling, VX/VY drift) is fixed and verified in it,
with a well-understood, honestly-reported residual blur-collapse
limitation. `stage2_flownet_h12_v7.pt` is kept as a documented
experimental variant (different, not clearly better, residual
limitation) — useful as evidence for the diagnosis above, not adopted
as a replacement default. Both checkpoints, and the three next-step
options above, should be reviewed by a human before further model-side
work continues, since the real remaining options are design decisions
about the modeling approach, not bugs to fix.

## 2026-09-22 — 100-step v7 rollout: "give up" collapse terminates in a periodic ripple/lattice, not inert black

Rendered a longer (100-step, vs. the standard 30) rollout comparison on
`stage2_flownet_h12_v7.pt` (`videos/stage2_flownet_v7_rollout_comparison_100step.mp4`,
seed 4738) to see where the "give up as hedge" failure mode goes past
the horizon already examined. Unbiased subagent frame-by-frame review
(unprimed) found the collapse is progressive and one-way, not static
once it starts:

- Steps 0-5: sharp match to ground truth degrades almost immediately.
- Steps 10-20: rapid content loss to near-black outside a few
  surviving bright blobs; a faint diagonal ripple begins appearing.
- Steps 25-50: mostly black except 2-3 static bright blobs (not
  tracking ground truth's continued settling) plus an expanding faint
  diagonal ripple/checkerboard texture.
- Steps 50-100: the surviving bright blobs fade out entirely by
  ~step 70-80; the ripple grows to **dominate the entire frame as a
  fine periodic checkerboard/lattice by step 90-100** — low-amplitude,
  no bright colors, no resemblance to ground truth's still-rich,
  evenly-spread ball field. Never recovers.

This adds a data point to the blur-vs-giveup writeup above: the "give
up" hedge doesn't converge to inert/flat output at long horizon, it
converges to a periodic lattice texture — visually reminiscent of this
session's earlier checkerboard/edge-gridding artifacts (both already
root-caused and fixed at short horizon: windowed-attention window bias,
and conv zero-padding + `grid_sample` OOB perimeter). Plausible
explanation: those fixes addressed the *specific* mechanisms that were
producing a lattice at short horizon, but did not remove every
periodic bias the conv/upsampling stack can express — once the model
has abandoned confident content (per the give-up mechanism above), a
weaker latent periodic bias that was previously masked by real content
has nothing competing with it and grows to dominate. Not yet
investigated further; relevant evidence for the redesign-scoping
conversation, not a new independent bug to chase in isolation.

## 2026-09-22 — three parallel investigations resolve the blur-vs-give-up tradeoff's open options

Dispatched three parallel subagents against the three next-step options
left open in `findings-peak-decay-dissolution.md`, plus the new
periodic-lattice evidence from the 100-step v7 rollout. Briefs and full
context: `docs/debugging/flownet-open-issues-v2.md`.

**Option 1 — regional/local peak-preservation loss**
(`findings-regional-peak-loss.md`): tested 8 formulations (global,
tile4/8/16, top-k, per-connected-component, ball-centered window5)
across 2-20 balls x 0.5-8px drift. A fixed 5x5 window centered on each
ball's true position beats the current global peak term badly at low
drift/moderate density (14-15/15 win rate), but two hard failure modes
persist across every formulation tested: neighbor-masking (a give-up
ball sharing a region with a correctly-predicted neighbor gets falsely
credited) and a fixed-window ceiling (fails outright once drift exceeds
the window's half-width). These trade off against each other — no
region size escapes both. Even at 10x the current `peak_weight`, the
hardest configs (20 balls, 4px drift) plateau at 60-80% win rate.
**Conclusion: real, adoptable improvement over the global term, but not
a resolution of the underlying tradeoff.**

**Option 2 — honest rollout horizon** (`findings-honest-horizon.md`):
Hungarian-matched ground-truth ball positions against predicted local
maxima (recall + position error), 3 seeds x 4 densities (20/50/125/250
balls) x 40 steps, cross-checked against an occlusion ceiling and a
shuffled-peak null baseline. v7 is **not** longer-horizon-safe despite
fixing the aggregate peak-decay metric — it drops below 50% recall
almost immediately (step 1 vs. v6's step 4-7), sacrificing tracking
breadth to keep one region sharp; past that it holds a small
above-chance signal out to 40+ steps but concentrated in one
unpredictable "give-up" blob, not usable as whole-scene tracking. v6
decays to genuine chance-level (confirmed via null test) by step
~10-12 regardless of density. **Recommended honest horizon (recall
≥0.5): 6 steps sparse (20-50 balls), 4 steps medium (125), 1-2 steps
dense (250) — same for both checkpoints.** Beyond ~10-12 steps neither
checkpoint should be trusted for any purpose. Confirms this is an
architectural/objective-level limit, not a training-progress one; the
practical lever available today is capping scenario density, not
rollout length.

**New evidence — long-horizon periodic lattice**
(`findings-long-horizon-lattice.md`): the diagonal ripple that grows to
dominate the v7 100-step rollout by step 90-100 is a **third, distinct
periodic bias** from the two already fixed this session (not the
border/`padding_mode="zeros"` OOB mechanism, not the period-2 temporal
oscillation). FFT shows a fine ~3-5px 2D diagonal weave, isolated via
counterfactual to **bicubic `grid_sample` resampling's own kernel-scale
texture** (added specifically to fix peak-decay) — swapping to
bilinear shifts it back to the old coarse ~10-13px scale. This texture
is latently present in both v6 and v7 identically (same bicubic-warp
code) but collapses to near-zero in v6 by step 10, while staying
substantial (9-42% of spectral power) indefinitely in v7 — it's not an
independent bug, it's a second byproduct of the same bicubic-for-
peak-decay tradeoff, unmasked once give-up empties the frame of
competing real content.

**Overall conclusion**: none of these three findings overturn the
`findings-peak-decay-dissolution.md` diagnosis — they sharpen it.
Regional peak-preservation is worth landing as an incremental
improvement, but the honest-horizon numbers (single-digit steps at
real densities, for both checkpoints) confirm the fundamental limit is
real and roughly checkpoint-independent, and the new lattice finding
shows that even the "fixed" side of the tradeoff (bicubic resampling)
carries its own latent cost that only surfaces once give-up removes
competing content. The three architecturally-scoped options from
`findings-peak-decay-dissolution.md` are now: adopt regional
peak-preservation as a modest quality bump, formally cap supported
scenario density/horizon per the numbers above, or commit to the
larger redesign (per-ball tracked state / probabilistic output) needed
to genuinely resolve the tradeoff. Still a human design decision, not
a bug to fix.
