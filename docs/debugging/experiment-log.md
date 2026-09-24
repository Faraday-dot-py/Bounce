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

## 2026-09-22 — two more loss-reduction investigations: focal reweighting inert, local mass-conservation is the strongest lever found

Two more parallel investigations, prompted by user design questions
about the loss beyond the earlier regional-peak-loss/honest-horizon/
lattice trio. Briefs: `docs/debugging/flownet-open-issues-v3.md` (focal
reweighting) and `flownet-open-issues-v4.md` (mass-conservation loss).

**Focal-style error reweighting** (`findings-focal-reweighted-loss.md`):
tested power reweighting (`err^2 * |err|^gamma`) and a CenterNet-style
confidence-modulated focal term on the same give-up-vs-commit synthetic
probe. **Inert — no measurable effect.** Win rate stayed within
15-trial sampling noise of the unweighted base loss (0.19-0.23 vs.
0.213) across every gamma tested, with identical saturation pattern, and
no synergy combined with `window5`. Root cause: give-up's and commit's
dominant per-pixel errors are already comparable in *magnitude*, just at
different pixel counts/locations — a purely magnitude-dependent
monotonic reweighting scales both sides similarly and can't change which
is cheaper. Correctly avoided rewarding hallucination (unlike `topk8`),
but doesn't move the needle. Not worth pursuing further.

**Local/regional mass-conservation loss**
(`findings-mass-conservation-loss.md`): two parts.

*Part A* — the model already hard-renormalizes global PROB mass every
step (architectural, not a loss — `model/net.py`), so first checked
whether that existing mechanism contributes to the give-up blob's
visual character. Instrumented rollout (v7, seed 4738) confirmed the
renorm rescale factor grows to ~1.3x by step 70-100, genuinely
brightening the surviving blob, but relative spatial concentration
(peak/total-mass) is actually *lower* with renorm on than off — renorm
spreads its correction across the occupied footprint rather than piling
onto one spot. Without renorm, mass hits exact zero by step ~35
regardless of strategy (the warp+threshold pipeline is intrinsically
lossy). **Renorm is a real but secondary contributor — brightens, does
not cause, the concentration — and its main role is preventing total
blackout.**

*Part B* — extended the regional-peak-loss probe with a genuinely
different operator: per-tile **integrated (summed) PROB mass** instead
of per-tile **max**. `tile16` (sum) hit mean win-rate 0.753 vs. peak's
best (`tile16` peak, 0.567), and critically **held up in the regime
that broke every peak-based formulation** — 13/15 at 20 balls/8px drift
vs. peak-`tile16`'s 1/15. Mechanism: sum is additive, so a
confidently-predicted neighboring ball's peak can't fully mask a
give-up ball's *missing* mass the way a shared "is there a peak
somewhere here" check could. `window5`-sum performed on par with
`window5`-peak, inheriting the same fixed-window drift ceiling.

**This is the strongest single lever found across all four loss-level
investigations this session** (regional-peak, honest-horizon,
long-horizon-lattice, focal-reweighting, mass-conservation). Recommend
prioritizing `tile16`-sum mass-conservation as the next real retrain
candidate (over `window5`-peak, the prior leading candidate) — it's the
first formulation tested that doesn't collapse in the
high-density-and-high-drift regime that defines a real multi-ball
rollout past the first few steps. Still does not, on its own, resolve
the redesign-scale options (2)/(3) from `findings-peak-decay-
dissolution.md` — untested end-to-end (synthetic-loss-value evidence
only, same caveat as every investigation in this chain) — but it is the
first candidate worth an actual retrain+rollout-video validation before
falling back to the honest-horizon-cap or redesign options.

## 2026-09-22 — v8 retrain (tile16-sum mass-conservation): synthetic-probe promise does NOT hold at the real-model level, new dominant periodic-fan artifact

Implemented `tile16`-sum mass-conservation as `mass_weight`/`mass_tile`
params in `occupancy_weighted_mse` (`model/losses.py`, commit `92fe715`),
wired through `model/train.py` (`--mass-weight`, `--mass-tile`), 3 new
regression tests (33/33 pass). Retrained as `stage2_flownet_h12_v8.pt`
(job 2831, same hyperparameters as v7 plus `--mass-weight 0.1
--mass-tile 16`), completed cleanly in ~10 min, final training loss
comparable scale to v7 (12.77 vs 12.30).

**Step-1 accuracy**: v8 MSE ratio 0.78x baseline, a modest regression
vs. v6's 0.71x and v7's 0.70x (`scripts/eval_step1_baseline.py`) —
expected, since the mass term targets longer-horizon behavior, not
step-1.

**Rollout behavior — the important result.** An unbiased subagent
review of a v6/v7/v8 diagnostic grid (steps 0-30, seed 4738) found v8
does **not** land anywhere close to what the synthetic give-up-vs-commit
probe predicted. Instead of holding structure longer or degrading more
gracefully than v6/v7, v8 develops **a strong, regular diagonal-stripe/
fan pattern that dominates the entire frame by steps 16-30** — far more
severe and far earlier than the faint diagonal ripple found in v7's
100-step rollout (`findings-long-horizon-lattice.md`, which only became
frame-dominant around step 90-100, and even then stayed low-amplitude).
v6 saturates to a bright blob, v7 fades toward black with a faint
stripe remnant; v8 fills the entire frame with a dense periodic grating
starting as early as step 8-12.

**Assessment — this is not a contradiction of the earlier synthetic
finding, but a real limit of that finding's scope.** The synthetic
probe in `findings-mass-conservation-loss.md` evaluated the *loss
value* of two hand-constructed candidate predictions (give-up vs.
commit-with-drift); it never ran gradient descent, so it could not
observe a third option gradient descent might discover: a periodic
texture that satisfies both the tile-sum mass constraint (correct
total mass per tile, everywhere) and the global peak constraint
(plenty of local peaks) simultaneously, without needing any real
per-ball localization. `findings-long-horizon-lattice.md` had already
established that bicubic `grid_sample` resampling carries a latent
periodic-texture bias, present in both v6 and v7, normally suppressed
by real content. The mass term's constant, ubiquitous pressure to keep
every tile's mass topped up appears to have given the optimizer a
direct incentive to lean on that latent bias as a cheap, spatially-
uniform way to satisfy the constraint everywhere at once — worse than
either of v6/v7's failure modes because it's not a hedge that only
shows up once give-up empties the frame, it's actively reinforced by
the training objective from early rollout steps.

**Recommendation: do not adopt v8. `stage2_flownet_h12_v6.pt` remains
the current default** (same status as before this investigation chain
started). `mass_weight`/`mass_tile` stay in `model/losses.py` and
`model/train.py` (real, tested, useful primitives — e.g. for a lower
weight or a follow-up investigation), but the checkpoint itself is not
promoted. This is the clearest demonstration yet in this investigation
chain of the standing risk flagged in every one of the synthetic-probe
docs: synthetic single-step loss-value comparisons show whether a term
*could* fix an incentive between two hand-picked candidates, not what
a trained network actually converges to under gradient descent — real
end-to-end validation is not optional, and in this case it overturned
the recommendation. If mass-conservation is revisited, it should be at
a substantially lower weight (the synthetic sweep never tested weight
sensitivity for the sum-based terms the way the peak-term investigation
did) and/or combined with an explicit penalty on the specific bicubic-
texture frequency band identified in `findings-long-horizon-lattice.md`,
rather than retried at the same nominal 0.1 weight.

## 2026-09-22 — richer temporal input (N-frame history) tested on both architectures: rejected on both, different failure modes

Motivated by a design question about why collisions produce any
uncertainty at all given deterministic physics: (1) collision outcomes
are a chaotic/sensitive function of exact contact position+velocity, so
small state-estimate error is amplified at contact; (2) both
architectures only ever saw a single previous frame `g_t`, so VX/VY
were learned/estimated quantities, not measured from real temporal
data. Hypothesis: stacking the last N frames as input (finite-difference-
style velocity context) might reduce state-estimate noise feeding into
collisions. Dispatched two parallel agents in isolated git worktrees to
test this independently on the current flow-warp architecture and the
abandoned windowed-attention architecture (recovered from `f0e709b^`).
Neither agent's work was merged to master — see worktree branches below.

**Flow-warp architecture** (worktree branch
`worktree-agent-a425f76b539cb1a76`, commit `31f3c18`): added a
`history` arg to `BounceNextFrameModel` (stem takes `3*history` stacked
channels; only the most recent frame is warped/corrected/renormalized,
identity-at-init preserved). Trained `stage2_flownet_h12_v9.pt`
(history=3, job 2834, same hyperparameters as v7/v8 otherwise).
Step-1 MSE regressed to 0.77x baseline (v6/v7 were 0.70-0.71x). An
unbiased subagent review found a new **dominant diagonal moiré/grating
pattern from step 8 onward**, worse than either v6's blur or v7's
give-up blob; blob-count check confirmed overcounting true ball count
by 1.5-3x from step 5 on. **Not adopted** — root cause of the moiré not
isolated. Findings: `docs/debugging/findings-richer-temporal-input.md`
(in that worktree, not on master).

**Windowed-attention architecture** (worktree branch
`worktree-agent-aadfa949e665f5be6`, commit `be58cc0`): recovered the
pre-`f0e709b` Swin-style backbone, widened the input stem for N-frame
history the same way. Trained N=3 at the architecture's final
best-tuned hyperparameters (job 2835) plus an N=1 baseline from the
recovered code for direct comparison. Step-1 MSE **regressed** to 1.29x
baseline (N=1 was 1.04x). The checkerboard/lattice artifact was
**present and unchanged in onset/timing in both** N=1 and N=3 (onsets
~step 8, dominates by step 20-30) — only its anisotropy shifted
(vertical/corner-blob vs. horizontal banding). Confirms richer temporal
input doesn't touch the window-partition-bias root cause documented
back at the top of this log. **Not adopted.** Findings:
`docs/debugging/findings-richer-temporal-input-windowed-attention.md`
(in that worktree, not on master). Diagnostic grid:
`videos/windowed_hist_comparison_grid.png` (copied to master's
`videos/`, not git-tracked).

**Conclusion**: richer temporal input is rejected as a lever on both
architectures tried this session, each for a different reason (new
moiré artifact on flow-warp; no artifact improvement plus regressed
accuracy on windowed-attention). `stage2_flownet_h12_v6.pt` remains the
overall project default. The underlying design question (how to reduce
collision-time state-estimate noise) is still open — richer temporal
input was one candidate answer and it didn't work; per-ball tracked
state remains the untried larger-redesign option from the
peak-decay/give-up investigation above.

Also this session: standard training epoch count for future Polaris
runs (`scripts/polaris_train.sh`, `scripts/polaris_train_v8.sh`) was
turned down from 80 to 50 per user direction (most runs don't converge
past there) — a project convention change, not an experiment result.

## 2026-09-22 — token-per-ball model designed, implemented, and first real training run submitted (job 2840)

New architecture (design doc `docs/superpowers/specs/2026-09-22-token-per-ball-model-design.md`,
plan `docs/superpowers/plans/2026-09-22-token-per-ball-model.md`): each
ball is a persistent tracked token (position/velocity/hidden state)
instead of an implicit pattern in a dense grid, with radius-graph
attention for collision dynamics (locality bias, needed for the
train-small/tile-large goal) and a predictive occlusion gate deciding
whether to trust a fresh grid observation or run open-loop through a
contact event. Output is still rasterized through `bounce.py`'s exact
splat formula, so it stays comparable to the flow-warp baselines'
loss/eval pipeline. Motivated by the same open question flagged at the
end of the richer-temporal-input investigation: collision-time state
noise, this time addressed via explicit per-token tracking rather than
more grid-frame context.

Implemented via subagent-driven development, 10 tasks, each with an
independent implementer + task-reviewer + fix-round cycle. Three plan
defects were caught and fixed along the way (test fixtures whose
radius/position combination didn't give the sub-pixel resolution their
assertions assumed; a real zero-gradient bug in the planned attention
code for single-neighbor tokens, fixed with a standard GAT self-loop).

**A final whole-branch review (dispatched on a more capable model)
found 3 Critical defects that had survived all 10 task-level reviews**,
because every task's unit tests happened to exercise a degenerate
configuration (`observation_weight=0.0`, detection radius=1.5, one
fixed grid origin) that the assembled system never actually runs at:

1. Off-grid detection crash (`RuntimeError` in `centroid_near` when a
   token drifts past the grid edge).
2. Time-misaligned observation blend: `TokenModel.step` compared a
   time-(t+1) prediction against a time-t observed frame, producing a
   near-zero "velocity" that multiplicatively destroyed tracked
   velocity every step at the shipped default `observation_weight=0.5`
   (verified: velocity `[2.0, 1.0]` collapsed to `[0.32, ~0]` within 2
   steps before the fix).
3. `TokenDynamics`'s attention consumed absolute token position, not
   relative geometry — translating an identical local 2-token
   configuration by a tile-scale offset produced completely different
   outputs, which would have made any tile-transfer validation result
   meaningless.

All three fixed in one fix wave and independently re-verified (exact
translation invariance now confirmed, `[2.0, 1.0]` velocity now holds
exactly across a 6-step diagnostic at the shipped default). Two
findings were ruled accepted, documented limitations rather than bugs:
detection accuracy at the production ball radius (0.75) is
meaningfully weaker than at the radius used in some unit test fixtures
(1.5) — real, unavoidable resolution limit, not something to engineer
around yet — and there's no token re-acquisition path if one drifts
off-grid (now much rarer after the velocity-matching bound added
alongside fix #2, revisit only if it resurfaces). Merged to master
(10 commits + fix wave, 73/73 tests passing).

**Job 2840** (Polaris, gpu01): first real training-scale run per the
design spec's staged validation plan — `n=20`, `2-6` balls, `horizon=12`,
50 epochs, seed 4738, `scripts/polaris_train_token_v1.sh` →
`checkpoints/token_model_h12_v1.pt`. Completed (29m39s, exit 0), but
loss was essentially flat (0.1569→0.1540 over 50 epochs). Root-caused,
not just hyperparameter noise: downloaded the checkpoint and probed
`TokenDynamics` directly — `delta_pos` was ~-30 cells (vs. sane ~0.3-0.4),
flinging every token off-grid within one step. An unbiased subagent's
review of a diagnostic step-grid (`scripts/render_token_diagnostic_grid.py`,
new this session) independently confirmed a sudden total blank-out at
step 2 (not a gradual fade), matching. Root cause: `token_grid_loss`
used a plain, unweighted per-pixel MSE (`weighted_channel_mse`), and
with the grid ~97% background, an all-empty "give up" prediction scores
better than a present-but-imperfectly-positioned ball — the exact same
degenerate solution already characterized for the flow-warp architecture
(`findings-peak-decay-dissolution.md`), reached here by the dynamics
network learning a large constant offset instead of by blur. Separately
confirmed (direct measurement: 37/79 tokens gated "occluding" at init)
that the occlusion gate's effective threshold (3.5) sits above the
default `neighbor_radius` (3.0), silently disabling observation
correction for any token with a graph neighbor at all.

Fixed (commit `31c66c4`): `token_grid_loss` now reuses
`occupancy_weighted_mse` (same background-downweighting/peak-term
approach already proven for the flow-warp models) instead of
`weighted_channel_mse`; default `neighbor_radius` raised 3.0→4.0 to sit
above the gate threshold. Local smoke test (100 samples, 8 epochs):
loss now decreases monotonically (0.1868→0.1713) and `delta_pos` stays
sane (~0.3-0.4). **Job 2842** (Polaris): re-run with the fix, otherwise
identical config, `scripts/polaris_train_token_v2.sh` →
`checkpoints/token_model_h12_v2.pt`.

**Job 2842 result**: completed (37m28s), loss plateaued at ~0.180 from
epoch ~3 onward (dipped there, then drifted slightly back up as
`sampling_p` ramped to 1.0), no further improvement through epoch 49.
Rollout video + diagnostic grid (`scripts/render_token_diagnostic_grid.py`)
+ a 16-seed grid video (`scripts/render_token_multiseed_grid_video.py`,
new this session) all reviewed by fresh unbiased subagents: the
catastrophic instant off-grid collapse is gone, but a milder, fully
consistent-across-all-16-seeds **gradual give-up dissolution** remains —
frames track ground truth closely through step ~3, blur and drop objects
one at a time through step 8, are majority-blank by step 12, nearly
total by step 20. Root-caused further (user noticed the balls visibly
drift toward one side before vanishing): traced actual `delta_pos`
values step-by-step and found a sustained, non-decaying bias in the
component matching the grid's column axis (screen-horizontal; bounce.py's
row axis, where gravity acts, stays stable) — `delta_head.bias[1]
≈ -0.38`, and mean per-step `delta_pos[:,1]` stays around -1.4 to -2.3
throughout a 10-step probe. The simulator has no left/right bias
(velocities spawn symmetrically), so this is a learned artifact: a
smooth, gradual escape to the same "vanish off-grid, stop being
penalized" state that job 2840 reached via one catastrophic jump —
cheaper for the optimizer to find because it still gets partial credit
for looking right during the first few steps, unlike an abrupt jump.
Same underlying incentive (occupancy_weighted_mse still permits
escape-to-vanish once tracking gets hard), different learned mechanism.

**Next step, not yet done**: bump `peak_weight` (currently the
flow-warp-derived default 0.1) and retrain — the peak term exists
specifically to penalize the predicted peak intensity vanishing, which
is exactly what this drift-to-vanish mechanism produces. Other
candidates discussed but not started: more training data/epochs (per
user's large-dataset/few-epoch-like-LLM-pretraining suggestion), or a
mass-conservation term (has a documented history of its own periodic-
texture failure mode on the flow-warp model, would need the same real-
retrain validation). Still not done regardless of which lever is tried
next: honest-horizon rollout comparison against `stage2_flownet_h12_v6.pt`,
per the design spec's validation plan, before any conclusion about
whether this architecture is worth adopting.

**`peak_weight` bump tried, dissolution not fixed.** Also restructured
the token dataset pipeline while at it: `BounceTokenSequenceDataset`
generated its `num_samples` trajectories from scratch in-memory on
every training-script invocation (no disk cache, unlike the flownet
path's `dataset_cache_seq*.npz`); split generation out into
`generate_dataset_samples()`/`save_dataset_samples()`/
`load_dataset_samples()` (`model/token_dataset.py`) plus a standalone
`scripts/generate_token_dataset.py`, and added `--dataset-cache` to
`token_train.py` so a training job can load a pre-generated cache
instead of regenerating it. Also applied the user's LLM-pretraining-
style intuition (large dataset, few epochs, since no grokking signal
had been observed to justify heavy repetition): job 2846 generated a
10k-sample cache (`checkpoints/token_dataset_10000_h12_seed4738.pt`,
6m18s, CPU-only); job 2847 trained 3 epochs against it with
`peak_weight=0.5` (5x default) and `ramp_epochs=2` (needed so
`sampling_p` actually reaches 1.0 within only 3 epochs — at the
original `ramp_epochs=25` the self-feed regime where this failure mode
lives would barely have been exercised at all) →
`checkpoints/token_model_h12_v6.pt`. Loss: epoch 0 (teacher-forced)
0.1847, epoch 1 (`sampling_p=0.5`) 0.1892, epoch 2 (`sampling_p=1.0`)
0.1959 — flat/mild uptick as self-feed increases, no collapse in the
aggregate number.

Diagnostic grid + an unbiased subagent review (per
`frame-artifact-review-prompt.md`) tell a different story: token_model
tracks ground truth closely through step ~8, then both rows diverge
sharply — by step 12 two of four balls have vanished/merged (no yellow
ball visible), and by step 20 the frame is completely blank while
ground truth still shows 3 distinct balls. Same gradual-give-up
dissolution pattern as job 2842's `peak_weight=0.1` run, at
essentially the same step range (partial loss by ~12, total blank by
~20) — the 5x `peak_weight` bump, 10k dataset, and reduced repetition
did not visibly delay or soften the collapse. `peak_weight` alone,
at least at this magnitude, is not the fix. Rollout video:
`videos/token_model_v6_peakweight0.5_10k3ep_rollout.mp4`.

**Not yet tried**: a larger `peak_weight` bump (0.5 was a first guess,
not tuned), the per-tile `mass_weight` term (already implemented,
never enabled for the token model — `mass_weight` defaults to 0.0 in
both `token_grid_loss` and `token_train.py`), or revisiting whether
`occupancy_weighted_mse`'s core incentive structure (vanish is always
almost-free) needs a structural fix rather than an additive penalty.

**`boundary_loss` added, dissolution actually stops — new failure mode
(motion stall) replaces it.** Rather than another indirect pixel-space
penalty, added `boundary_loss` (`model/token_losses.py`) operating
directly on `TokenModel.step`'s tracked (x, y) positions: quadratic
penalty for any coordinate outside `[0, n)`, wired into
`token_rollout_loss` via `--boundary-weight` (5 new tests, 46/46 suite
passing). Job 2849: same recipe as 2847 (10k-sample cache, 3 epochs,
`ramp_epochs=2`) plus `peak_weight=0.5` and `boundary_weight=0.1` →
`checkpoints/token_model_h12_v7.pt`.

Training logged `loss=nan` twice (once per epoch, both times on a
single reporting window, both times followed immediately by normal
finite losses on the very next window). Investigated before trusting
the result: downloaded the final checkpoint and directly checked every
parameter tensor for NaN/Inf — none found (9,956/9,956 clean). Most
likely mechanism: `boundary_loss`'s squared term is unbounded, so a
rare, still-early-training extreme position produces a forward value
large enough to overflow to `inf`; `clip_grad_norm_`'s scaling
coefficient becomes `max_norm/(inf+eps) = 0` (well-defined, not NaN,
for a finite numerator over an infinite denominator), making that
step's optimizer update a no-op rather than corrupting weights — which
matches the observed instant recovery. (`rasterize_tokens` was checked
too: an extreme position there degrades gracefully to an all-zero/
background contribution, not NaN, so it's not the source.) Exactly
where the printed value flips from `inf` to `nan` wasn't pinned down
further, but since the saved weights are verified clean, this doesn't
invalidate job 2849's result. Follow-up, not urgent: clamp the
boundary distance before squaring so a training run isn't spending a
no-op step on this at all.

Diagnostic grid + an unbiased subagent review: **no object ever
disappears or fades in the token_model row across all 8 sampled steps
through step 20** — a real, qualitative change from job 2842/2847's
gradual give-up-to-blank pattern. But a new symptom replaces it: the
two rows track closely through step ~5, then `token_model`'s objects
settle into a near-static clustered arrangement from step ~8 onward
(positions barely change between step 8/12/20) while `ground_truth`
keeps moving/rearranging — a motion stall, not vanishing. Net read:
`boundary_weight` measurably fixed the specific failure mode it was
built for (drift-to-off-grid-and-vanish), but revealed the model
substituting a different cheap escape (stop moving) rather than
learning genuinely stable long-horizon dynamics. Rollout video:
`videos/token_model_v7_peakweight0.5_boundary0.1_rollout.mp4`.

**Not yet tried**: a motion/velocity-floor term to penalize near-zero
predicted displacement (the stall's direct analog to `peak_weight`
for vanishing), tuning `boundary_weight` down in case it's
over-dominating and pinning tokens near their current position, or the
still-untried `mass_weight`/structural-incentive options noted above.

**State-space loss added (`token_state_loss`, `model/token_match.py`,
8cbae15): direct per-token MSE against the dataset's exact ground-truth
`(x, y, vx, vy)`, matched once at init instead of comparing rasterized
grids -- the structural argument being that occupancy_weighted_mse's
give-up shortcut only exists because it's grid-space (background to
hide behind); a coordinate-space target has no such shortcut. Kept
`token_grid_loss` as a smaller secondary term, ramped in from 0 on the
existing `sampling_p` schedule per user direction.** Job 2850 crashed
immediately (`state_seq` tensors left on CPU while positions were on
CUDA -- `token_train.py` was discarding `state_seq` before this change
and never needed to move it); fixed (939ffb2), resubmitted as job 2851
(same recipe as 2849: 10k-sample cache, 3 epochs, `ramp_epochs=2`,
`peak_weight=0.5`, `boundary_weight=0.1`, `state_weight=1.0`,
`grid_weight=0.1`) → `checkpoints/token_model_h12_v8.pt`. Same
recoverable NaN pattern as 2849 (a few isolated batches per epoch,
weights unaffected) but worth noting `token_train.py`'s per-epoch
`epoch_loss` accumulator has no reset like the windowed `running_loss`
does, so one NaN batch permanently zeroes out that epoch's reported
loss for the rest of training -- not fixed yet, doesn't affect the
checkpoint, but makes per-epoch loss trend unreadable from the log.

Diagnostic grid + unbiased subagent review (no hypothesis primed):
early rollout (steps 0-5) tracks ground truth closely, including color
and orientation, matching almost pixel-for-pixel through step 3. Late
rollout (steps 8-20) diverges sharply -- by step 8 two of four objects
are missing entirely from the model's frame; by step 12 the model
shows overlapping/merged blobs plus a trailing streak while ground
truth still has 4 clearly separated objects; by step 20 only one clear
blob survives (plus a faint smear), and even that blob is
desaturated/lower-contrast than ground truth. No checkerboard/banding.
Net read: **this is job 2842's original give-up-to-vanishing pattern,
not job 2849's motion-stall** -- the state-space loss did not
structurally close the shortcut in practice, at least not at these
weights/this little training (3 epochs, same as every run in this
series). Video: `videos/token_model_v8_state_loss_rollout.mp4`,
diagnostic grid: `videos/token_model_v8_diagnostic_grid.png`.

Open question, not yet investigated: whether this is the state loss
being underweighted relative to grid/boundary loss once `grid_weight`
ramps up (by epoch 2, `sampling_p=1.0` so `grid_weight` is at full
strength and self-feed is 100% -- errors compound during self-fed
rollout regardless of which loss shaped training), the small
`vel_weight=0.1` letting velocity error accumulate positional drift
unchecked, or simply too few epochs/samples for the state signal to
dominate the learned behavior yet. Not yet tried: a state-loss-only
ablation (`grid_weight=0`, `boundary_weight=0`) to isolate whether
`token_state_loss` alone actually prevents vanishing, before assuming
the structural argument was wrong.

**Ablation run: `token_state_loss` in isolation (job 2852, `scripts/polaris_train_token_v9.sh`,
`grid_weight=0`, `boundary_weight=0`, `state_weight=1.0`, otherwise
identical recipe to job 2851) → `checkpoints/token_model_h12_v9.pt`.**
Answers the open question from job 2851 cleanly: **not** a
drowned-out-by-grid/boundary-loss problem -- with those terms fully
zeroed, the give-up-to-vanishing pattern is still there. Unbiased
review: early rollout (steps 0-5) tracks ground truth closely (small
~1-2 cell drift by step 5, and the model visibly smooths over a
close-contact/collision event around step 3 that ground truth renders
sharply). Late rollout diverges the same way as job 2851 -- by step 8,
2 of 4 objects have vanished; the dropout persists through step 20.
Surviving objects keep moving (not stalled) but drift further from
ground truth as steps increase.

New lead the reviewer surfaced, not present in prior job read-outs:
**dropout specifically targets the lowest-contrast/faintest ball
first**, consistently, across both this run and (re-reading job
2851's report) job 2851 as well. This points away from "the loss
function's incentive structure" as the sole explanation -- a
coordinate-space loss with no background to hide behind still lost
this ball, so the failure may be upstream: `find_token_positions`'
`detect_threshold` clipping a low-intensity ball's detection, or
`centroid_near`'s observation correction being systematically weaker
for low-signal tokens, either of which would starve that token's
hidden state of a usable observation and let it drift/decay under
self-feed regardless of what the training loss rewards. Not yet
investigated: check whether the same ball index is the one that
vanishes across multiple seeds/runs, and whether its rasterized PROB
peak is measurably lower than its siblings' at frame 0. Video:
`videos/token_model_v9_state_loss_only_rollout.mp4`, diagnostic grid:
`videos/token_model_v9_diagnostic_grid.png`.

**Detection-strength diagnostic (`scripts/diagnose_token_dropout.py`), v9
checkpoint, 48 seeds, 4 balls, 20 steps.** Instruments the rollout
directly (not just rendered video) to check the two open questions
from job 2852: does the same ball index vanish across seeds, and is
its frame-1 PROB peak measurably lower.

**Same-index hypothesis rejected**: dropout is spread across all 4
ball indices roughly evenly (0:2, 1:2, 2:3, 3:4 across 11 dropout
events / 48 seeds) -- no fixed "weak" token. Aggregate frame-1 peak
PROB by index is also flat (0.39-0.42 across all four), so there's no
structural per-index detection weakness either.

**Episode-relative peak rank is the real signal**: ranking each
episode's own tokens by frame-1 PROB peak, 6/11 dropout events are the
single *lowest*-peak token in that episode, and the mean rank across
all 11 is 1.09 (out of 0-3) vs. 1.5 expected under no effect --
whichever ball happens to be faintest at initialization in a given
episode is the one likely to disappear later, even though no ball is
intrinsically faintest across episodes. This matches the "faintest
ball first" read from job 2852's video review, but sharpens it: it's
about relative signal strength within an episode, not ball identity.

**Occlusion-gate hypothesis rejected**: within-episode nearest-neighbor
rank shows the *opposite* of what the occlusion theory predicts --
dropped tokens skew toward the *most isolated* token (nn_rank 2-3 out
of 3 most often), not the most contact-prone. `occluding_mask`
suppressing corrections near collisions is not what's causing this.

**Read**: weaker initial sub-pixel localization (lower rasterized PROB
peak → less precise `centroid_near` centroid at init) compounds under
self-feed until the token's predicted position drifts far enough that
`centroid_near`'s window goes empty (`total <= 1e-6` bailout in
`model/token_detect.py`), after which the token runs on unstable
`TokenDynamics` extrapolation alone with no way back. Points at the
observation-correction path itself -- either the `centroid_near` bail
condition (hard cutoff, no soft fallback/widening) or `TokenDynamics`
lacking any way to recover a token once its observation is lost --
rather than at loss weighting, which the last two ablations already
ruled out. Not yet tried: widen or soften the bailout window when a
token has drifted (re-detect from a larger search radius before giving
up), or an explicit per-token confidence signal the observation branch
could use instead of a hard threshold.

**Bailout confirmed directly (`--trace` flag added to
`diagnose_token_dropout.py`)**: instrumented `centroid_near`'s own
window-mass check step-by-step for all 11 dropout events from the
48-seed run above. Every one shows the same shape -- `window_total`
declines gradually over 3-10 steps (not a sudden jump) while mostly
unoccluded, crosses to exactly `0.0000`, and stays exactly `0.0000`
every step thereafter with zero recovery, all the way to step 18. The
1e-6 bailout in `model/token_detect.py` is confirmed as the absorbing
state: once a token's predicted position drifts far enough that its
own ball leaves the observation window, there is no mechanism -- no
widening search, no soft fallback -- that can ever bring it back, so
whatever caused the initial drift becomes permanent and total.

One outlier (seed 4773, ball 0) shows a second path into the same
trap: 11 consecutive steps flagged `occluded` (a persistent neighbor
keeping the correction gate shut) before window_total ever starts
declining -- chronic occlusion-gate suppression, not drift, starves
the correction and leads to the same permanent bailout.

Rules out "PROB genuinely vanishes while position is still tracked" --
the decline precedes and causes the bailout, not the reverse. Next:
prototype softening the bailout (progressively widen the search window
on decline, or drop the hard cutoff for a soft distance-weighted
fallback) rather than the current all-or-nothing threshold.

**Widened-search fallback implemented (`model/token_detect.py`,
`centroid_near`)**: on an empty window, retries with the search radius
grown by `ceil(radius)` per attempt (up to 3 expansions, capped
deliberately -- unbounded widening risks latching onto a different
ball's mass instead of recovering the token's own, worse than staying
lost). 89/89 tests pass unchanged (existing off-grid/empty-window
tests still hold: the widened window never reaches far enough to
contaminate those cases).

**Result on v9 checkpoint (no retrain, same 48-seed diagnostic,
`--trace` extended with ground-truth position error + swap
detection)**: dropout events drop 11 -> 8. But the trace reveals two
distinct mechanisms bundled under "give-up dropout", not one:

1. **Genuine recoverable bailout** (3 of 11 fixed: seeds 4744, 4767,
   one of the two 4773 events): position tracks ground truth closely
   (gt_err ~1-2 cells) right up to the window emptying: exactly the
   dead-end this fix targets, and it works.
2. **Dynamics divergence, untouched by this fix** (4/11 remaining:
   seeds 4742, 4750, 4752, 4757): `gt_err` climbs to 6-19 cells over a
   handful of steps *before* the window ever empties -- the position
   estimate is already badly wrong for reasons that have nothing to do
   with the observation window. Widening a search radius can't recover
   a token whose predicted position is already 10+ cells from its
   ball. The `SWAP->ballN` markers confirm these are real large
   divergences (position ends up nearer a *different* ball than its
   own), not swaps caused by the widening itself -- swaps only appear
   after gt_err was already huge.

**Read**: the bailout fix is real but addresses a minority of dropout
events. The larger remaining cause is `TokenDynamics` producing
`delta_pos` predictions that compound into double-digit-cell errors
within a handful of self-fed steps, independent of detection/occlusion
entirely. Next: investigate why `delta_pos` diverges this badly for
specific tokens -- likely candidates are the radius-graph attention
(`model/token_net.py`) losing a token once its predicted position
drifts outside `neighbor_radius` of all others (no neighbors to
attend to -> degenerate self-only update), or unconstrained
per-step displacement with nothing bounding `delta_pos` magnitude.

**Both candidate hypotheses falsified by direct instrumentation**:
traced `delta_pos`/`delta_vel` magnitude, radius-graph neighbor count,
and hidden-state norm per step for all 4 "dynamics divergence" seeds
(4742, 4750, 4752, 4757). `delta_pos` never exceeds ~0.5 cells/step --
nowhere near the 6-19 cell errors observed, ruling out unbounded
displacement. The "no neighbors -> degenerate" theory doesn't hold
either: `TokenDynamics.forward` adds a self-loop unconditionally
(`model/token_net.py:50-52`), so an isolated token (`nbrs=0`) still
gets a normal self-attention update, not a broken one -- confirmed by
reading the code directly, no ambiguity here.

**Actual mechanism: bad velocity at `init_tokens`, not bad dynamics**.
Comparing `init_tokens`' finite-difference velocity estimate against
ground-truth velocity at frame 1 for the same 4 seeds shows large,
confident errors that survive the existing `max_init_speed=20`
rejection (which only catches literally-impossible speeds): e.g. seed
4752 token0 estimated `[-4.97, -6.15]` vs. true `[2.41, -1.98]` --
wrong in both magnitude and direction, but at ~7.9 cells/s, comfortably
under the 20 cells/s cap. With `delta_pos`/`delta_vel` both small per
step (confirmed above) and no explicit velocity-correction mechanism
(by design, see `TokenModel.step`'s docstring -- job 2851 already
showed adding one via loss doesn't fix this in practice), a bad initial
velocity just dead-reckons forward almost unchanged, producing exactly
the smooth, steady position-error growth seen in the trace (`window_total`
stays healthy the whole time -- the observation branch is actively
"correcting" every step, just not enough to counter a large constant
velocity bias). This is confirmed directly: seeds with small init
velocity error track cleanly to gt_err < 0.5 by step 18; seeds with
large init velocity error are exactly the 4 that diverge.

**Root cause of the bad velocity estimate**: `init_tokens` paired each
frame-1 detection to its nearest frame-0 detection independently per
row (plain `argmin`), with no uniqueness constraint. Two failure modes
found directly in the data:
1. **Duplicate match** (seed 4750): two frame-1 tokens both nearest-
   matched to the *same* frame-0 detection (ball 1's true frame-0
   position), leaving the other frame-0 detection (actually a second,
   genuine peak from ball 1's own blob splitting into two under the
   NMS window at close range -- a separate, undiagnosed peak-detection
   bug, not yet fixed) unclaimed. One of the two duplicate tokens gets
   a real velocity, the other whatever's left.
2. **Near-tie ambiguity** (seed 4742, 4752, 4757): when two frame-0
   balls are close together, a frame-1 ball's nearest frame-0 detection
   can be a close call between its true predecessor and the other
   ball's -- confirmed directly from the distance matrices (e.g. seed
   4752: `[1.19, 1.31, ...]`, a ~10% margin), and independent per-row
   argmin has no way to know it picked wrong.

**Fix (`model/token_model.py`, `_assign_velocity_pairs`)**: replaced
per-row `argmin` with an optimal one-to-one assignment (`scipy.optimize
.linear_sum_assignment`, added to `requirements.txt`), which prevents
mode 1 outright (two rows can no longer claim the same column). Added
an explicit ambiguity check for mode 2: a row's assigned distance must
beat its own next-best candidate distance by >= 0.5 cells, or the
velocity is rejected (zeroed) the same way the existing
`max_init_speed` check already does -- consistent with the file's own
stated philosophy (zero is recoverable, confidently wrong is not).
3 new unit tests on `_assign_velocity_pairs` plus one integration test
on `init_tokens`; 93/93 tests pass.

**Result on v9 checkpoint (no retrain, same 48-seed diagnostic)**:
dropout events drop 8 -> 6 (11 -> 8 -> 6 across the two fixes so far).
All 4 previously-traced "dynamics divergence" seeds' initial velocity
estimates are now either correct or safely zeroed (no more
confidently-wrong cases). The still-open duplicate-peak-detection bug
noted in mode 1 above (`find_token_positions` producing 2-3 peaks for
what should be one ball at close range, reproduced independently while
writing this fix's test) is the most likely source of some remaining
dropout events and is the next thing to investigate -- not yet
root-caused or fixed.

**Duplicate-peak bug has (at least) two distinct causes, one now fixed
(`model/token_detect.py`, `_suppress_tied_peaks`)**: reproducing it
directly (a single ball at x=10.5, radius=1.5) showed the raw
`prob == pooled` non-max suppression doesn't break exact ties -- a
ball splatting equal mass onto the two cells it straddles registers
both as separate peaks. Fixed by keeping only the highest-value peak
within `radius` of any other candidate (deterministic tie-break, same
merge-distance philosophy the function already documents for
close-together balls). New regression test confirms this: fails
(2 peaks) before the fix, passes (1) after. 94/94 tests pass.

This does **not** explain seed 4750's remaining duplicate, though:
re-inspecting its *raw* (pre-`centroid_near`) peak coordinates shows
4 well-separated cells (`(11,18) (13,7) (15,9) (18,11)`, all >2 cells
apart -- no tie, no near-tie), each of which correctly corresponds to
a distinct true ball (`(13,7)` -> ball 3 at true (13.24, 6.69), 0.39
cells off; `(15,9)` -> ball 1 at true (14.95, 9.08), 0.09 cells off).
Initially misread this as a splat-kernel tail artifact -- wrong.
`splat_ball` (`bounce.py`) is a strictly monotonic linear-falloff disk
bounded to radius 0.75 from its own center, so a single ball cannot
produce mass 2+ cells from itself; `(13,7)`'s value can only come from
ball 3's own splat. Checked directly.

**Actual mechanism, confirmed by printing the window
`centroid_near` reads from `(13, 7)`**: `centroid_near`'s window
(`ceil(radius + margin)` = `ceil(0.75 + 1.0)` = 2 cells) extends far
enough from `(13, 7)` to reach `(15, 9)` -- 2.83 cells away, inside
the window's 2√2 ≈ 2.83-cell diagonal reach. Ball 3's own mass in
that window totals ~0.4, but ball 1 (closer to its own integer grid
cell, hence a higher post-saturation value at `(15,9)`: 0.58) sits
right at the window's corner and dominates the intensity-weighted
centroid, pulling `(13,7)`'s refined position onto ~(14.2, 8.2) --
right on top of ball 1's own refined detection, silently erasing
ball 3's. This is exactly the "neighbour contaminates the centroid
readout" scenario `TokenModel._gate_radius`'s docstring already
describes and the occlusion gate already exists to prevent -- just
not applied here, because `find_token_positions`' peak-refinement
step (used only at init) has no gate at all, unlike per-step tracking.

**Fix (`model/token_detect.py`, `find_token_positions`)**: gate
`centroid_near` refinement on the same `occluding_mask` threshold
`TokenModel._gate_radius` uses at tracking time, evaluated on the
raw (pre-refinement) peak coordinates. A peak flagged as occluding by
a neighbour keeps its coarse integer-cell coordinate instead of being
refined -- less precise, but immune to hijacking, the same tradeoff
the tracking-time gate already accepts. Regression test reproduces the
real seed-4750 pair directly via `bounce.splat_all` (not synthetic):
fails (one ball's detection lost) before the fix, passes after.
95/95 tests pass.

**Result on v9 checkpoint (no retrain, same 48-seed diagnostic)**:
seed 4750 confirmed fixed directly (both balls now detected
independently: `(13,7)` and `(15,9)`, no longer collapsed). Aggregate
dropout count unchanged at 6/48 -- seed 4761 newly crosses the
dropout threshold, offsetting seed 4750's fix. Given this session's
run of real, verified mechanism fixes (11 -> 8 -> 6 -> 6, each
individually confirmed against its own root cause), the flat aggregate
here looks like normal seed-to-seed variance in a 6/48 tail rather
than a sign this fix didn't work -- worth re-running the full 48-seed
sweep after the next fix to see if the trend resumes, rather than
reading too much into one flat step.

**Diagnostic tool itself was stale, fixed
(`scripts/diagnose_token_dropout.py`, `window_total_at`)**: it only
ever read the *base* (un-widened) window, so every `BAILOUT` label in
the trace output since the centroid_near widening fix (two commits
back) was potentially a false positive -- the real `centroid_near`
call the model uses tries up to 3 widening expansions before actually
giving up. Updated `window_total_at` to mirror that same widening loop
so `BAILOUT` means what it says again.

**Re-traced all 6 remaining dropout seeds with the corrected
diagnostic**: all 6 are genuine bailouts even under the full widened
search -- `window_total` declines gradually over several steps (never
a sudden jump) before settling at exactly 0.0 and staying there for
the rest of the rollout, the same absorbing-state shape documented
in the original "Bailout confirmed directly" entry, just now a
smaller residual set after tonight's fixes removed the init-time
contributions (bad velocity, tied/hijacked peaks) that were inflating
it. This matches, not contradicts, job 2851/2852's conclusion: a
loss-based fix (state-space loss) didn't resolve this gradual-decline
mechanism in practice, because the actual defect is in how
`TokenDynamics` + the observation correction jointly track a token
once its predicted position starts drifting, not something an
inference-time detection/matching patch can reach. Three real,
independently-verified bugs were fixed tonight (11 -> 6 dropout
events across 48 seeds); this residual is the same open architectural
question the session already flagged before these fixes started, and
is not resolved by anything of this shape -- stopping the inference-
time patch search here rather than continuing to chase a 6/48 tail
that needs an actual dynamics/training change.

## 2026-09-23 — where rollout error actually compounds: velocity, not position, and training a fix for it doesn't close the gap

Direct instrumentation (`scripts/measure_error_compounding.py`, v9
checkpoint, 48 seeds): compares teacher-forced (positions/velocities
reset to ground truth every step before calling `model.step`) against
standard self-fed rollout, same seeds, same model. Teacher-forced
per-step position error stays flat and small (0.2-0.9 cells,
non-growing) across all 19 steps -- one dynamics step, given a correct
input, is essentially fine. Self-fed position error grows
monotonically from 0.48 cells (step 0) to 5.9-6.0 cells by step
10-11, then plateaus/slightly recedes. Self-fed **velocity** error
grows even faster in relative terms: 1.70 cells/s (step 0) to 9.4
cells/s (step 9), tracking the position-error curve's shape almost
exactly. `TokenModel.step`'s own docstring already says velocity is
"corrected only implicitly" (no per-step observation correction the
way position gets one via `centroid_near`) -- this measurement
confirms that asymmetry is load-bearing: velocity drifts hard under
self-feed, and `final_pos = corrected_pos + velocities * dt +
delta_pos` carries that drift straight into position every step.

**Naive inference-only fix rejected by direct test.** Added a second,
independent implementation of TokenModel.step
(`scripts/probe_velocity_correction.py`) that re-anchors velocity each
step via finite difference of two consecutive `centroid_near` reads
(mirroring `init_tokens`' own frame0/frame1 approach, applied every
step instead of only at init), spliced onto the frozen v9 checkpoint
with no retraining. Result: monotonically **worse**, not better, at
every velocity_weight tested (0.3/0.5/0.8/1.0) -- position error at
step 16 goes from 5.36 (baseline) to 6.73 (weight=1.0). Read: the
frozen dynamics network's `delta_vel` implicitly depends on velocity
following its own learned internal trajectory; splicing in an
external, noisier estimate is off-distribution for weights that were
never trained to expect it.

**Built the correction into the real architecture and retrained --
still didn't help.** `model/token_model.py`'s `TokenModel.step` now
takes this velocity correction as a first-class, opt-in parameter
(`velocity_weight`, default 0.0, backward compatible -- 97/97 tests
pass with the refactor). Trained job 2854 (`checkpoints/token_model_h12_v10.pt`):
identical recipe to v9 (job 2852: state-loss-only ablation, 10k cache,
3 epochs) with `velocity_weight=0.5` as the only change, so the
dynamics network sees the correction from the start instead of having
it spliced on afterward. Result on the same 48-seed diagnostic (v9 ->
v10): self-fed position error curve is nearly unchanged (5.45 -> 5.22
cells at step 9, 5.50 -> 5.55 at step 18 -- within seed-to-seed
noise), and dropout count is actually worse (6 -> 10 events across 48
seeds). Training the correction in, not just splicing it on, ruled
out the "off-distribution" explanation for the earlier negative
result, but the underlying fix still isn't there: this specific
mechanism (explicit two-frame velocity re-anchoring) is not the answer
to the compounding problem, at least not at this weight/this little
training. Not yet separated: whether 0.5 is the wrong weight, whether
gating it on occlusion the same way position is gated is too
conservative here, or whether the real fix has to be architectural in
a different way (e.g. the dynamics network needs velocity error fed
back as an explicit residual/loss term rather than a blended
correction, or the give-up/absorbing-state mechanism itself needs
fixing directly rather than trying to prevent the drift that leads to
it).

**Also running: job 2855 (`checkpoints/token_model_h12_v11.pt`),
`velocity_weight=0.0` (v9's own recipe) but epochs 3 -> 15,
ramp_epochs 2 -> 10** -- tests a separate, previously untested
confound: v6-v9's "large dataset, few epochs" recipe was an explicit,
never-validated deviation from the project's own default training
budget (`docs/debugging/experiment-log.md`'s job 2846/2847 entry:
"applied the user's LLM-pretraining-style intuition... since no
grokking signal had been observed to justify heavy repetition"). 3
epochs on 10k samples is 30k total gradient steps, fewer than the
project default recipe's 2000 samples x 50 epochs = 100k -- the
"confirmed architectural" conclusion in the 2026-09-22 entries above
was drawn from a checkpoint trained on a third of the default's total
updates.

**Result: more epochs alone makes it WORSE, not better.** Job 2855
(`checkpoints/token_model_h12_v11.pt`, 15 epochs / 150k gradient steps,
5x v9's budget) on the same 48-seed diagnostic: self-fed position
error is flat-to-slightly-worse vs v9 at every step (6.15 vs 5.50
cells at step 18; 5.18 vs 5.45 at step 9 -- within noise), teacher-
forced error is unchanged (still flat, 0.2-0.9 cells). But dropout
count is dramatically worse: **30/48 seeds** (8/7/8/7 by ball index),
5x v9's 6/48. Rules out "v9 was just undertrained" cleanly -- this
isn't a training-budget confound, more training on this exact recipe
actively teaches the give-up-to-vanishing shortcut harder, consistent
with `occupancy_weighted_mse`-style give-up incentives job 2840
originally found, except here even with `grid_weight=0` and
`token_state_loss` as the only real signal, more optimization against
that objective still finds a way to make more tokens disappear. Read:
whatever objective/architecture combination is being optimized here
has "some balls vanish" as a *better*, not worse, local optimum once
given more steps to find it, at least for a nontrivial fraction of
episodes.

**Job 2856 (`checkpoints/token_model_h12_v12.pt`), `state_vel_weight`
0.1 -> 1.0 (token_state_loss weights velocity error equally with
position instead of 10x down-weighted)**: still running as this entry
is written; result appended below once known.

**Three single-variable fixes tried tonight, all motivated by direct
measurement rather than guessing, none of them helping so far**:
explicit trained-in velocity correction (v10: dropout 6->10/48,
position error flat), 5x more training on the identical recipe (v11:
dropout 6->30/48, position error flat-to-worse), and the vel_weight
result pending (v12). Per this project's own systematic-debugging
discipline: v10 and v11 are two independent, clean negative results
against the same symptom (self-fed give-up dropout), both worse than
baseline, not just "no better" -- that's already the signal to stop
proposing more minor-variant patches of this shape and ask whether the
symptom itself is being mis-modeled. Candidate reframing, **not yet
tested**: this may not be a "correction" problem (nudge position/
velocity back toward truth) at all -- it may be that `token_state_loss`
+ self-feed, optimized harder, is discovering that letting a token's
window go empty and freezing/drifting is cheaper than continuing to
track a hard-to-observe ball, i.e. the model is finding a genuine
local optimum of the stated objective, not failing to reach one. If
so, the fix isn't a better correction mechanism (position or velocity)
at all -- it's removing the incentive to vanish, e.g. an explicit
penalty for `window_total` going to zero (currently nothing in the
loss ever looks at detection/observation strength, only ground-truth
state error), or making token permanence itself a structural
guarantee rather than an emergent property the network has to learn
not to violate.

**Job 2856 result (`checkpoints/token_model_h12_v12.pt`,
`state_vel_weight` 0.1 -> 1.0): also worse, not better.** Same 48-seed
diagnostic: self-fed position error flat-to-worse vs v9 (6.27 vs 5.50
cells at step 18), teacher-forced error unchanged (still flat, 0.2-0.9
cells) -- confirms yet again that single-step dynamics accuracy was
never the issue. Dropout count **27/48** (9/5/7/6 by ball index),
4.5x v9's 6/48. Weighting velocity error 10x harder in the direct
state loss didn't teach better velocity tracking under self-feed; it
made more tokens vanish, similar in kind (if not quite degree) to v11.

## Synthesis: three single-variable fixes tried tonight, all negative -- stopping here, this needs a human call

Starting point was the user's own question: single-step (t -> t+1)
prediction looked "nearly perfect," so where does rollout error
compound? Answered that cleanly and for real
(`scripts/measure_error_compounding.py`): teacher-forced per-step
position error is flat and small (0.2-0.9 cells) across every step
tested, on all four checkpoints trained tonight, no exceptions --
single-step dynamics genuinely is fine, every time. Self-fed velocity
error is where it actually compounds (1.7 -> 9+ cells/s within 9
steps), and position error tracks it in lockstep through
`final_pos = corrected_pos + velocities * dt + delta_pos`.

Three independent, single-variable attempts to fix that, each
isolated against the v9 baseline (same recipe, one change each), each
directly motivated by that measurement rather than guessed:

| job | change | self-fed pos err (step 18) | dropout / 48 |
|---|---|---|---|
| v9 (baseline) | -- | 5.50 | 6 |
| v10 | trained-in explicit velocity re-anchoring (`TokenModel.step`, `velocity_weight=0.5`) | 5.55 | 10 |
| v11 | 5x training (epochs 3->15) | 6.15 | 30 |
| v12 | `token_state_loss`'s `vel_weight` 0.1->1.0 | 6.27 | 27 |

All three are flat-to-worse on position error and *categorically*
worse on dropout count -- not noise, 1.7x to 5x more vanishing tokens
than baseline in every case. Per this project's own
systematic-debugging discipline (`superpowers:systematic-debugging`):
3+ fix attempts at the same symptom, each revealing the same failure
in a different place rather than closing it, is the signal to stop
patching and question the architecture/objective itself, not attempt
a 4th minor variant. Stopping here rather than launching a v13.

**What's now conclusively ruled out**: v9's checkpoint being
undertrained (v11 -- more training makes it worse); the specific
mechanism of an explicit, blended velocity correction, whether spliced
on at inference (earlier tonight, also negative) or trained in from
the start (v10); and naively reweighting the existing loss to
penalize velocity error harder (v12). None of these are "didn't help
enough" -- all three measurably increased dropout relative to
baseline, which is a real, reproducible direction, not sampling noise.

**What remains genuinely untested, and is the most promising next
direction**: the reframing already noted above and in
[[project_token_per_ball_model_status]] -- that give-up dropout may be
a real local optimum of `token_state_loss` + self-feed, not a failure
to reach a better one. Nothing in the current loss ever looks at
`window_total`/detection strength; a token can drift its
`centroid_near` window empty and the training signal has no way to
distinguish "recoverable drift" from "acceptable to abandon." Two
concrete, not-yet-tried directions that follow from that: (a) an
explicit loss penalty when a token's own `window_total` collapses
toward zero, so vanishing is never free regardless of what it does to
state-space MSE; (b) treating token permanence as structural rather
than emergent -- e.g. hard-constraining `centroid_near`'s widened
search to never fully give up (no "return position unchanged" branch)
so there is no absorbing state for training to discover as a shortcut
in the first place. Both are more invasive than tonight's A/Bs (loss
function redesign vs. hyperparameter/mechanism swap) and are exactly
the kind of change that deserves a human decision before spending more
GPU-hours chasing it -- flagging here rather than guessing which one
to build unattended.

**Unbiased visual review** (`videos/token_model_v{9,10,11,12}_night_diagnostic_grid.png`,
fresh subagent, no hypothesis primed, per
[[feedback_auto_run_video_analysis_on_sim_finish]]) corroborates the
quantitative numbers: all four checkpoints track ground truth closely
through step 3, then lose distinct blobs (not blur/streak/checkerboard
-- surviving blobs stay crisp) starting around step 5. Ranked by blob
count surviving to step 20: v9 (3 blobs, most spread out) > v11 (3,
but bunched into one overlapping cluster) > v10 (2) > v12 (1, worst --
collapses to a single isolated blob by step 5 and never regains a
second one). This matches the dropout-count ranking (v9=6 < v10=10 <
v12=27 < v11=30) for v9/v10/v12, but v11's "3 blobs by step 20" visual
read sits oddly against it having the *worst* dropout count of the
four -- likely explained by the reviewer counting undifferentiated
blob clusters, not per-token identity: `diagnose_token_dropout.py`
matches each token to its ground-truth ball once and tracks that
specific identity, so several tokens collapsing onto nearly the same
position (which the trace already shows happening via `SWAP->ballN`
markers in earlier entries) would read as "multiple blobs" visually
while still counting as multiple individual dropouts numerically. Net
effect: the visual review doesn't overturn the quantitative ranking,
but adds a concrete additional failure shape worth remembering -- v11
in particular may be converging tokens onto each other (identity
collapse) rather than only vanishing them independently, which the
existing instrumentation doesn't directly measure and would be worth
checking before any future work on this line.

## 2026-09-23 (12:46) — v13 (`window_collapse_loss`, job 2857): worse than all four previous checkpoints, and by a mechanism that confirms the v11 identity-collapse suspicion

Implemented the untested direction (a) from the v9-v12 synthesis above:
an explicit loss penalty (`window_collapse_loss`, `model/token_losses.py`)
when a token's own rasterized PROB mass near its tracked position falls
below a floor, reading the step's own `pred_grid` (not the detached
self-fed `observed_frame`) so gradient flows through the same forward
pass that produced both. 8 new tests, 105/105 passing before launch.
v13 (job 2857): identical recipe to v9's state-loss-only ablation plus
`--collapse-weight 0.5 --collapse-floor 0.3`.

**Result: dropout 34/48 — worse than v9 (6), v10 (10), v12 (27), and
v11 (30). The single worst checkpoint produced across all five
attempts.** Self-fed position error at step 18: 6.35 cells (vs v9's
5.50); teacher-forced error stays flat/small (0.33-0.79 across all 20
steps) -- the now-repeated finding that single-step dynamics accuracy
is never the problem holds a fifth time.

**Training-log NaN, root-caused and fixed, unrelated to the A/B
result**: the job's log showed intermittent `loss=nan` in per-epoch/
per-window prints (3 occurrences across 30k steps). Traced (not
guessed) to a pre-existing bug in `boundary_loss`: a rare zero-token
rollout step (`init_tokens` detects no balls) hits `.mean()` on an
empty tensor, which is NaN in PyTorch, and `0.0 * nan` is still nan --
so this leaked into logs even at this recipe's `boundary_weight=0.0`.
Confirmed harmless to the actual trained weights before concluding
anything from the run: backward through a zero-cardinality path can't
multiply a real nan into any parameter's gradient (verified both by
direct PyTorch experiment and by checking the downloaded checkpoint's
state_dict for NaN/Inf -- none found). Fixed with an empty-token guard
matching `token_state_loss`'s and `window_collapse_loss`'s existing
pattern, regression test added (106/106 passing), uploaded to Polaris.
This bug predates v13 and almost certainly affected v9-v12's logs too
(their logs have since rotated off Polaris and couldn't be checked),
but never affected any of those checkpoints' actual weights for the
same zero-cardinality reason.

**Unbiased visual review** (`videos/token_model_v13_diagnostic_grid.png`,
fresh subagent, no hypothesis primed): blob count and fidelity match
ground truth through step ~3, matching v9-v12's read. From step 5
onward the failure mode is **not** vanishing to an isolated blob (v9's
and v12's shape) but **merged, blended multi-color patches at object
boundaries** -- by step 12, "several balls have collapsed into a
single multi-colored patch"; by step 20, a "merged yellow+teal
checker-like patch (two adjacent mismatched-color cells) rather than a
clean single-color object." No frame-wide checkerboard/streaking --
the artifact is localized to where balls overlap or pass close to each
other.

**This is the mechanism, not just a worse number, and it directly
confirms the v11 suspicion flagged at the end of the last synthesis**:
`window_collapse_loss` penalizes a token for having *no* mass nearby,
but doesn't care *whose* mass satisfies it. The cheapest way to escape
the penalty when a token's own ball is hard to track isn't to keep
tracking it better -- it's to drift onto a *neighboring* ball's mass,
which is already-existing, real, non-zero PROB the penalty is happy to
accept. That produces exactly the observed failure shape: two tokens
converging onto one ball's mass (merged patch, wrong color mixing)
instead of one token vanishing cleanly. The fix this A/B was designed
around made the underlying incentive problem worse, not better: it
didn't just fail to reward correct tracking, it rewards a specific
wrong behavior (identity theft) that wasn't as strongly incentivized
before.

**Four single-variable attempts at this symptom (v10, v11, v12, v13),
each in a different loss/mechanism location, all worse than baseline,
one now actively worse by a wide margin with a distinct and understood
failure mode.** Per this project's own systematic-debugging discipline,
this is well past the "stop patching, question the architecture"
threshold flagged after v10-v12 -- reinforced now, not resolved, by
v13. Direction (b) from the earlier synthesis -- removing the give-up
branch structurally (no "return position unchanged" absorbing state in
`centroid_near`) -- remains the only genuinely untested direction, but
v13's failure mode is a specific warning for it too: any fix in this
family needs to penalize (or structurally prevent) a token latching
onto a *neighbor's* mass, not just penalize having none. Flagging for
human decision before further GPU-hours on this line, per the same
judgment call made after v12.

## 2026-09-23 (14:29) — v14 (territory masking, job 2858): fixes v13's identity-theft mechanism but dropout count much worse (66 vs v9's 8)

Implemented the architecture-redesign direction chosen by the user after
the v13 synthesis (`docs/superpowers/specs/2026-09-23-token-territory-masking-design.md`,
`docs/superpowers/plans/2026-09-23-token-territory-masking.md`): a
Voronoi territory mask (`model.token_detect.territory_mask`) over all
currently-tracked token positions, applied to `centroid_near`'s and
`window_collapse_loss`'s window reads so a token's observation can never
include mass closer to a different tracked token. `centroid_near`'s
give-up search widened `max_expansions` 3->6 (`TokenModel.__init__`), now
considered safe since a wider search can no longer cross into a
neighbor's territory. 8 new unit/integration tests (119/119 passing
before launch). v14 (job 2858): identical recipe to v9 (state-loss-only
ablation, 3 epochs, same cached dataset) -- single-variable A/B, no
change to epochs or dataset size.

**Result: dropout 66 events across 48 seeds (ball-index breakdown {0: 18,
1: 16, 2: 16, 3: 16}) vs. v9's 8 (re-measured fresh with the identical
script/environment for a fair comparison, not the previously-logged 6 --
same ballpark, treated as the current baseline). Categorically worse
than v9, and worse than v13's 34/48 too.**

**Mechanism confirmed by the qualitative failure shape, not just the
number.** Unbiased subagent review of the diagnostic grid
(`videos/token_model_v14_diagnostic_grid.png`, no hypothesis primed):
balls track well through step 2, then vanish one by one with no blur or
color bleed -- "the missing balls don't leave a smeared or blended
trace, they just disappear... surviving blobs stay fairly crisp." This
is qualitatively different from v13's failure (merged multi-color
patches at ball boundaries) and confirms the identity-theft mechanism
this design targeted is actually gone -- no evidence of one token's
mass being stolen by another. But clean vanishing is now happening far
more often than under the unmasked v9 baseline, not less.

**Leading hypothesis (not yet verified): territory boundaries are drawn
between *tracked* positions, which are themselves imprecise, not between
true ball centers.** When two tracked tokens are near each other but
neither is exactly centered on its own ball (normal tracking noise, not
a bug), the Voronoi split can fall in the middle of one token's own real
mass, handing part of its own ball's splat to the neighbor's territory
and starving both windows below `floor`/the total-mass check -- turning
ordinary tracking jitter into a for-real dropout that would have
resolved itself unmasked (the unmasked window would have simply summed
the whole nearby mass, correctly, most of the time). This would explain
why the count rose roughly evenly across all four ball indices (not
concentrated on one) and why widening `max_expansions` (intended to
help) didn't offset it -- a wider search still can't cross a wrongly-
placed territory boundary, so widening only helps the give-up case this
design already handles, not the new one it may have introduced. Not yet
verified against the actual per-step territory-mask/window-mass trace;
flagging as the leading explanation, not a confirmed root cause.

**Per this project's systematic-debugging discipline and the plan's own
validation step 5: not chaining another fix onto this result.** The
territory-masking design achieved its stated goal (no more cross-token
mass theft) but introduced a larger regression by a different, unverified
mechanism. This is a new decision point for the human, not a "closer but
not there yet" -- the architecture-redesign path itself needs
reassessment (e.g., a softer/probabilistic territory boundary instead of
a hard nearest-token partition, or reverting to unmasked windows for
low-confidence/high-uncertainty tracked positions) before spending more
GPU-hours on this line.

## 2026-09-23 (14:45) — v14 final review: root cause confirmed, fix applied, v9 baseline number corrected

Dispatched a fresh reviewer (Opus, no prior context) for the standard
whole-branch review before merging the territory-masking branch. It
found and directly verified the actual mechanism behind v14's
regression, superseding the "leading hypothesis" (Voronoi split through
a token's own mass) logged above -- that hypothesis does not hold at
inference; the real mechanism is simpler and more fundamental:

**Under self-feed, a hard territory mask has no recovery path once a
token's own rendered mass leaves its own territory (e.g. gravity carries
it off-grid).** Before this fix existed, an unmasked widened search
could still grab a *neighbour's* mass and drag the token's tracked
position back onto the grid -- the "identity theft" this design targeted
was, empirically, also the only mechanism that ever recovered an
off-grid token. Masking closed that path with nothing to replace it, so
every token that ever falls off-grid under self-feed now stays lost
forever instead of occasionally recovering via a wrong-neighbour
correction. The reviewer confirmed this directly: 62 of v14's 66 dropped
tokens end off-grid (x≈20-25 on a 20-cell grid, consistent with gravity),
not on-grid and merged/vanished as the earlier hypothesis assumed.

**The logged "v9 baseline = 8" number was also measured under the wrong
code.** That re-measurement used v9's *weights* but v14's *inference
code* (masking + `max_expansions=6`). Under v9's own original inference
settings (no masking, `max_expansions=3`), the reviewer measured 6 --
matching the originally-logged number. Most of the v9(6)->v14(66) gap
reflects what the network *learned* under masked training, not what
masking does at inference time alone; the reviewer's inference-only
ablation (same v9/v14 weights, masking and `max_expansions` varied)
found masking makes both checkpoints' *inference* worse on their own
(v9: 6->8, v14: 48->66), and that widening `max_expansions` without
masking *helps* both (v9: 6->3, v14: 48->32) -- i.e. Task 6's widening
was a real, verified improvement on its own; it only became a no-op once
combined with a mask that had no fallback.

**Fix applied** (commits `e72f790`, `f7076b2`): `centroid_near` now
retries unmasked when the masked search finds nothing anywhere (own
territory completely empty at every expansion) before giving up --
restoring the old recovery path only as a last resort, so a token that
still has any of its own mass is still protected from a neighbour's
(the property masking exists for), but a token with genuinely nothing
left of its own is no longer stuck. `TokenModel.territory_masking`
defaults to `False` (was implicitly `True` with no way to disable) --
existing checkpoints, including v9, now get byte-for-byte unchanged
inference unless explicitly opted in via `--territory-masking`
(`token_train.py` and all token-model render/diagnostic scripts).
`scripts/diagnose_token_dropout.py`'s `--trace` window-mass mirror was
also stale (no masking, hardcoded `max_expansions=3`) and has been
corrected to match whatever settings the model under test actually uses.

**Not yet retrained with the fix.** v14's checkpoint was trained under
the old (fallback-less) masked code and is not representative of what
the corrected design would learn -- v15 (fixed code, `--territory-masking`
enabled, otherwise identical v9 recipe) is the real test of whether
territory masking helps once it can no longer strand a token
permanently. See the next entry for that result.

## 2026-09-23 (15:20) — v15 (fixed territory masking + fallback, job 2859): regression eliminated, but still worse than the unmasked baseline

`token_model_h12_v15.pt` (job 2859): identical recipe
to v9, plus `--territory-masking` (now the opt-in flag, using
`centroid_near`'s unmasked-fallback fix from the previous entry).
Training completed cleanly (no NaN, loss curve matches v9/v14's family).

**Re-verified v9's true baseline under the now-fixed default (no
masking) code: 3/48** -- lower than both the originally-logged 6 and the
previously-reported 8 (which the last entry already explained was
measured under always-on masking with no way to disable it). 3 is the
correct number to compare against now that `territory_masking` defaults
to `False` and the diagnostic script's own `window_total_at` mirror was
also fixed to match.

**v15 result: 16/48 dropout with `--territory-masking` enabled** (ball
breakdown `{0: 4, 1: 3, 2: 3, 3: 6}`) -- categorically better than v14's
66 (the regression from the fallback-less bug is gone), but still worse
than v9's unmasked 3.

**Unbiased subagent review of the diagnostic grid**
(`videos/token_model_v15_diagnostic_grid.png`, no hypothesis primed):
blobs track well for the first 1-2 steps, then "progressively blur and
bleed into neighbouring objects when two blobs come close together,
sometimes averaging into an intermediate color," with the apparent
object count dropping below ground truth's steady 4 by mid-rollout; no
checkerboarding or high-frequency noise. This is a *third* distinct
failure shape across the three checkpoints tried this line: v13
(uncontrolled theft, merged multi-color patches), v14 (clean vanishing,
tokens fall off-grid with no recovery), v15 (color-bleed at close
approach, milder than v13 but the same family) -- consistent with the
unmasked fallback re-engaging exactly when two tokens are near each
other and one's territory happens to empty out, which is also the
highest-risk moment for a real collision.

**Read: territory masking, even correctly implemented, does not net
help this metric at the current recipe (3 epochs, state-loss-only).**
The fallback that fixes v14's permanent-loss regression is also what
reopens a milder version of the original identity-blending problem,
specifically at the moment (close approach) the whole design exists to
protect. This is not a bug to chase further with another single-variable
tweak -- three checkpoints (v13, v14, v15) have now each surfaced a
different failure mode from a different point in the same design space
(loss penalty, hard mask, masked+fallback), all worse than the unmasked
v9 baseline on this metric. Per this project's systematic-debugging
discipline, flagging for a human decision on whether to continue
investing in the territory/masking family of fixes, revisit the
loss-space direction instead, or treat v9 as the working baseline while
this line stays parked.

Checkpoints and videos: `checkpoints/token_model_h12_v15.pt`,
`videos/token_model_v15_diagnostic_grid.png`,
`videos/token_model_v15_rollout.mp4`.

## 2026-09-23 (21:06) — v17 (track-query attention, job 2873): near-parity with v9, and a structurally different failure mode from the whole masking family

`token_model_h12_v17.pt` (job 2873, Polaris): identical recipe to
v9/v15 (state-loss-only ablation, `--dataset-cache
checkpoints/token_dataset_10000_h12_seed4738.pt`, 3 epochs, ramp-epochs
2) plus `--track-query` -- the two-stage self-/cross-attention
architecture designed in
`docs/superpowers/specs/2026-09-23-token-track-query-design.md`, which
replaces `occluding_mask`/`centroid_near`/blending entirely for this
path instead of tuning that structure further (v10-v12) or adding a
hard mask on top of it (v13-v15).

**Implementation bug found and fixed before this run produced a valid
checkpoint.** The first submission (job 2868) crashed at epoch 0, batch
6800 with `RuntimeError: element 0 of tensors does not require grad and
does not have a grad_fn`. Root cause: `TrackQueryDynamics.forward`'s
zero-token early return built its output tensors via bare
`positions.new_zeros(...)`/`hidden.new_zeros(...)`, bypassing
`gru`/`delta_head` entirely -- unlike `TokenDynamics`, which always
routes its zero-token output through both regardless of token count and
so always keeps a `grad_fn`. Any training batch whose rollout sample had
zero detected tokens at every step (rare but not impossible with this
dataset's 2-6 ball range) produced a total loss with no gradient path,
which crashed `loss.backward()`. Fixed by letting `n==0` flow through
the same self-attention/cross-attention/gru/delta_head path as every
other token count, only special-casing the `torch.stack`-on-empty-list
crash for the cross-attention loop itself (commit `b0af379`). Neither
the design's own review-focus tests nor the final code review's manual
n=0 probe caught this -- both checked shapes and NaN-freedom, not
gradient-graph connectivity under an actual training loss. Resubmitted
as job 2873, completed cleanly (26 min, no further errors).

**v17 result: 4/48 dropout** (ball breakdown `{1: 2, 2: 1, 3: 1}`) --
just above v9's unmasked baseline of 3/48, technically not a win against
the design's stated primary success criterion ("dropout count below
v9's 3/48"), but dramatically better than every prior structural fix
attempt this investigation tried: v13 (34/48), v14 (66/48), v15
(16/48). At 48 seeds the 3-vs-4 gap is not distinguishable from noise on
its own.

**Unbiased subagent review of the diagnostic grid**
(`videos/token_model_v17_diagnostic_grid.png`, no hypothesis primed):
tracks closely with ground truth through step 3, starts diverging by
step 5, and by step 20 shows objects "drifting to different positions
than ground truth and appearing to sit closer together or partially
overlapping compared to the more spread-out ground-truth layout, with
colors... staying consistent throughout rather than fading" -- no
checkerboarding, no color bleeding, no merging. This is a *different
failure shape* from the entire v13-v15 masking family, whose defining
symptom in every case was color-blending/identity-theft at close
approach. v17's failure looks like plain positional drift compounding
over the rollout, colors and identity staying intact -- consistent with
what v16 (memory: "v16 verified 0/16 dropout, positional drift not
identity dropout") had already suggested about this architecture line
in an earlier, narrower probe.

**Attention-weight trace** (`scripts/visualize_track_query_attention.py`,
seed 4754, the ball-1 dropout case): ground-truth error on the failing
token was already 4-5 cells by step 7-12 -- well before the step-19
dropout threshold -- with a brief partial recovery at step 17 (error
0.79) before `BAILOUT` (empty cross-attention window at every expansion)
at step 18. Cross-attention weights stayed diffuse throughout the whole
trace, on every token, not just the failing one (max softmax weight
typically 0.05-0.3, rarely above 0.4) -- consistent with an
undertrained model (3 epochs, state-loss-only, same recipe as every
other checkpoint in this investigation) rather than a structurally
incapable one.

**Read: track-query is a structurally sound alternative that resolves
the identity-blending failure mode entirely, but hasn't yet matched v9
on raw dropout count at this recipe.** Every masking-family variant
(v13-v15) traded one failure mode (give-up dropout) for another
(color-bleed/identity-theft); track-query instead fails the same way
v9 does (positional drift, no identity confusion) just slightly more
often at this training budget. Given the diffuse attention weights
observed above, more training (this recipe is 3 epochs / state-loss-only
across the whole investigation, never the project's 50-epoch default)
is the most likely lever, not another architectural change. Flagging for
a human decision on whether to invest in a longer track-query training
run to see if it closes or crosses the 3/48 gap, or treat this result as
sufficient evidence the architecture direction is sound and stop here.

Checkpoints and videos: `checkpoints/token_model_h12_v17.pt`,
`videos/token_model_v17_diagnostic_grid.png`,
`videos/token_model_v17_rollout.mp4`.

## v18: observation-free rollout (free_rollout) -- 2026-09-23

Spec `docs/superpowers/specs/2026-09-23-token-free-rollout-design.md`, plan
`docs/superpowers/plans/2026-09-23-token-free-rollout.md`. After
`init_tokens` on frames 0/1, tokens evolve in state space only
(`TokenModel.step_free`, `TokenFreeDynamics` = TokenDynamics + wall-proximity
features); the rendered frame is never read back, so dropout cannot occur.
Job 2876: 10 epochs, horizon ramp 4 -> 16 over 5 epochs, state loss +
boundary + 0.1 grid loss, dataset h24 (job 2874). 10 epochs not 50: per-epoch
cost scales with unroll length, 50 epochs at 24 steps did not fit the 4h
limit (job 2875 cancelled). Loss (per-step mean) rose with horizon
(2.72 -> 9.87 at 16 steps) then plateaued 9.79 -> 9.64 over epochs 6-9.

Scored with `scripts/eval_free_rollout.py`, seed 4738, 300 steps, mean
position error vs ground truth (grid cells; token matched to ball once at
frame 1). The dropout metric is retired for v18 (tokens cannot vanish);
"faded" is the fraction of tokens whose own rendered PROB peak < 0.05 at the
last step.

20x20, 4 balls, 48 seeds:

| model | err @20 | @100 | @300 | swaps | faded |
|---|---|---|---|---|---|
| v9 (observation loop) | 5.83 | 9.54 | 15.12 | 141 | 42.8% |
| v9, observation off | 5.64 | 10.32 | 19.55 | 144 | 88.8% |
| v17 track-query | 5.10 | 9.23 | 10.19 | 141 | 0.5% |
| v18 free rollout | 5.07 | 7.47 | 8.63 | 139 | 0.0% |

50x50, 100 balls, 8 seeds (OOD):

| model | err @20 | @100 | @300 | faded |
|---|---|---|---|---|
| v9 | 21.21 | 24.08 | 41.28 | 79.5% |
| v17 | 14.46 | 29.98 | 35.94 | 8.1% |
| v18 | 16.09 | 25.71 | 26.16 | 0.0% |

Read: v18 has the lowest position error at 100 and 300 steps in-distribution
(8.63 vs v9's 15.12 at 300, -43%) and at 300 steps OOD; v17 is best at 20
steps OOD and v9 marginally best at 100 steps OOD (24.08 vs 25.71). Caveats:
(1) identity swaps are ~equal for all models and saturated OOD (~695 of ~700
tokens), so the metric does not discriminate at these horizons; (2) 0%
faded for v18 is true by construction, not evidence of accuracy; (3) mean
error ~5 cells at step 20 means every model, v18 included, has lost the
individual balls by step ~12-20 -- v18's edge is that its blobs keep moving
and stay in the region where the true balls gather, not that it tracks them;
(4) no trivial baseline (e.g. mean-position) computed, so the absolute error
scale is uninterpreted; (5) 4 of 48 seeds have a token count != 4 in all
models (init detection merges two balls), identical seed set.
Unbiased subagent frame review (grid steps 0-60): both models match through
step 3; v18 tracks roughly to step 12, then blobs bunch into a 2-3 blob group
near where the true balls settle, never vanish or smear; v9's blobs thin to
single pixels and freeze, one stays parked in empty space through step 60.
Neither tracks identity after step ~12-20.

Artifacts: `checkpoints/token_model_h24_v18.pt`, `results/eval_*.json`,
`videos/token_model_v18_diagnostic_grid.png`,
`videos/token_model_v9_long_diagnostic_grid.png`,
`videos/token_model_v18_rollout.mp4` (mp4 gitignored).

### Trivial baselines for the v18 comparison (2026-09-24)

`scripts/eval_trivial_baselines.py`, same seeds/scenarios. stay = frozen at
frame-1 ball positions; centroid = oracle true ball centroid each step;
velocity = constant frame-1->2 velocity, clamped.

20x20, 4 balls, 48 seeds (err @20 / @100 / @300):

| baseline | @20 | @100 | @300 |
|---|---|---|---|
| stay | 5.83 | 9.55 | 10.20 |
| centroid (oracle) | 6.33 | 5.47 | 5.39 |
| velocity | 7.30 | 11.58 | 12.57 |

50x50, 100 balls, 8 seeds:

| baseline | @20 | @100 | @300 |
|---|---|---|---|
| stay | 22.49 | 25.78 | 25.34 |
| centroid (oracle) | 13.18 | 16.74 | 18.09 |
| velocity | 19.64 | 28.11 | 32.40 |

Read: the v18 "win" is much smaller than the model table suggested. In
distribution v18 @300 (8.63) beats stay (10.20) by 15% and v9 (15.12) by
43%, but the oracle centroid (5.39) beats every model. OOD v18 @300 (26.16)
is no better than stay (25.34). v9 @20 (5.83) equals stay @20 (5.83) exactly
-- at step 20 no model beats standing still in distribution (v18 5.07, v17
5.10 are ~13% better). v9's late error (15.12/41.28) is worse than stay, so
its edge over the old baseline is really that v9 diverges. Free rollout
fixes divergence/fading; it does not add tracking skill.

### Predictability spike: is identity loss a chaos limit? (2026-09-24)

Throwaway probes: `scripts/probe_predictability.py`, `probe_init_error.py`,
`probe_oracle_init.py`. 20x20, 4 balls, 48 seeds (start 4738).

Twin simulator runs, frame-0 pos+vel perturbed by N(0, eps), mean per-ball
divergence (cells):

| eps | @5 | @10 | @20 | @50 | @100 | median steps to >1 / >3 cells |
|---|---|---|---|---|---|---|
| 0.001 | 0.003 | 0.009 | 0.15 | 4.14 | 8.18 | 39 / 44.5 |
| 0.01 | 0.034 | 0.083 | 0.97 | 5.44 | 8.53 | 30 / 36.5 |
| 0.1 | 0.340 | 0.727 | 3.21 | 7.09 | 8.81 | 13 / 23 |

Divergence saturates ~8.5-8.8 (unrelated-trajectory level).

v18 init_tokens error at frame 1 (44 seeds with 4 tokens): position 0.251
cells; velocity 2.10 vs mean true speed 2.46 (finite-difference of detected
positions over dt=0.15 amplifies position noise ~1/dt).

v18 free rollout from detected vs ground-truth init (position error, 44 seeds):

| init | @5 | @10 | @20 | @50 | @100 |
|---|---|---|---|---|---|
| detected | 1.00 | 1.85 | 4.80 | 8.45 | 7.41 |
| oracle | 0.64 | 1.40 | 4.40 | 8.17 | 7.39 |

Read: (1) individual-ball prediction is chaos-limited: even eps=0.1 loses
3 cells by step ~23 and everything is at the saturation level by ~50. v18's
~8.6 @300 is that saturation level; the stay-put baseline and v18 are
equally "unpredictable" there. (2) Init error is not the bottleneck --
oracle init helps early (0.64 vs 1.00 @5) and is gone by step 20 (4.40 vs
4.80). (3) Dynamics error is: v18 @5 with oracle init (0.64) is ~2x the
eps=0.1 twin (0.34); @20 (4.40) is ~1.4x (3.21). Headroom exists only at
steps ~3-20; beyond ~25 no model can track identity from two frames.
Consequence: position-error-at-300 and swap counts at 300 steps are not
meaningful comparators; swap count at 300 (~139-144 of ~170 tokens) is
saturated by chaos, not model quality.

### Horizontal velocity source (2026-09-24)

Not drift: `bounce.py` `compute_forces` sets `base_x = gravity + wfx`, so
gravity (9.0) acts along **x** (the first coordinate); y has no gravity.
"Horizontal velocity" is the gravity axis when x is drawn horizontally.
`scripts/probe_vx_drift.py`, 44 seeds, mean over 4 tokens:

| step | truth mean vx | v18 mean vx | truth speed | v18 speed |
|---|---|---|---|---|
| 1 | 0.90 | 0.96 | 2.46 | 2.84 |
| 5 | 5.16 | 5.29 | 6.20 | 5.86 |
| 10 | 4.01 | 3.30 | 9.03 | 6.75 |
| 20 | -1.52 | -0.90 | 6.00 | 2.62 |
| 50 | 1.36 | -0.24 | 6.94 | 2.23 |
| 100 | 0.09 | -0.56 | 7.87 | 2.88 |

v18 matches truth through step ~5, then loses kinetic energy: speed ~2.5-2.9
from step 20 on vs truth 6-8 (mean |vx| 1.4 vs 5.0 at 100). The
long-horizon blob "bunching" in the frame review is consistent with this
energy loss (a damped model), separate from the chaos limit.

### Correction: leftward momentum is a model bias in y (2026-09-24)

The section above attributing the drift to gravity was wrong for the
reported symptom: gravity is along x (`bounce.py:137`), and the truth has
no y force, but the v18 model develops a systematic -y (leftward) velocity.
Probes: `scripts/probe_y_bias.py`, `probe_y_bias_source.py`,
`probe_y_bias_context.py`, `probe_y_mirror.py`, `probe_head_bias.py`.

- Mean token vy, 44 seeds (model / truth): step 10 -0.00 / +0.24, step 15
  -0.54 / +0.29, step 20 -0.99 / +0.31, step 30 -2.37 / +0.15; fraction of
  seeds with mean vy < 0 at step 30: 1.00.
- The simulator is exactly symmetric under y -> 19-y, vy -> -vy. Teacher-
  forced, model prediction on mirrored vs original states violates this by a
  mean vy bias of -0.125 per step (|.| 0.345), uniform across context: free
  -0.114, neighbor<3 -0.124, y-wall -0.145.
- `dynamics.delta_head.bias` = [dpos_x -0.050, dpos_y +0.004, dvel_x -0.145,
  dvel_y -0.099]: the -0.099 dvel_y bias is a constant y-acceleration in a
  system with none, matching the -0.125 mirror violation.
- Wall response is not mirror-symmetric (teacher-forced d_vy vs truth: near
  y<3 +1.33, near y>16 -1.89), bounces are weak (hand-built test: vy -3 in,
  ~0.8 out), and pair collisions do not conserve momentum (2-ball head-on:
  total vy wanders +-0.4, truth exactly 0).

### v19 stepwise curriculum vs v18 (2026-09-24)

Job 2878 (`scripts/polaris_train_token_v19.sh`): stepwise 1..20 curriculum,
plateau advance (3%, 1000-batch chunks, <=4 chunks/stage), 25% replay,
21m43s. Stage losses (mean per-step over the unroll): stage 1 2.07->1.81,
stage 4 3.23->3.11, stage 8 5.78->5.39, stage 18 12.2->11.9; the stage-1
floor of ~1.8 is high for a 1-step prediction.

`scripts/eval_free_rollout.py`, 48 seeds, 20x20, 4 balls, mean position
error at step (`results/eval_v18_short.json`, `eval_v19_short.json`):

| step | v18 | v19 |
|---|---|---|
| 1 | 0.431 | 0.424 |
| 2 | 0.641 | 0.640 |
| 3 | 0.833 | 0.863 |
| 5 | 1.176 | 1.269 |
| 10 | 2.038 | 2.174 |
| 15 | 3.582 | 3.714 |
| 20 | 5.073 | 5.192 |

Read: the curriculum gives no improvement (slightly worse at 5-20). Step-1
error is 0.43 for both, above the 0.25 init position error, so the floor is
set by init-state error (velocity error 2.10), not by the training regime.

Unbiased subagent frame review of v19 vs v18 (seeds 4738, 4739, steps 0-30,
no hypothesis primed): both match ground truth to step ~4-8, then model
blobs migrate to the lower part of the frame (the gravity/x-floor side) and
stay there, clustering or fragmenting, while truth balls keep bouncing back
through the upper half and the right/left edges. Neither model shows balls
bouncing back up. v19's blobs lump more (seed 4739: packed into ~5 cells by
step 15); v18 stays somewhat more spread; differences between them are small
next to their common departure from truth. Consistent with the energy-loss
finding: the dominant long-horizon defect is inelastic floor/wall contact.
Grids: `videos/token_model_v19_diagnostic_grid_s473{8,9}.png`.

### Init velocity from the frame's VX/VY channels (2026-09-24)

Subagent finding (probe: scratchpad `probe_init_refine.py`): `init_tokens`
finite-differences two detected positions, but every frame carries per-cell
VX/VY channels (bounce.py: probability-weighted mean velocity of the covering
balls). Reading them (window radius 1, weighted by the undone PROB
saturation -log(1-PROB)) gives init velocity error 0.010 vs 2.075 for the
finite difference, 40/48 seeds with 4 tokens. Position error at init stays
0.24 (single-frame quantisation of a radius-0.75 disk); a joint two-frame
fit reaches 0.04 (not implemented). 8/48 seeds merge nearby balls in
`find_token_positions` (min pair distance 0.78-1.93 cells).

Implemented as `TokenModel(velocity_readout=True)` /
`--velocity-readout` (default off). No-retrain eval on the v18/v19
checkpoints (48 seeds, position error at step):

| step | v18 | v18+readout | v19 | v19+readout |
|---|---|---|---|---|
| 1 | 0.431 | 0.361 | 0.424 | 0.336 |
| 3 | 0.833 | 0.595 | 0.863 | 0.656 |
| 5 | 1.176 | 0.831 | 1.269 | 0.962 |
| 10 | 2.038 | 1.653 | 2.174 | 1.752 |
| 20 | 5.073 | 4.613 | 5.192 | 4.783 |

The checkpoints were trained with noisy init velocity, so this is a lower
bound. v21 (job below) retrains the v19 curriculum with readout.
