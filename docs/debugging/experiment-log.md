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
