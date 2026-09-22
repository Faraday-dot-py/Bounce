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
