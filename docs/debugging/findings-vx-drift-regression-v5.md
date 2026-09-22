# Findings: VX/VY drift regression in v5

Investigation of why the VX drift (previously addressed in
`findings-correction-drift-and-mass-dissolution.md` /
`findings-padding-mass-conservation.md`) came back substantially worse
in `checkpoints/stage2_flownet_h12_v5.pt` (the PROB soft-threshold
retrain that fixed the quilting artifact, `findings-quilting-artifact.md`)
than it was in `checkpoints/stage2_flownet_h12_v4.pt`. Investigation
only — `model/net.py` was not edited by the investigating subagent; all
counterfactuals below run against frozen weights in scratch scripts.

## v5's flow field is not larger or more convergent than v4's

Direct apples-to-apples measurement (same seed 4738 IC, each model's
own actual trained forward pass) ruled out the "v5 learned a stronger/
more convergent flow field" hypothesis:

- whole-frame flow magnitude: v5 0.969 vs v4 1.136 (v5 smaller)
- occupied-cell flow magnitude: v5 1.039 vs v4 1.086 (v5 smaller)
- flow saturation fraction: v5 0.0% vs v4 0.66%
- flow convergence at occupied cells: v5 -0.044 vs v4 -0.054 (v5 no
  more convergent)

v5's flow is smaller by every measure checked, so "the model learned to
herd content more aggressively with flow" is not what's happening.

## Real mechanism: the pre-existing warp non-conservation bug was always there; v4's own PROB bug was accidentally masking it

Confirmed cleanly across the full 30-step rollout for both models:
`correction`'s contribution to VX/VY's per-step spatial mean is exactly
`0.00000` at every single step for both v4 and v5 (centering works
exactly as designed in both). **100% of the VX/VY mean growth comes
from the warp (`grid_sample`) step itself** — the same
Jacobian-determinant non-conservation mechanism already identified in
`findings-padding-mass-conservation.md`, which was deliberately left
unfixed for VX/VY ("not physically justified" to renormalize a velocity
field the way PROB mass is renormalized).

What actually changed between v4 and v5: **v4's own (buggy) PROB
diffusion accidentally acted as a brake on this pre-existing drift.**
v4's PROB channel diffused to near-uniform noise across 80-100% of the
grid by step 8 (the quilting bug); once the field is spatially flat,
there's very little structure left for convergent flow to
concentrate, which accidentally kept v4's VX drift plateaued at
~0.46-0.49. v5's threshold fix correctly keeps PROB sparse/structured
(nz-frac 0.55-0.65 throughout the rollout, as intended — that's the fix
working) — so that accidental brake is gone, and the pre-existing,
always-latent VX pump now runs unchecked up to ~1.37.

A causal single-variable test (v5's own trained weights, only the
renorm formula swapped back to v4's un-thresholded style) confirms
this directly: reduces peak VX by ~15-25% (a partial effect, since v5's
raw pre-threshold PROB predictions differ somewhat from v4's own raw
predictions — the retrain changed more than just the threshold's
downstream effect).

## Candidate fixes tested (frozen v5 weights)

- **(a) Full 3-channel renorm** — recenter VX/VY's frame-wide *mean*
  (additive, not multiplicative like PROB's sum-based rescale, since
  VX/VY are signed) back to the pre-warp frame's mean, every step. This
  is the fallback documented (but not previously adopted) in
  `findings-padding-mass-conservation.md`. Result: **exactly flattens**
  VX/VY drift (stays at 0.0162 / -0.0098, unchanged to 5 decimal places,
  verified stable to 60 steps), bit-identical step-1 MSE (0.684x either
  way).
- **(b) "Recenter only the warp's own injected mean-shift"** (i.e.
  recenter VX/VY immediately after `grid_sample`, before `correction`
  is added, rather than after the full `out`) — **mathematically proven
  and numerically confirmed identical to (a)** (max diff 4.8e-8), since
  `correction` is already exactly zero-mean per-channel per-step. Not a
  distinct softer option, just a different point in the forward pass to
  apply the same operation.
- **(c) Reducing `max_flow`** (2.0 or 1.0, down from 4.0) — only
  partially slows the drift, never flattens it. Confirms the same
  verdict `findings-correction-drift-and-mass-dissolution.md` already
  reached transfers unchanged to v5 (this isn't a magnitude-cap
  problem, it's a lack-of-restoring-force problem).

## Recommendation

Apply (a) (equivalently (b)) as the default: additively recenter
VX/VY's frame-wide spatial mean to match the pre-warp frame's mean,
every step, mirroring how `correction` is already centered but applied
to the whole per-step update rather than just the correction term.

This is the second time leaving VX/VY unrenormalized has failed — first
as an unaddressed drift that `findings-padding-mass-conservation.md`
chose not to fix, now as a regression surfaced by an otherwise-correct
and wanted fix (the quilting fix removed an accidental brake on this
same pre-existing bug). It costs nothing on every metric checked
(step-1 MSE, period-2 oscillation, edge/gridding lattice all
unaffected) and, because it's applied incrementally every step, it
still allows genuine within-step structured changes from `correction`
and from non-uniform components of the warp — it only removes the
warp's own spatially-uniform (DC) non-conservation each step, not any
spatially-varying physical signal. The tradeoff flagged previously
(pinning VX/VY's frame-wide mean prevents representing a genuine
multi-step secular trend, e.g. gravity's effect on VY's global mean)
is accepted here given two consecutive failures of the alternative and
zero measured cost against every metric tracked so far.

## Repro / instrumentation scripts (not committed to the repo)

Frozen-weight probes (flow-magnitude/convergence comparison, per-step
warp-vs-correction mean-growth decomposition across 30 steps, the
renorm-formula swap-back test, and the (a)/(b)/(c) candidate-fix
comparisons) live in the investigating subagent's scratchpad, not the
repo, per the same convention as the other findings docs in this
directory.
