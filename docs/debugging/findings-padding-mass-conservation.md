# Findings: padding-mode mass conservation (drift-padding-conservation-brief.md)

Investigation of whether `grid_sample`'s `padding_mode="border"` is the
primary driver of `stage2_flownet_h12_v3`'s persisting VX-channel drift
(mean1 climbing to 1.30 by step 30). Investigation only —
`model/net.py` was not edited; all counterfactuals below run scratch
copies of `forward()`'s logic, matching the convention of the prior
`findings-*.md` docs in this directory.

## Headline finding: the brief's leading hypothesis is real but secondary

The brief's synthetic rigid-shift ramp probe (a uniform-direction shift
applied to a fixed ramp field, no model) showed `padding_mode="border"`
pumps a field's mean under sustained directional shifting while
`"zeros"` doesn't. That mechanism is real, but for the actual trained
model it turns out to be a **minor, secondary** contributor, confirmed
three ways:

1. **Real-model counterfactual**: swapping only `padding_mode`
   (`"border"` -> `"zeros"`), same weights, same IC (n=50, 150 balls,
   seed=4738), 30-step rollout: VX mean drops from 1.302 to 0.765 — a
   ~40% reduction, not elimination. If border-padding pumping were the
   whole story, `"zeros"` should have flattened the drift close to the
   correction-only baseline (~0.14-0.16 from the earlier centering
   counterfactual); it doesn't.
2. **Interior-only breakdown**: restricting analysis to a 5px margin
   where zero pixels ever sample out-of-bounds (so `padding_mode`
   provably never executes for any of these pixels), `warped`'s mean
   grows almost identically under `"border"` and `"zeros"` (e.g. step 8:
   1.0102 vs 1.0055) — the interior drift is essentially unaffected by
   padding mode, so padding mode cannot be the interior-region driver.
   Flow's spatial divergence is consistently negative (converging)
   during the window of fastest growth.
3. **Clean synthetic proof**: a non-rigid, spatially convergent
   (non-uniform) flow field, with sample coordinates constructed to be
   strictly interior (OOB fraction exactly 0 by construction, padding
   mode literally never invoked) still pumps mass 2.4x over 15 steps,
   and `"zeros"`/`"border"` results are bit-identical in this
   configuration (confirms padding mode plays no role when nothing goes
   out of bounds).

## The real mechanism: Jacobian oversampling from a locally convergent flow field

`grid_sample` performs no Jacobian-determinant correction for local
compression/expansion of the sampling map. When the predicted flow field
is *locally convergent* (negative spatial divergence — sample points
pulling inward from a wider area into a smaller one, which is what a
"balls falling together" or "balls settling" scene's flow legitimately
looks like in places), each resample effectively oversamples the source
region relative to the destination, inflating values there. This is a
structural property of bilinear/bicubic warping under any
non-divergence-free flow field, **independent of `padding_mode`** — it
happens identically whether or not any sample point ever goes out of
bounds. This explains why the padding-mode swap only ever produced a
partial (~40%) reduction: it was removing a real but smaller, additive
contributor on top of this larger, boundary-independent one.

## Candidates tested (frozen weights, seed 4738, no retraining)

| Candidate | VX mean @ step 30 |
|---|---|
| A. baseline (current: `border`, centered correction, bicubic) | 1.302 |
| B. `padding_mode="zeros"` | 0.765 (partial only, confirms it's secondary) |
| C. renormalize PROB (channel 0) only, per step | 0.958 (~26% reduction via the PROB->VX/VY feedback loop) |
| D. renormalize all 3 channels, per step | 0.016 (flat — fully eliminates the drift) |
| E. `max_flow=1.0` (down from 4.0) | 0.413 |
| F. `max_flow=2.0` | 0.560 |
| G. damp flow magnitude within 5px of the edge only | 0.889 |
| H. `padding_mode="reflection"` | 1.285 (~= border, consistent with Part 1's finding that reflection pumps the same way) |

All of A/C/D stable to 60 steps (no oscillation, no re-blowup).
**No MSE regression** from renormalization: step-1 PROB MSE-vs-copy-
baseline slightly *improves* with PROB-only renorm (0.841x vs baseline's
0.881x); 30-step rollout PROB/VX/VY MSE vs ground truth is essentially
unchanged between A/C/D. Renormalization fixes the visible
drift/ripple/edge-brightening symptom but does **not** fix VX/VY's
underlying single-step prediction accuracy (MSE vs ground truth stays
15-20 at step 20 regardless of which candidate) — that's a separate,
unaddressed problem, not claimed to be fixed here.

A per-pixel Jacobian-determinant correction (the more principled,
textbook semi-Lagrangian fix for this exact mechanism) was attempted but
not successfully implemented in the time available — a
divide-vs-multiply sign/convention bug produced either 11.3x growth or
over-corrected to 0.525x decay depending on direction. Flagged as a
possible follow-up if the simpler renormalization proves insufficient
later, not recommended now given it isn't working yet and the simpler
fix already validates cleanly.

## Recommendation

1. **Keep `padding_mode="border"`** (don't revert to `"zeros"`) — it is
   not the primary drift mechanism, and reverting would only recover
   ~40% of the fix while reopening the previously-confirmed,
   independently-diagnosed OOB-perimeter/edge-brightening problem from
   `findings-gridding-artifact.md`.
2. **Add explicit post-warp mass renormalization for channel 0 (PROB)
   unconditionally** — physically well-justified (total occupied "mass"
   should be roughly conserved frame to frame in this physics), validated
   with no MSE regression, reduces the VX drift by ~26% as a side effect
   (through the PROB->VX/VY coupling identified in
   `findings-period2-oscillation.md`'s Jacobian probe).
3. **For VX/VY, don't force exact per-step sum-matching (candidate D) as
   the default** — it fully eliminates the visible drift artifact, but
   pins VX/VY's global spatial mean to the initial-frame value for the
   entire rollout, which isn't more physically correct (real velocity
   fields should be able to have genuine net drift, e.g. from gravity)
   and MSE is statistically unchanged with or without it. Keep it
   documented as a tested, safe fallback if fully eliminating the visible
   symptom becomes the priority over physical fidelity, but it is not the
   primary recommendation.
4. **Don't use tighter `max_flow` or edge-only flow damping as the
   primary fix** — both are weaker than PROB renormalization (E/F/G all
   still leave VX well above the correction-only baseline) and shift the
   model's already-trained operating point, which would require
   retraining to re-adapt, unlike renormalization which is a pure
   post-hoc correction compatible with the existing trained weights.

## Repro / instrumentation scripts (not committed to the repo)

Scripts used for this investigation live in the session scratchpad, not
in the repo, per the same convention as the prior `findings-*.md` docs
in this directory:
- Real-model padding-mode counterfactual (`border` vs `zeros`, candidate
  B)
- Interior-only (zero-OOB-margin) breakdown isolating padding mode from
  the Jacobian-oversampling mechanism
- Synthetic strictly-interior convergent-flow proof (padding mode never
  invoked, mass still pumps)
- Candidate harness (C through H) with per-step VX/VY/PROB mean logging
  and step-1/30-step MSE comparison
