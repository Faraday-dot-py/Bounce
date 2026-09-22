# Findings: blocky/tiled "quilt" artifact (v4)

Investigation of the new artifact reported after
`docs/debugging/findings-padding-mass-conservation.md`'s PROB
renormalization fix landed (`stage2_flownet_h12_v4.pt`, job 2825): a
coarse, axis-aligned blocky/tiled mosaic appears from ~step 8-12 onward
and dominates the rollout through step 30. Investigation only —
`model/net.py` was not edited; all probes below run against frozen v4
weights in scratch scripts.

## Ruling out the obvious suspects

All candidate mechanisms from the investigation brief were checked
directly against instrumented rollouts and ruled out with evidence,
not assumption:

- **`grid_sample(mode="bicubic")` / `padding_mode="border"` OOB or
  boundary pathology**: OOB fraction stays 2-5% throughout the
  rollout, never spikes near the artifact's onset step.
- **Flow saturating `max_flow=4.0`**: saturation fraction of the flow
  field is ≈0 at every step checked — flow magnitudes never approach
  the tanh bound.
- **GroupNorm(8, 64) behaving spatially-blocky**: confirmed
  numerically (not just by reading the docs) that it normalizes
  globally per channel-group over all spatial positions — a
  hand-computed reference matches PyTorch's GroupNorm output to 2.4e-7,
  ruling out any per-block normalization behavior.
- **PROB renorm scale (`prob_in / (prob_out + 1e-6)`) blowing up or
  behaving pathologically**: scale stays bounded in [0.70, 1.00] across
  the rollout; clamping it in a counterfactual is a bit-identical
  no-op, confirming the scale value itself isn't the lever.
- **NaN/Inf**: none anywhere in the rollout.
- **Matplotlib/rendering artifact**: ruled out — blockiness is present
  in the raw tensor values themselves. A 10x10-block variance-ratio
  metric (variance-of-block-means / mean-of-in-block-variances) climbs
  from 0.03 at step 0 to 0.89 by the artifact's mature state, computed
  directly on the tensor, no plotting involved.
- **Dead-neuron / receptive-field "basin of attraction" pattern**: a
  Jacobian-fold probe testing whether the blocky regions correlate with
  a specific receptive-field/saturation structure found ~0 correlation.
- Flow and correction fields themselves are smooth with no block edges
  — the blockiness is not present in the model's raw per-step outputs
  in isolation, only in the accumulated state.

## Real mechanism: bicubic diffusion + exact mass renorm compounding

The root cause is the already-documented resampling diffusion from
`findings-correction-drift-and-mass-dissolution.md` (Part 2) —
`grid_sample`'s bicubic resampling spreads PROB mass into
neighboring cells every step — combined with v4's own mass-renorm fix
in a way that wasn't anticipated:

1. **PROB's occupied area saturates the entire 50x50 grid by ~step
   8**, confirmed against ground truth: GT stays around ~10% nonzero
   area throughout (physically, a bounded number of discrete balls),
   but the model's nonzero-area fraction reaches 82-93% by steps 6-10.
   This happens almost identically whether renorm is on or off (both
   reach >90% nonzero area by step 6-8) — **the area-saturation itself
   is the pre-existing bicubic-diffusion bug, not something the renorm
   fix introduced.**
2. **What v4's renorm changes is what the saturated state looks like,
   not whether saturation happens.** Without renorm (pre-v4), the
   diffused, spread-out mass keeps growing in total (the drift bug),
   which visually presented as columnar streaking. With renorm (v4,
   which is otherwise correct and necessary — it's what fixed the
   VX/VY drift), total PROB mass is pinned exactly constant every
   step. But because the diffusion has already spread that fixed
   amount of mass across ~90% of the grid, and the flow/correction
   heads keep producing their own smooth, low-frequency residual
   structure on top, that residual structure never gets to fade to
   invisible background the way it would pre-renorm (where it would
   eventually just decay along with everything else) — instead it gets
   perpetually renormalized back up to match the fixed total mass,
   turning the model's smooth low-frequency output noise into a
   persistent, growing, blocky mosaic.

This is a **third, distinct interaction** in the same drift/dissolution
family as the two previously-fixed bugs, not a new independent
architectural flaw: bicubic diffusion (Part 2 of the prior findings)
was previously masked because, pre-renorm, the same diffusion just
showed up as decaying peak intensity (`max0` dropping) — renorm is
what's now surfacing it as sustained mosaic structure instead of decay.

## Candidate fixes tested (frozen v4 weights)

| candidate | effect |
|---|---|
| `mode="nearest"` instead of bicubic | reduces the blockiness metric somewhat, but risks aliasing under real predicted (non-rigid) flow fields, and does not address the underlying area-saturation |
| local/windowed renorm (per-tile instead of whole-frame) | worse — introduces its own block-periodic artifact from the windowing itself |
| clamp renorm `scale` to a tighter range | no-op (bit-identical to unclamped) — confirms `scale` magnitude isn't the mechanism |
| **soft-threshold PROB before renorm**: `relu(prob - thresh)` with `thresh ≈ 0.01-0.02`, applied to `out[:,0:1]` before computing `prob_out`/rescaling | **recommended** — cuts nonzero-area fraction from 1.00 to 0.46-0.65 and the blockiness variance-ratio metric from 0.89 to 0.06-0.22, no reintroduction of VX/VY drift or period-2 oscillation in the frozen-weight counterfactual |

## Recommendation

Don't change the renormalization formula itself (`findings-padding-mass-conservation.md`'s
fix is working exactly as designed and is still needed for the VX/VY
drift fix). Instead, address the actual source: soft-threshold the
PROB channel (zero out near-background noise below a small threshold)
*before* computing the renormalization sum, so residual sub-threshold
diffusion noise isn't counted as "mass" that then gets amplified back
up across the whole grid. This directly targets the confirmed
mechanism (diffusion spreading mass into near-zero background,
renorm then treating that background noise as real occupied mass)
rather than papering over the visual symptom.

Requires retraining/reverification like every other forward-pass
change this session (`docs/debugging/experiment-log.md`'s established
pattern), plus a small threshold sweep (0.01-0.02 range tested so far)
against real step-1 MSE to pick the final value, since a threshold
that's too aggressive would start clipping genuinely dim but real
occupancy.

## Repro / instrumentation scripts (not committed to the repo)

Scratch probes and diagnostic images (GT-vs-model nonzero-area
comparison, renorm on/off counterfactual, block-variance-ratio metric,
Jacobian-fold correlation probe, candidate-fix comparisons including
the soft-threshold candidate) live in the investigating subagent's
scratchpad, not in the repo, per the same convention as
`findings-correction-drift-and-mass-dissolution.md`. If a follow-up
wants these as permanent repo tools they should be cleaned up and
added under `scripts/`.
