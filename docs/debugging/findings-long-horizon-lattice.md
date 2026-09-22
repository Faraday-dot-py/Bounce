# Findings: long-horizon periodic lattice in the v7 100-step rollout (Problem 3)

Investigation of the faint diagonal ripple that grows to dominate
`stage2_flownet_h12_v7.pt`'s 100-step rollout
(`videos/stage2_flownet_v7_rollout_comparison_100step.mp4`, seed 4738), per
`docs/debugging/flownet-open-issues-v2.md` Problem 3. Repro/analysis script:
`scripts/investigate_long_horizon_lattice.py` (run with `PYTHONPATH=. python3
scripts/investigate_long_horizon_lattice.py --out-dir <dir>`), plus two
one-off scratch checks (band-power fraction, bicubic/bilinear counterfactual)
described inline below.

## Diagnosis (short version)

**This is a third, previously-unflagged periodic bias — not a resurfacing of
either the border/`padding_mode="zeros"` OOB mechanism
(`findings-gridding-artifact.md`) or the period-2 temporal brightness
oscillation (`findings-period2-oscillation.md`).** It is sourced by the
`grid_sample(..., mode="bicubic")` resampling kernel added to fix peak-decay
(`findings-correction-drift-and-mass-dissolution.md`), is architecturally
present in **both** v6 and v7 (same `net.py`, same trained warp behavior at
the relevant scale), and only becomes visually/spectrally dominant in v7
because v7's "give up" collapse (`findings-peak-decay-dissolution.md`) empties
the frame of competing real content — the same *unmasking* dynamic already
documented for the border artifact, but a structurally different underlying
mechanism (interpolation-kernel texture, not zero-padding discontinuity).

## Evidence

### 1/2. Real rollouts (v7 and v6), FFT of interior 40x40 crop, steps 5-100

Using the exact FFT method from `findings-gridding-artifact.md` Part 2
(Hanning-windowed FFT of the interior 40x40 crop of the 50x50 PROB channel,
top peaks by magnitude):

```
v7 (production forward(), real weights):
  step   5: amp=0.058  period ~(40, 6.7)px   [transient, still real content]
  step  10: amp=0.046  period ~13.3px
  step  20: amp=0.042  period ~4.0px
  step  40: amp=0.029  period ~(3.6, 5.0)px
  step  60: amp=0.022  period ~(3.6, 5.0)px
  step  80: amp=0.022  period ~(3.3, 5.0)px
  step 100: amp=0.024  period ~(3.3, 5.0)px

v6 (production forward(), real weights):
  step   5: amp=0.026  period ~(40, 20)px
  step  10: amp=0.013  period ~13.3px
  step  40: amp=0.010  period ~13.3px
  step  60: amp=0.010  period ~8-10px
  step  80: amp=0.010  period ~8-10px
  step 100: amp=0.010  period ~8-10px
```

By step 60+, v7's dominant spatial period settles to **~3.3-5px**, distinctly
*finer* than the 13-40px coarse banding found for the short-horizon lattice in
`findings-gridding-artifact.md`. The (+3.3,-5.0)/(-3.3,+5.0) peak pair (equal
magnitude, opposite-sign diagonals) is a genuine 2D diagonal weave, matching
the unbiased reviewer's "diagonal ripple" / "fine periodic checkerboard"
description exactly. v6, at the same steps, stays anchored at the *coarser*
8-40px scale and never develops the ~3-5px signature — see Part 3 for why
this doesn't mean v6's architecture lacks the mechanism.

Rendered comparison grid (`long_horizon_lattice_grid.png`, steps
1/20/40/60/80/100): v7 shows sharp diagonal interference-fringe bands
radiating from a corner region, growing to cover the whole crop by step 100.
v6 shows a smooth top-dark/bottom-light gradient (its "blur everywhere"
collapse) with no periodic texture at any step — a qualitatively different
failure geometry, not a fainter version of the same pattern.

### 3. Is this present, low-amplitude, at v7's early steps / in v6 anywhere?

Spectral **band-power fraction** in the 3-6px period band (fraction of total
interior-crop FFT power, not just top-peak location — a more sensitive test
than peak-picking for a masked/latent signal):

```
           step 5   step 10  step 20  step 40  step 60  step 80  step 100
v7 band%:  34.2%    41.4%    42.0%    23.8%    15.0%     8.8%    11.1%
v6 band%:  30.3%     1.6%     0.7%     3.3%     0.3%     0.2%     0.2%
```

Both checkpoints start with comparable 3-6px power at step 5 (splat-initialization
transient, not yet the lattice). v7's band power **stays substantial (9-42%)
indefinitely**; v6's **collapses to near-zero by step 10 and never returns**.
This directly answers investigation point 3: v6 does not carry a detectable
low-amplitude version of this specific lattice once real content is present —
the mechanism is latent in the *architecture* (both checkpoints share
identical bicubic-warp code) but v6's broader, never-collapsing content
distribution (uniform blur, not give-up) apparently suppresses/dominates it
in a way v7's near-empty late-rollout frames do not. This is consistent with,
not contradictory to, the "unmasking" framing: the difference between v6 and
v7 here is how much real content competes with the latent bias, not whether
the bias exists in the weights.

### 4. `padding_mode` counterfactual (same method as `findings-gridding-artifact.md`)

Swapped `grid_sample`'s `padding_mode` between the shipped default
(`"border"`, already the fix from `findings-gridding-artifact.md`) and the old,
pre-fix `"zeros"`, at inference only, same trained v7 weights, no retraining
(instrumented forward mirrors `net.py`'s current `forward()` exactly, verified
by a sanity check: `padding_mode="border"` instrumented rollout matches the
real `model.forward()` rollout to **0.000000 max abs diff**):

```
                 step 40         step 60         step 80         step 100
border (cur.):  amp=0.0285      amp=0.0223      amp=0.0217      amp=0.0238
                period~(3.6,5.0) period~(3.6,5.0) period~(3.3,5.0) period~(3.3,5.0)
zeros (old):    amp=0.0376      amp=0.0246      amp=0.0216      amp=0.0244
                period~(20,40)   period~(3.6,6.7) period~(3.3,4.4) period~(3.3,4.4)
```

Unlike the short-horizon result in `findings-gridding-artifact.md` (where
`"border"` measurably softened the sharp lattice-line texture), at long
horizon **both padding modes converge to the same ~3.3-5px period and near-
identical amplitude by step 60+**. `padding_mode` produces only a transient
difference at intermediate steps (20-40) that washes out — it is **not** a
causal driver of the long-horizon lattice. This rules out mechanism (1) from
`findings-gridding-artifact.md` as the explanation here.

### 5. Follow-up counterfactual: `grid_sample` interpolation mode (bicubic vs. bilinear)

Since neither padding-mode nor v6-vs-v7 weight differences explain the ~3-5px
period, and that scale roughly matches bicubic interpolation's 4x4 sample
support, swapped `grid_sample`'s `mode` from `"bicubic"` (shipped default,
added specifically to fix peak-decay/blur — see `net.py` docstring and
`findings-correction-drift-and-mass-dissolution.md`) to `"bilinear"`, inference
only, same v7 weights, `padding_mode="border"` held fixed:

```
                 step 40          step 60          step 80          step 100
bicubic (cur.):  amp=0.0285       amp=0.0223       amp=0.0217       amp=0.0238
                 period~(3.6,5.0) period~(3.6,5.0) period~(3.3,5.0) period~(3.3,5.0)
bilinear:        amp=0.0257       amp=0.0196       amp=0.0192       amp=0.0192
                 period~10px      period~10px      period~13.3px    period~10-13px
```

Swapping to bilinear **shifts the dominant period from ~3.3-5px back to the
coarser ~10-13px scale** (matching the short-horizon-artifact family's range)
and measurably lowers amplitude at every late step. This isolates
`mode="bicubic"` as a causal driver of the fine ~3-5px lattice specifically —
the interpolation kernel itself, not the border/padding handling, produces a
persistent high-frequency diagonal texture under repeated self-composition
across ~60-100 autoregressive steps.

## Answer to "same mechanism, or novel?" (investigation point 5)

**A distinct, third periodic bias**, not a resurfacing of either previously-
fixed issue:

- **Not** the `padding_mode="zeros"`/border-OOB mechanism
  (`findings-gridding-artifact.md`): that fix is confirmed in place in
  `net.py` (`padding_mode="border"` on `grid_sample`, `padding_mode="replicate"`
  on all convs), and the long-horizon lattice's period/amplitude are
  insensitive to reverting it (Part 4) — plus its characteristic scale
  (13-40px) doesn't match what's observed here (3-5px) at the steps where the
  lattice dominates.
- **Not** the period-2 temporal brightness oscillation
  (`findings-period2-oscillation.md`): that is a *temporal* alternation
  (frame-to-frame overall brightness flip) already addressed by bounded+
  centered correction; this is a *spatial* diagonal texture within a single
  frame, unrelated in kind.
- **Is** a latent bias of `grid_sample(..., mode="bicubic")` at its ~4px
  kernel-support scale, architecturally shared by both v6 and v7 (Part 5), that
  becomes spectrally and visually dominant specifically in v7 because v7's
  give-up collapse leaves almost no competing real content to mask it by
  step 60-100 (Part 3) — the same *unmasking* dynamic as the border artifact,
  but a different underlying source. It is not specific to v7's peak-loss
  training mechanism itself; it is specific to v7's *failure mode* (give-up
  emptying the frame), which the peak loss produces.

## Implication

This is evidence for, not a contradiction of, `findings-peak-decay-
dissolution.md`'s conclusion that the blur-vs-give-up tradeoff is the real
problem: bicubic resampling was adopted specifically to fix peak-decay/blur
(trading a diffusion-dominated failure for a sharper-but-narrower one), and
that same interpolation choice carries a small persistent high-frequency
texture that was already present under v6 too but stayed below the noise
floor of "blur everywhere." Chasing this lattice directly (e.g. reverting to
bilinear, per Part 5) would only trade back into faster peak decay — it is
not an independent bug worth fixing in isolation, consistent with the
standing guidance in `flownet-open-issues-v2.md` to treat problems 1-3 as
inputs to the redesign-scoping conversation rather than isolated patches.
