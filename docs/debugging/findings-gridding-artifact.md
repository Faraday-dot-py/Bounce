# Findings: grid/lattice artifact + edge brightening (Problem 1)

Investigation of `stage2_flownet_h12`'s rollout lattice/window-pane pattern
and edge/corner brightening, per `docs/debugging/flownet-open-issues.md`.
Repro/analysis script: `scripts/investigate_gridding_artifact.py` (run with
`PYTHONPATH=. python3 scripts/investigate_gridding_artifact.py`).

## Diagnosis (short version)

**The leading hypothesis -- classic dilated-conv "gridding" from the
`(1, 2, 4, 8, 4, 2, 1)` schedule -- is not supported by the evidence and
should not be the basis for a fix.** The schedule starts and ends at
dilation 1, and every block has a residual identity connection, so the
receptive field has no literal coverage holes and no meaningful high-
frequency (2/4/8px-scale) periodic bias -- the kind of artifact the HDC
paper describes doesn't show up here, theoretically or empirically.

**The actual, confirmed root cause is the interaction of two things at the
frame boundary:**

1. The dilated-conv stack's zero-padding creates an anomalous flow
   prediction confined almost entirely to the outermost 1px ring
   (`|flow|` there is 2.2x the interior mean, and drops back to interior
   levels immediately one pixel in -- not a gradual multi-pixel decay).
2. `grid_sample(..., padding_mode="zeros")` turns any out-of-bounds sample
   into a hard 0. Because of (1), **literally 100% of the frame's 1px
   perimeter ring (196/196 pixels on the 50x50 grid) samples out of
   `[-1, 1]` on every single rollout step**, including step 1 on a clean,
   non-degraded input (this is not something that only emerges after the
   rollout has already degraded).

Every step, the entire border ring's warped term is therefore exactly zero,
and `correction_head` has to reconstruct the whole border pixel from
scratch with no continuity from the previous frame -- a strictly harder,
noisier prediction task than at interior pixels, which get a real warped
value to correct. This is the direct cause of the reported edge/corner
brightening, and (see below) a substantial contributor to the coarser
"lattice" pattern that appears later in rollout.

The interior "lattice"/window-pane pattern is real, but its measured
spatial period (13-40px on a 50px grid, i.e. a handful of broad bands) is
far coarser than the 2/4/8px scale a dilation-gridding checkerboard would
produce. It is consistent with the border defect propagating inward over
repeated autoregressive steps through the network's own receptive field
(the seven dilations sum to a ~22px half-width, i.e. close to the whole
image in one hop) and interacting with the correction head's overshoot
instability (see `findings-period2-oscillation.md` / Problem 2) -- not
with an intrinsic mid-stack dilation artifact.

## Evidence

### 1. Theoretical coverage of the dilation stack does not show gridding holes

`path_count_coverage()` propagates a unit impulse through 7 blocks of
`{3x3 conv, dilation d} x2 + residual`, matching `ResidualConvBlock`
structure, with `DILATIONS = (1, 2, 4, 8, 4, 2, 1)` and ones-kernels (the
standard linear proxy used in the dilated-conv gridding literature, e.g.
Wang et al., *Understanding Convolution for Semantic Segmentation*).

```
zero-coverage cells in 41x41 window around impulse: 0/1681
mean coverage even/odd row parity: 15824398336.0 / 15797315584.0  (ratio 1.0017)
top non-DC FFT peaks of coverage map:
  period=(inf px, 24.5px) / (24.5px, inf px)   [window-edge artifact, not a real periodicity]
```

No pixel in the receptive field is ever reached by zero paths, and the
even/odd parity bias is 0.17% -- negligible. The only non-DC FFT energy is
at ~24.5px, which is an artifact of the 49px analysis window's edge, not a
genuine periodic signal. **This falsifies the "holes in the receptive
field" mechanism for this specific schedule** -- it has dilation-1 layers
at both ends and residual/identity paths at every block, which is exactly
what keeps a "hole"-style gridding artifact from forming.

### 2. The actual rollout's spatial period is far coarser than dilation scale

FFT of the interior 40x40 crop of real rollout frames at steps 16/20/25/30:

```
step 16: dominant periods ~20-40px
step 20: dominant periods ~13-40px
step 25: dominant periods ~13px
step 30: dominant periods ~13px
```

If this were a dilation-gridding checkerboard, the expected periods would
track the dilation rates directly (2px, 4px, 8px). Instead the pattern is
2-4 broad bands across the whole 50px grid -- a low-frequency phenomenon.

### 3. The border ring is deterministically, fully out-of-bounds every step

```
step1 |flow| at border ring: 0.4333  vs interior (>=10px in): 0.1967  ratio 2.20x
step1 OOB fraction: 0.0784  (perimeter ring = 196/2500 = 0.0784)
OOB per-row counts (step1): [50, 2, 2, 2, ..., 2, 2, 50]
```

The OOB fraction (fraction of pixels whose `grid_sample` lookup coordinate
falls outside `[-1, 1]`) is *exactly* `perimeter / total = 196/2500`, and
the per-row OOB count confirms the OOB set is precisely the 1px border
ring (row 0 and row 49 are 100% OOB; every other row has exactly 2 OOB
pixels -- its own column-0 and column-49 cells). This holds at step 1 on a
freshly-splatted, undegraded input, and stays essentially constant through
step 30 -- it is a standing property of the trained flow head at the
image boundary, not a compounding autoregressive side effect.

`|flow|` at the border ring is 2.2x the interior magnitude, and the
elevation is confined to the outermost pixel (falls to interior levels
immediately at 1px in) -- consistent with a genuine boundary effect from
the conv stack's implicit zero-padding (border pixels' receptive fields
include synthetic zero context that never occurs in the true, toroidal-
free training distribution), not a smooth multi-pixel dilation artifact.

### 4. Counterfactual: `padding_mode="border"` measurably changes the pattern, with no retraining

Same trained weights, only `grid_sample`'s `padding_mode` swapped from
`"zeros"` to `"border"` (clamp-to-edge) for the rollout:

```
-- padding_mode=zeros --
  step 16: edge=0.0133 interior=0.0460 max=0.0704
  step 20: edge=0.0169 interior=0.0450 max=0.0798
  step 25: edge=0.0078 interior=0.0076 max=0.0620
-- padding_mode=border --
  step 16: edge=0.0210 interior=0.0375 max=0.1680
  step 20: edge=0.0278 interior=0.0419 max=0.1446
  step 25: edge=0.0255 interior=0.0057 max=0.1526
```

Visually (see `counterfactual_padding_mode.png`, regenerate via the
script), `"border"` noticeably softens/removes the crisp thin grid-line
texture that `"zeros"` produces at steps 16-30 -- the pattern becomes a
smoother, more diffuse blob instead of sharp window-pane lines. It does
*not* fully fix the rollout (overall brightness still swings -- max values
actually go up in "border" mode, and the low-frequency banding/period-2
oscillation persist, both symptoms of the separate correction-head
overshoot issue in Problem 2). This isolates `padding_mode="zeros"` as a
causal driver specifically of the *sharp lattice-line texture*, while
confirming it is not the sole source of rollout instability.

## Recommended fix (code-level)

Do **not** change `DILATIONS` as the primary fix -- the evidence doesn't
support dilation gridding as the mechanism, so an HDC-style schedule
change (e.g. `(1, 2, 3, 5, 3, 2, 1)`) would not address the observed
artifact and shouldn't be prioritized for this problem.

Instead, in `model/net.py`'s `BounceNextFrameModel.forward`:

1. **Change `grid_sample`'s `padding_mode` from `"zeros"` to `"border"`.**
   This directly removes the hard zero discontinuity at the boundary --
   an out-of-bounds sample now reads the nearest valid edge pixel instead
   of exactly 0, so `correction_head` isn't forced to reconstruct the
   entire border ring from scratch every step. Confirmed above to soften
   the sharp lattice-line texture with zero retraining.

2. **Address the root cause, not just the symptom: switch
   `ResidualConvBlock`'s two `nn.Conv2d` layers from implicit zero padding
   to `padding_mode="replicate"`** (supported directly by `nn.Conv2d`).
   This removes the synthetic zero context that border pixels currently
   see during the forward pass, which is what's producing the anomalous
   2.2x flow magnitude at the border ring in the first place. This is
   more root-cause than (1) alone, which only patches how the warp handles
   an already-anomalous flow field.

3. (1) and (2) should be applied together and then the checkpoint
   retrained/fine-tuned to confirm the border-ring flow anomaly and OOB
   fraction drop -- re-running `scripts/investigate_gridding_artifact.py`
   against the new checkpoint gives a direct before/after comparison on
   the same metrics used here (border `|flow|` ratio, OOB fraction, FFT
   period of the interior crop).

4. This is architectural, not a hyperparameter/training-duration issue:
   the border-ring OOB defect is present on step 1 of a clean input, before
   any autoregressive compounding, so more epochs or a different learning
   rate would not resolve it -- retraining is only appropriate *after*
   these code changes, to let the model adapt to the new padding
   semantics, not as a first attempt to "train it away."

## Relation to Problem 2

Overlapping but distinct: the border defect explains the edge-localized
brightening and is a strong contributor to the sharp lattice *texture*,
but the low-frequency banding/period-2 oscillation in overall brightness
(see `findings-period2-oscillation.md`) persisted under the
`padding_mode="border"` counterfactual, consistent with that being the
separate correction-head overshoot/damping issue described there. Fixing
both is likely necessary to fully clean up the long-horizon rollout.
