# Flownet architecture — open issues (2026-09-21)

Context for subagents investigating the two problems below. Read this
file first; don't ask for background, it's all here. Repo is
`/home/awebb/Research/Bounce`, run everything with
`PYTHONPATH=. python3 ...` from the repo root.

## Background

`model/net.py`'s `BounceNextFrameModel` was rewritten this session from
a windowed-attention (Swin-style) backbone to a dilated residual conv
stack + flow-warp head (see `git show f0e709b` for full rationale, and
`docs/debugging/experiment-log.md` for the investigation trail). Summary
of the current architecture:

- Stem: `Conv2d(3, channels, 3x3)`.
- `depth` (default 7) `ResidualConvBlock`s, each `norm-conv-norm-conv +
  skip`, with dilation drawn from the cyclic schedule `DILATIONS = (1,
  2, 4, 8, 4, 2, 1)` (module-level constant in `model/net.py`).
- `flow_head`: conv to 2 channels `(dx, dy)`, `tanh`-scaled by
  `max_flow` (default 4.0).
- `correction_head`: conv to 3 channels, added after the warp.
- Forward: build a per-pixel sampling grid from `pixel_coords + flow`,
  `F.grid_sample(g_t, sample_grid, mode="bilinear",
  padding_mode="zeros", align_corners=True)`, then `+ correction`.
- Both heads zero-initialized so the model starts as an exact identity.

Trained checkpoint under test: `checkpoints/stage2_flownet_h12.pt`
(job 2822 on Polaris, horizon=12, 80 epochs, `--channels 64 --depth 7
--max-flow 4.0`, dataset cache `checkpoints/dataset_cache_seq_h12.npz`).
Training entry point: `model/train.py` (`rollout_loss`,
`occupancy_weighted_mse` in `model/losses.py` — unchanged from before,
architecture-agnostic).

This checkpoint is a real improvement over every prior (windowed-
attention) checkpoint on single-step accuracy: step-1 MSE vs. a trivial
"copy the input frame forward" baseline is **0.80x** (script:
`scripts/eval_step1_baseline.py`), vs. 0.90x-1.35x for every prior
checkpoint (see experiment-log.md). Sparse ball structure visibly
survives through rollout step ~5 in a diagnostic grid render (vs.
dissolving to noise by step 1 in the old architecture).

But two problems remain at longer rollout horizons, found via
`scripts/render_diagnostic_grid.py` + the reusable review prompt in
`docs/debugging/frame-artifact-review-prompt.md`, and via a manual
per-step stats dump. **Do not just retrain with different
hyperparameters as the first move — assess whether each is an
architectural defect and say so if it is** (standing rule, see
`~/.claude/CLAUDE.md` global memory and Bounce project memory
`feedback_prefer_architecture_over_hparam_tuning`).

## Problem 1: grid/lattice pattern re-emerges by rollout step ~16-30

An independent subagent (fresh, no hypothesis primed, exact prompt from
`docs/debugging/frame-artifact-review-prompt.md`) read a diagnostic
grid of `stage2_flownet_h12` at steps 0,1,2,3,5,8,12,16,20,25,30 and
reported: sparse dots preserved through ~step 3-5, a sudden shift to
dense warm-toned noise/texture by step 8, then "a distinct grid/lattice
of light horizontal and vertical bands (like a tic-tac-toe or
windowpane pattern)" clearly visible by step 16, intensifying through
step 30 where it becomes "the dominant visual feature." Also reported
edge/corner brightening (bottom and right edges especially) from step 8
onward.

**Leading hypothesis (not yet verified): dilation gridding artifact.**
The `DILATIONS = (1, 2, 4, 8, 4, 2, 1)` schedule is the textbook setup
for the "gridding artifact" documented in dilated-CNN literature (e.g.
Wang et al., *Understanding Convolution for Semantic Segmentation*,
introducing Hybrid Dilated Convolution to fix it): when stacked dilation
rates share a common factor greater than 1 (here, all are powers of 2),
certain input pixels are never combined by any layer in the receptive
field, leaving a periodic gap. Under one forward pass this might be
subtle; under autoregressive self-feed (this model's own output becomes
next input, repeatedly) any small periodic bias could compound the same
way the old windowed-attention's periodic bias did.

Separately, the edge/corner brightening could be a distinct or related
issue: `grid_sample`'s `padding_mode="zeros"` means any sample point
that lands outside the grid (e.g. flow pushing a lookup off-frame)
reads as exactly 0, which is a hard discontinuity the flow field then
has to "route around" every step — `padding_mode="border"` (clamp to
edge) might remove this if it's a contributing factor.

**What to investigate:**
1. Verify or falsify the gridding-artifact hypothesis directly — e.g.
   visualize/compute each layer's effective per-pixel receptive field
   coverage for the current dilation schedule (a pixel is "covered" if
   some path of 3x3-dilated-conv taps reaches it); check whether
   there's a periodic pattern in coverage with period matching the
   observed lattice spacing in the diagnostic grid images (regenerate
   via the command in `docs/debugging/frame-artifact-review-prompt.md`
   with `--checkpoints stage2_flownet_h12=checkpoints/stage2_flownet_h12.pt`).
2. If confirmed, identify a concrete fix: e.g. a non-power-of-2
   dilation schedule with no shared common factor (HDC-style, e.g. `(1,
   2, 3, 5, 3, 2, 1)` or similar), and/or inserting periodic
   dilation=1 "smoothing" convs between dilated ones.
3. Separately check whether `padding_mode="zeros"` in the `grid_sample`
   call (`model/net.py` forward) is contributing to the edge
   brightening — e.g. instrument a rollout and check whether
   high-error/high-brightness regions correlate with pixels whose
   sampling coordinates land outside `[-1, 1]` (out of bounds).
4. Do NOT retrain to "see if it's better" as your primary method — form
   and state a concrete diagnosis with evidence (visualizations, computed
   coverage maps, instrumented stats) the way this session's prior
   investigations did for the windowed-attention checkerboard.

**Write your findings** to
`docs/debugging/findings-gridding-artifact.md` (create it) with your
diagnosis, evidence, and a concrete recommended fix (code-level, not
just "retrain"). Do not edit `model/net.py` yourself — this is
investigation only, a human/coordinator will apply the fix.

## Problem 2: period-2 brightness oscillation in autoregressive rollout

Manually dumping per-step frame stats for `stage2_flownet_h12`'s
rollout (same initial condition: n=50, 150 balls, seed=4738, grid built
via `bounce.make_scenario_uniform` + `bounce.splat_all` with
radius=0.75) showed a clean period-2 oscillation in mean/max PROB
(channel 0) brightness starting around step 14-16 and continuing to
step 30:

```
step 14: mean=0.029   step 18: mean=0.050   step 22: mean=0.030   step 26: mean=0.019
step 15: mean=0.009   step 19: mean=0.003   step 23: mean=0.007   step 27: mean=0.013
step 16: mean=0.039   step 20: mean=0.037   step 24: mean=0.024   step 28: mean=0.016
step 17: mean=0.003   step 21: mean=0.004   step 25: mean=0.010   step 29: mean=0.010
```

Every other frame is "bright", alternating with a "dim" frame -
roughly 5-15x swing in max brightness between adjacent steps. This is a
different problem from Problem 1 (though both are visible in the same
rollout and may interact) - a clean period-2 alternation is a signature
of an unstable/oscillating fixed point in the autoregressive map (like
a discrete dynamical system's 2-cycle bifurcation), suggesting the
effective per-step "gain" through the flow+correction path exceeds 1 in
some direction so error overshoots and flips sign each step rather than
damping.

Repro script (adapt as needed, this is NOT saved anywhere yet - write a
proper version as part of your investigation):

```python
import random
import numpy as np
import torch
import bounce
from model.dataset import make_scenario_uniform
from model.net import BounceNextFrameModel

model = BounceNextFrameModel(channels=64, depth=7)
model.load_state_dict(torch.load("checkpoints/stage2_flownet_h12.pt", map_location="cpu"))
model.eval()

n = 50
rng = random.Random(4738)
balls = make_scenario_uniform(150, n, 2.3, rng)
G0 = bounce.make_grid(n)
bounce.splat_all(G0, n, balls, 0.75)
g0 = np.array(G0, dtype=np.float32)

x = torch.from_numpy(g0.transpose(2, 0, 1)).unsqueeze(0)
frames = [x[0, 0].numpy().copy()]
with torch.no_grad():
    for _ in range(30):
        x = model(x)
        frames.append(x[0, 0].numpy().copy())
# frames[i].mean() / .max() per step shows the oscillation
```

**What to investigate:**
1. Confirm the oscillation is real (not an artifact of how the repro
   script indexes/aliases arrays) and characterize it further: is it
   isolated to the PROB channel (0) or present in VX/VY (1,2) too? Does
   it appear at the same step count regardless of initial condition
   (try other seeds)? Does it appear in a plain (non-scheduled-sampled,
   i.e. `sampling_p=0` teacher-forced) forward-only setting, or only
   under true self-feed autoregression?
2. Isolate which component drives it: instrument the model (e.g. a
   forked copy of `BounceNextFrameModel.forward` or hooks) to log the
   flow field's mean magnitude and the correction head's mean magnitude
   per step during the same rollout - does one of them oscillate in
   sync with the observed brightness oscillation?
3. Consider whether this is a known failure mode of residual/warp
   autoregressive predictors (e.g. lack of damping - the correction
   head has no mechanism discouraging it from overshooting) and what an
   architectural fix would look like (e.g. a learned or fixed damping
   coefficient on the correction term, a smaller correction_head init
   scale, clamping correction magnitude, or something else - use your
   judgment on what's principled vs. a band-aid).
4. Same rule as Problem 1: diagnose before recommending a retrain.
   State plainly whether this is architectural (e.g. "the correction
   head has no damping, that's the bug") vs. something else.

**Write your findings** to
`docs/debugging/findings-period2-oscillation.md` (create it) with your
diagnosis, evidence, and a concrete recommended fix. Do not edit
`model/net.py` yourself - investigation only.

## General notes for both subagents

- Full test suite: `PYTHONPATH=. python3 -m pytest tests/ -q` (27
  tests, all passing as of this investigation - use this to confirm
  you haven't broken anything if you write any test/probe scripts that
  touch real code, though you shouldn't need to edit `model/net.py`).
- Checkpoints and rendered videos/images are gitignored - only commit
  scripts, docs, and your findings file.
- Standing tool: `scripts/render_diagnostic_grid.py --checkpoints
  name=path... --steps ... --out /tmp/whatever.png`, and the review
  prompt in `docs/debugging/frame-artifact-review-prompt.md` for an
  unbiased description from a fresh subagent if useful to your own
  investigation.
- Be concrete and quantitative. Prefer computed evidence (coverage
  maps, per-step instrumented magnitudes, controlled repro) over visual
  impression alone.
