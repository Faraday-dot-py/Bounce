# Investigation brief: correction-head damping + mass drift/dissolution (2026-09-21)

Context for a subagent investigating this. Read this file first, it has
all the background — repo is `/home/awebb/Research/Bounce`, run everything
with `PYTHONPATH=. python3 ...` from the repo root.

## Immediate problem: undamped drift in the VX channel

`model/net.py`'s `BounceNextFrameModel` (dilated conv stack + flow-warp
head, see `git show f0e709b`) was just fixed (`git show b05e1a6`) to stop
two bugs:

1. Period-2 oscillation — `correction_head` was unbounded; now
   `correction = tanh(correction_head(x)) * max_correction` (0.2).
2. Edge/gridding artifact — `grid_sample` now uses `padding_mode="border"`
   and the conv stack uses `padding_mode="replicate"`.

Retrained as `checkpoints/stage2_flownet_h12_v2.pt` (job 2823). Both bugs
are confirmed fixed (see `docs/debugging/findings-period2-oscillation.md`,
`docs/debugging/findings-gridding-artifact.md`, and the newest entry in
`docs/debugging/experiment-log.md`). But two new problems appeared:

- Step-1 MSE-vs-copy-baseline regressed to 1.08x (worse than doing
  nothing) from 0.78-0.80x pre-fix. Capping `correction` at 0.2 likely
  removed real predictive capacity the model needs.
- **A new slow instability**: per-step instrumentation (script below) shows
  the VX channel (channel 1) mean growing monotonically from ~0.016 at
  step 0 to ~1.43 by step 12, then decaying back toward 0 by step 30 —
  not oscillating (that's fixed), but drifting in one direction for many
  steps before collapsing. This produces visible horizontal banding in the
  rollout by step ~8-12 (confirmed by an unbiased subagent image review).

```python
import random
import numpy as np
import torch
import bounce
from model.dataset import make_scenario_uniform
from model.net import BounceNextFrameModel

model = BounceNextFrameModel(channels=64, depth=7)
model.load_state_dict(torch.load("checkpoints/stage2_flownet_h12_v2.pt", map_location="cpu"))
model.eval()

n = 50
rng = random.Random(4738)
balls = make_scenario_uniform(150, n, 2.3, rng)
G0 = bounce.make_grid(n)
bounce.splat_all(G0, n, balls, 0.75)
g0 = np.array(G0, dtype=np.float32)

x = torch.from_numpy(g0.transpose(2, 0, 1)).unsqueeze(0)
with torch.no_grad():
    for step in range(31):
        arr = x[0].numpy()
        print(step, arr[0].mean(), arr[0].max(), arr[1].mean(), arr[2].mean())
        x = model(x)
```

Bounding `correction`'s magnitude stopped the sign-flipping (negative
eigenvalue) instability from Problem 2, but a flat magnitude cap does
nothing to stop the correction from pushing the *same direction* for many
consecutive steps — a positive-drift failure mode a cap alone can't fix.
**Investigate a damping/decay mechanism** for the correction path: e.g. a
term that discourages `correction` from having a large moving average
(rather than just a large instantaneous magnitude), or an explicit decay
coefficient applied to the accumulated state, or normalizing/centering
`correction` per-channel per-step. Use your judgment on what's principled;
state plainly what you recommend and why, with evidence (instrumented
per-step magnitude/mean of `correction` over a rollout, before/after a
proposed change, same style of evidence as the period-2 investigation
that found this bug).

## Bigger-picture concern: balls drift apart / dissolve, across every
## architecture tried this session

The user's observation, stated directly: **"I think we need a way to keep
the balls together, they seem to drift apart no matter what we do."** This
has been true across every architecture this session has tried —
windowed-attention (checkerboard + occupancy hallucination), the first
flow-warp version (step-1 blur, then lattice/oscillation), and now this
fixed version (new drift). Don't treat this ticket as "fix the VX
instability and stop" — step back and investigate whether there's a
structural reason mass keeps spreading/dissolving under autoregressive
rollout regardless of the specific bug fixed each time.

**A concrete hypothesis worth checking, not yet investigated:** the
flow-warp head uses `F.grid_sample(..., mode="bilinear", ...)` every
single rollout step, and the output is fed back as input to the same warp
next step. Repeated bilinear resampling is a classic source of numerical
diffusion in semi-Lagrangian advection schemes (each resample averages a
cell with its neighbors, even when the "true" motion is a rigid
translation with no blurring) — over N compounding steps this can look
exactly like "the balls drift apart and dissolve" even with an otherwise
perfect flow field and zero correction. This would explain why the
symptom recurs across totally different architectures (windowed attention
had it too, for different underlying reasons) — it may be inherent to
*any* iterative resampling-based next-frame model unless something
explicitly counteracts it.

**What to investigate for this second part:**
1. Isolate whether numerical diffusion from repeated bilinear warping (as
   opposed to the correction head, or occupancy/mass leakage elsewhere) is
   a meaningful contributor to the dissolve/drift-apart symptom. A clean
   test: warp a synthetic sharp image (e.g. a single splatted ball, no
   model involved, real bounce.py physics ground truth positions) through
   N repeated pure-bilinear `grid_sample` warps using the *true* flow
   field each step (no model, no correction) and see how much a single
   ball's peak intensity/sharpness degrades purely from resampling, vs.
   how much the real model's rollout degrades. If pure-bilinear-with-true-
   flow degrades comparably to the real model's rollout, that's strong
   evidence this is a structural resampling-diffusion problem, not
   (only) a correction-head bug.
2. If confirmed as a real contributor, consider what's used in the video-
   prediction/optical-flow literature to counteract this: e.g. nearest-
   neighbor or higher-order (bicubic) resampling instead of bilinear
   (trade-offs), an explicit mass-conservation/renormalization step after
   each warp (rescale so total occupancy mass is preserved step to step),
   or predicting occupancy via a sharper mechanism than direct bilinear
   resampling (e.g. a small sharpening/super-resolution correction applied
   specifically to counteract known blur, or predicting a per-step
   "peakiness" regularization during training).
3. Distinguish this from the correction-head drift bug above — they may
   interact (a correction head might be learning to partially compensate
   for warp-blur by pushing mass values up, which given no damping term
   could be *why* it drifts in one direction for many steps rather than
   correcting evenly) but are conceptually different: one is a bug in a
   specific unbounded term, the other (if confirmed) is closer to an
   inherent property of the warping approach that needs an explicit
   counter-mechanism.
4. Per this session's standing rule (see `~/.claude/CLAUDE.md` global
   memory and Bounce project memory
   `feedback_prefer_architecture_over_hparam_tuning`): diagnose before
   proposing a retrain. State plainly whether the drift-apart symptom is
   (a) the correction-head damping bug alone, (b) inherent numerical
   diffusion from repeated bilinear warping, (c) both, or (d) something
   else you found — with evidence for whichever you conclude.

## What NOT to do

- Do not edit `model/net.py` yourself — this is investigation only, write
  up a concrete recommended fix (code-level) and a human/coordinator will
  apply it.
- Do not just retrain with different hyperparameters as your primary
  method — this is explicitly the pattern the user has corrected before
  this session (see `feedback_prefer_architecture_over_hparam_tuning`
  project memory). Diagnose the mechanism first.
- Videos you render should go in `videos/` (repo root), not left only in
  `/tmp` — see `feedback_save_videos_to_videos_folder` project memory.
  `*.mp4` is already gitignored.

## Tools already available

- `scripts/render_diagnostic_grid.py --checkpoints name=path... --steps ...
  --out ...` — per-frame auto-scaled grid render.
- `docs/debugging/frame-artifact-review-prompt.md` — reusable unbiased
  subagent review prompt, if a fresh visual read would help your
  investigation.
- `scripts/eval_step1_baseline.py` — step-1 MSE vs. copy-forward baseline.
- `scripts/investigate_gridding_artifact.py` — has reusable helper
  patterns (path-count coverage, FFT peak-finding, OOB-fraction probing)
  you may want to borrow from, though it's scoped to the edge-artifact
  investigation specifically.
- Full test suite: `PYTHONPATH=. python3 -m pytest tests/ -q` (28 tests
  passing as of this brief).

## Write your findings to

`docs/debugging/findings-correction-drift-and-mass-dissolution.md` —
cover both the immediate damping-term question and the bigger-picture
drift/dissolution hypothesis, with your diagnosis and a concrete
recommended fix (or fixes) for each.
