# Investigation brief: padding-mode-driven mass drift (2026-09-21)

Context for a subagent investigating this. Read this file first — repo is
`/home/awebb/Research/Bounce`, run everything with
`PYTHONPATH=. python3 ...` from the repo root.

## Background: two prior fixes, now in tension

`model/net.py`'s `BounceNextFrameModel` has had two rounds of fixes this
session (see `docs/debugging/experiment-log.md` for the full history,
newest entries at the bottom):

1. **Gridding-artifact fix** (`git show b05e1a6`): `grid_sample`'s
   `padding_mode` changed from `"zeros"` to `"border"`, and the conv stack
   changed to `padding_mode="replicate"`. This fixed a confirmed bug where
   `"zeros"` padding turned an anomalous border-ring flow prediction into
   a guaranteed 100%-out-of-bounds perimeter every step, forcing
   `correction_head` to reconstruct the whole border from nothing each
   step (see `docs/debugging/findings-gridding-artifact.md`).
2. **Correction-centering + bicubic fix** (`git show 4ec5348`):
   `correction` is now centered per-channel per-step (`correction =
   correction - correction.mean(dim=(2,3), keepdim=True)`), and
   `grid_sample`'s interpolation `mode` changed from `"bilinear"` to
   `"bicubic"` (see
   `docs/debugging/findings-correction-drift-and-mass-dissolution.md`).
   This fixed an undamped-integrator drift in the VX channel caused by
   `correction_head`.

Retrained as `checkpoints/stage2_flownet_h12_v3.pt` (job 2824) with both
fixes applied. Verification found step-1 accuracy restored (0.88x
MSE-vs-copy-baseline) and the period-2 oscillation/horizontal-banding
gone — but **a VX-channel drift persists and is worse than expected**:
`mean1` (VX, channel 1) climbs monotonically from 0.016 at step 0 to
**1.30 by step 30** (the pre-retrain counterfactual for the centering fix
alone predicted a plateau around 0.14-0.16, not continued growth to 1.3).
An unbiased subagent review of a diagnostic grid render found a new
periodic diagonal ripple pattern emerging from rollout step ~12, plus
persistent top/bottom edge brightening.

## The newly-found root cause: `padding_mode="border"` is not mass-conserving under sustained directional flow

Found via a synthetic, no-model probe (repo root,
`PYTHONPATH=. python3 ...`):

```python
import torch
import torch.nn.functional as F

n = 50
x = torch.linspace(0, 1, n).unsqueeze(0).repeat(n, 1).unsqueeze(0).unsqueeze(0)
ys, xs = torch.meshgrid(torch.arange(n, dtype=torch.float32), torch.arange(n, dtype=torch.float32), indexing="ij")

def shift_repeatedly(mode, padding_mode, dx, steps):
    field = x.clone()
    means = [field.mean().item()]
    for _ in range(steps):
        sample_x = xs.unsqueeze(0) + dx
        sample_y = ys.unsqueeze(0)
        norm_x = sample_x / (n - 1) * 2 - 1
        norm_y = sample_y / (n - 1) * 2 - 1
        grid = torch.stack([norm_x, norm_y], dim=-1)
        field = F.grid_sample(field, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
        means.append(field.mean().item())
    return means

for pm in ("zeros", "border", "reflection"):
    for mode in ("bilinear", "bicubic"):
        means = shift_repeatedly(mode, pm, dx=0.6, steps=20)
        print(pm, mode, [f"{means[i]:.4f}" for i in (0, 5, 10, 15, 20)])
```

Results:

```
zeros      bilinear   0.5000 -> 0.4721  (slight decay, mass-losing)
zeros      bicubic    0.5000 -> 0.4782  (slight decay, mass-losing)
border     bilinear   0.5000 -> 0.7121  (steady growth, mass-pumping)
border     bicubic    0.5000 -> 0.7066  (steady growth, mass-pumping)
reflection bilinear   0.5000 -> 0.7095  (steady growth, mass-pumping -- nearly identical to border)
reflection bicubic    0.5000 -> 0.7051  (steady growth, mass-pumping -- nearly identical to border)
```

**No model involved at all** — this is a pure property of `grid_sample`
under repeated same-direction shifting. `"border"`/`"reflection"` clamp
out-of-bounds reads to the nearest valid edge pixel, so when content is
shifted persistently in one direction (the flow field consistently
pointing e.g. downward under gravity), the same high-value edge pixels
get re-sampled and duplicated into newly-revealed frame area every step,
pumping the spatial mean up monotonically. `"zeros"` doesn't pump (it
loses a little mass each step instead) but that's the mode the
gridding-artifact fix specifically moved away from, for a real,
previously-confirmed reason (100% OOB perimeter every step under
`"zeros"`, edge brightening from `correction_head` having to reconstruct
the border from nothing).

**These two fixes are now in tension**: `"border"`/`"reflection"` fix the
edge-reconstruction problem but drift the frame-wide mean of whatever
channel keeps shifting the same direction; `"zeros"` doesn't drift but
reintroduces the edge artifact. Centering `correction` (the second prior
fix) only removes the *correction-head-driven* part of drift — it does
nothing about this padding-mode-driven mechanism, which is a property of
the warp operation itself, not something more training can fix (gradient
descent on this loss can't undo padding-mode kernel behavior, and the
real flow field legitimately needs to point consistently
downward/outward under gravity for real physics, so "don't shift the
same direction for many steps" isn't an option either).

## What to investigate

1. **Confirm the mechanism transfers to the real model.** The synthetic
   probe above uses an idealized ramp field and constant shift direction.
   Verify (or refine) that this mechanism, not something else, explains
   the real model's VX drift — e.g. instrument
   `checkpoints/stage2_flownet_h12_v3.pt`'s actual rollout to check
   whether the predicted flow field is persistently biased in one
   direction (which would make the padding-mode-pumping mechanism
   applicable), and whether disabling the pump (see below) fixes the real
   rollout's drift, not just the synthetic test.
2. **Find a fix that keeps the gridding-artifact fix's benefit (no
   100%-OOB perimeter every step, no forced full-border reconstruction)
   while removing the directional mass-pumping.** Candidates to evaluate,
   with the same counterfactual-probe rigor used in this session's prior
   investigations (test on frozen weights before recommending a retrain):
   - **Explicit mass renormalization per step**: after `warped +
     correction`, rescale (at least the PROB channel, channel 0) so its
     frame-wide sum matches the pre-warp sum. Directly counteracts the
     pumping regardless of padding mode. Consider whether this makes
     sense for VX/VY (channels 1/2) too, or whether those need a
     different treatment (they're physical velocity fields, not a
     conserved "mass" quantity — pushing their frame-wide mean back to
     match the input every step may not be physically correct if the
     real dynamics legitimately have net acceleration, e.g. from
     gravity).
   - **A different boundary treatment**: is there a way to get
     `"border"`'s benefit (no hard zero discontinuity) without its
     duplication-under-shift property? E.g. bounding the flow field more
     tightly so predicted displacement rarely reaches the boundary in the
     first place (reducing `max_flow`), or explicitly detecting and
     damping/zeroing flow near the frame edge only, rather than changing
     the global padding mode.
   - **Revisit whether `"zeros"` + a bounded/damped
     `correction_head` (this session's other completed fix) is now
     sufficient** — the original zeros-mode edge problem was
     specifically that `correction_head` was forced to reconstruct
     the entire border from nothing every step; now that `correction`
     is bounded (0.2) and centered, re-test whether `"zeros"` produces
     an acceptable result rather than the severe edge brightening seen
     pre-fix (which was diagnosed against a checkpoint that had neither
     of the later fixes applied).
   State plainly which option you recommend and why, with the same kind
   of before/after quantitative evidence (a synthetic no-model probe for
   the mechanism, plus a check against the real trained model) used
   throughout this session's investigations.
3. Per this session's standing rule (see `~/.claude/CLAUDE.md` global
   memory and Bounce project memory
   `feedback_prefer_architecture_over_hparam_tuning`): diagnose before
   proposing a retrain. Any fix you recommend should be validated on
   frozen weights (a counterfactual swap, like the padding-mode and
   resampling-mode tests already done this session) before concluding
   retraining is needed.

## What NOT to do

- Do not edit `model/net.py` yourself — investigation only. Write up a
  concrete, code-level recommended fix; a human/coordinator will apply
  it.
- Do not default to "just retrain and see" as your primary method.
- Videos you render should go in `videos/` (repo root), not left only in
  `/tmp` — see `feedback_save_videos_to_videos_folder` project memory.
  `*.mp4` is already gitignored.

## Tools already available

- `scripts/render_diagnostic_grid.py --checkpoints name=path... --steps ...
  --out ...` — per-frame auto-scaled grid render.
- `docs/debugging/frame-artifact-review-prompt.md` — reusable unbiased
  subagent review prompt.
- `scripts/eval_step1_baseline.py` — step-1 MSE vs. copy-forward baseline.
- `scripts/investigate_gridding_artifact.py` — has reusable helper
  patterns (OOB-fraction probing, FFT peak-finding) though it currently
  hardcodes `mode="bilinear"` in its own `instrumented_forward` (stale —
  the real model now uses `"bicubic"`; update it in your own scratch copy
  if you need it rather than trusting its current numbers as-is).
- Checkpoint under test: `checkpoints/stage2_flownet_h12_v3.pt` (job
  2824, both prior fixes applied, `--channels 64 --depth 7 --max-flow 4.0
  --horizon 12`, seed 4738).
- Full test suite: `PYTHONPATH=. python3 -m pytest tests/ -q` (28 tests
  passing as of this brief).

## Write your findings to

`docs/debugging/findings-padding-mass-conservation.md`.
