# Flownet — open issues, round 4 (2026-09-22)

Context for the subagent investigating the problem below. Read this file
first; don't ask for background, it's all here. Repo is
`/home/awebb/Research/Bounce`, run everything with `PYTHONPATH=. python3
...` from the repo root.

## Background — read before starting, important nuance up front

**Global PROB mass conservation is already enforced architecturally,
not via a loss term.** Read `model/net.py`'s `BounceNextFrameModel`
docstring (lines ~60-94) and `docs/debugging/findings-padding-mass-
conservation.md` and `docs/debugging/findings-quilting-artifact.md`
first. Summary: after each warp step, the PROB channel is soft-
thresholded (`relu(prob - prob_threshold)`) then **multiplicatively
rescaled** so its frame-wide sum exactly matches the pre-warp frame's
sum, every single step, by construction — this is a hard, always-on
renormalization inside the forward pass, not something the loss
incentivizes or could fail to enforce. VX/VY are separately
additively recentered (their frame-wide mean, not sum, since they're
signed velocity fields not a conserved quantity).

So "investigate a mass conservation loss" needs to be read carefully —
the naive version (a loss term penalizing frame-wide total mass drift)
is very likely already fully redundant with the existing hard renorm
and would add nothing. The two things actually worth investigating,
scoped below, are (1) whether that existing hard renorm mechanism is
itself *contributing to* the give-up/blur pathology documented in
`docs/debugging/findings-peak-decay-dissolution.md`, and (2) whether a
**local/regional** (not global) mass-conservation loss — a genuinely
different operator than anything tried so far — changes the give-up
incentive.

Also read `docs/debugging/findings-regional-peak-loss.md` in full
(this session's investigation into regional *peak*-based loss terms:
partial mitigation, undermined by neighbor-masking at high density and
a fixed-region ceiling at high drift, no formulation escaped both) —
the local mass-conservation idea below uses the same synthetic
give-up-vs-commit probe methodology, but compares **integrated sum**
per region instead of **max** per region, which is a different
operator with potentially different failure characteristics (sum is
continuous/graded rather than the near-binary "is there a peak"
check that drove peak-loss's neighbor-masking failure).

## What to investigate

### Part A: does the existing hard global renorm amplify the give-up blob?

Hypothesis to test, not assumed true: when the model "gives up" on most
balls and concentrates confidence into one surviving region (the
failure mode documented in `findings-peak-decay-dissolution.md` and
observed directly in the 2-ball and 100-step v7 rollout reviews), the
existing hard mass-renorm step rescales that region's mass *upward* to
compensate for the abandoned balls' missing mass (since it force-matches
the frame-wide sum every step regardless of *where* mass is) — which
could make the surviving blob brighter/more concentrated than it would
otherwise be, i.e. the renorm mechanism itself could be amplifying the
give-up strategy's payoff rather than being neutral to it.

1. Instrument a real rollout (`stage2_flownet_h12_v7.pt`, seed 4738, the
   session's standard scenario — see `model/dataset.make_scenario_uniform`
   usage in `scripts/render_flownet_rollout_video.py` for the exact
   construction) and directly measure, per step: total PROB mass before
   vs. after the renorm rescale, and the rescale factor applied. Confirm
   or refute that the rescale factor grows as the give-up collapse
   progresses (i.e. is it in fact inflating a shrinking pool of "real"
   mass to match a much larger target sum, which necessarily makes
   whatever mass remains locally brighter)?
2. Run a counterfactual: same trained weights, disable the renorm
   rescale for inference only (keep the soft-threshold, skip the
   multiplicative rescale — or use a fixed scale of 1.0) and compare the
   give-up blob's brightness/concentration trajectory against the normal
   (renorm-on) rollout, same seed. Does removing renorm change how
   concentrated/bright the surviving blob gets, or does the give-up
   collapse look basically the same either way?
3. State plainly whether the hard renorm mechanism is a meaningful
   contributor to the give-up failure mode's visual character (a
   concentrated bright blob rather than a dim residual), a neutral
   bystander, or something in between — with the instrumented numbers
   to back it.

### Part B: local/regional mass-conservation loss — a genuinely new operator

1. Extend the synthetic give-up-vs-commit probe from `findings-
   regional-peak-loss.md` (same scenario builder, same swept ball counts
   {2, 5, 10, 20} and drift levels {0.5, 1, 2, 4, 8} px, same 15-trials-
   per-cell convention — read that doc's "Method" section directly
   rather than re-deriving it) with a new loss term: **per-tile
   integrated PROB mass matching** — `F.avg_pool2d` (or sum-pool) the
   PROB channel into tiles (test at least tile sizes 4/8/16 cells, same
   granularities already tested for the peak variant so results are
   directly comparable) and compare pred's vs. target's summed mass per
   tile, e.g. `((pred_tile_mass - target_tile_mass) ** 2).mean()`.
2. Run the same win-rate table (balls x drift) as `findings-regional-
   peak-loss.md` for each tile size, and compare directly against that
   doc's `tile4`/`tile8`/`tile16` (peak-based) columns — same tile
   geometry, different operator (sum vs. max). Does a give-up ball still
   get masked by a neighboring correctly-predicted ball sharing its
   tile (the mechanism that broke tile-based peak formulations), or does
   the graded/continuous nature of a sum-based comparison behave
   differently (e.g. partial credit vs. the peak formulation's
   all-or-nothing "is *a* peak present" check)?
3. Also test a ball-centered window variant (mirroring `window5` from
   the peak investigation) using summed mass in a 5x5 window instead of
   max, for direct comparison to that doc's best-performing formulation.
4. State plainly whether local mass-conservation is a meaningfully
   different (better, worse, or equivalent) lever than the regional
   peak terms already investigated, with the win-rate evidence to back
   it.

**Standing rule** (same as this session's other investigations):
diagnose before recommending a retrain (`~/.claude/CLAUDE.md` global
memory, `feedback_prefer_architecture_over_hparam_tuning` project
memory). Be concrete and quantitative. Do not edit `model/net.py`,
`model/losses.py`, or `model/train.py` — investigation only; a human/
coordinator applies any fix that comes out of this. Do not commit.

## General notes

- Full test suite: `PYTHONPATH=. python3 -m pytest tests/ -q` (30
  tests as of this investigation).
- Checkpoints and rendered videos/images are gitignored.
- Save any new videos to `videos/`, not repo root or `/tmp` only.

**Write findings** to `docs/debugging/findings-mass-conservation-loss.md`
(create it), and report back a concise summary (under 300 words) of
your conclusion, covering both Part A and Part B.
