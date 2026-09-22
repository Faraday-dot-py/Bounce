# Flownet — open issues, round 3 (2026-09-22)

Context for the subagent investigating the problem below. Read this file
first; don't ask for background, it's all here. Repo is
`/home/awebb/Research/Bounce`, run everything with `PYTHONPATH=. python3
...` from the repo root.

## Background — read before starting

Read, in this order: `docs/debugging/findings-peak-decay-dissolution.md`
(root diagnosis: `occupancy_weighted_mse`, evaluated on synthetic
scenarios, provably prefers a diffuse mass-matched blob or an outright
give-up over a sharp-but-slightly-mispositioned ball, once positional
uncertainty exceeds about one ball radius — the normal state past the
first few autoregressive steps of a real multi-ball rollout), and
`docs/debugging/findings-regional-peak-loss.md` (this session's
follow-up: regional/local peak-preservation terms are a real but partial
mitigation — every tile/window formulation tested either gets masked by
a neighboring correctly-predicted ball at high density, or is capped by
its own fixed size at high drift; turning up `peak_weight` amplifies
whichever formulation is used but doesn't remove the structural
tradeoff).

The current loss (`model/losses.py`, `occupancy_weighted_mse`) reduces
the per-pixel squared error to a scalar via a **plain occupancy-weighted
mean**: `sq_err = (pred - target) ** 2`, then summed/divided by
`occ_count + bg_weight * bg_count` (see lines 15-26). Every pixel's
squared error contributes to that mean in direct proportion to its
magnitude and which mask (`occ`/`bg`) it falls in — there is no
per-pixel reweighting by how confidently-wrong that pixel is. This is
the specific thing to interrogate: does replacing the plain mean with a
**nonlinear, error-magnitude-dependent reduction** (a focal-loss-style
term) change the give-up-vs-commit incentive that the peak and regional
peak terms only partially fixed?

## What to investigate

Extend the existing synthetic single-step probe methodology from
`findings-peak-decay-dissolution.md` Part 5 and `findings-regional-
peak-loss.md` (same give-up vs. commit scenario builder, same swept
ball counts {2, 5, 10, 20} and drift levels {0.5, 1, 2, 4, 8} px, same
15-trials-per-cell convention, same `weights=[1.0, 0.1, 0.1]`,
`bg_weight=0.05` defaults from `model/train.py` — read that probe
methodology directly out of `findings-regional-peak-loss.md`'s "Method"
section rather than re-deriving it, so results stay comparable) to test
whether a focal-style reweighting of the base loss changes which
strategy (give-up vs. commit-but-drifted) is loss-optimal.

1. Implement, standalone in a scratchpad script (do not edit `model/
   losses.py`), at least these two focal-style variants of the
   `prob_loss` reduction in `occupancy_weighted_mse`:
   - **Power reweighting**: replace `(pred-target)**2` with
     `(pred-target)**2 * |pred-target|**gamma` (i.e. raise the error to
     a higher power before averaging) for gamma in {1, 2, 4} — this
     up-weights large-error pixels relative to small-error ones
     directly.
   - **Confidence-modulated focal term** (true focal-loss style, as in
     Lin et al. *Focal Loss for Dense Object Detection* and CenterNet's
     penalty-reduced heatmap loss): weight each pixel's error by a
     factor that grows with how "wrong" the *prediction's own
     confidence* was at that pixel (e.g. `(1 - min(pred, target))**gamma`
     or similar — use your judgment on the exact functional form, but
     justify it and cite the mechanism, the same way the existing docs
     did for the peak term).
   Keep the existing `occ`/`bg` masking and `bg_weight` structure
   unchanged around whichever reweighting you test, so you're isolating
   the effect of the reduction, not re-deriving the occupancy mask logic.
2. For each variant and each gamma, run the same give-up-vs-commit
   win-rate sweep used in `findings-regional-peak-loss.md`, and produce
   the same kind of win-rate table (balls x drift) so it's directly
   comparable to the `base`/`global`/`window5`/etc. columns already
   published there.
3. Specifically check whether focal reweighting changes the *mechanism*
   of the tradeoff, not just the win rate: does it still exhibit a
   neighbor-masking-style failure at high density (probably not, since
   this reduction is inherently local/per-pixel with no region-sharing
   step — confirm or refute this directly), and does it still saturate
   below full coverage at the hardest (density, drift) cells, or does it
   actually resolve them? Be concrete with numbers, not impression.
4. Also test combining a focal reweighting with the best regional
   formulation from `findings-regional-peak-loss.md` (`window5`) to see
   if they're complementary (focal fixes the "commit is punished too
   hard" mechanism, window5 fixes the "no signal near the true position
   at all" mechanism identified there) or redundant.
5. Sanity-check against the same kind of adversarial check that
   disqualified `topk8` in `findings-regional-peak-loss.md` — confirm a
   focal-reweighted loss doesn't inadvertently reward hallucinated
   high-confidence wrong predictions over honest give-up.
6. State plainly: does focal-style reweighting change the fundamental
   blur-vs-give-up tradeoff, is it a partial mitigation like the
   regional terms, or does it not help at all — with the win-rate
   evidence to back whichever conclusion you reach.

**Standing rule** (same as the other three investigations this
session): diagnose before recommending a retrain
(`~/.claude/CLAUDE.md` global memory,
`feedback_prefer_architecture_over_hparam_tuning` project memory). Be
concrete and quantitative. Do not edit `model/net.py`, `model/
losses.py`, or `model/train.py` — investigation only; a human/
coordinator applies any fix that comes out of this. Do not commit.

## General notes

- Full test suite: `PYTHONPATH=. python3 -m pytest tests/ -q` (30
  tests as of this investigation — use to confirm you haven't broken
  anything if you write scripts that touch real code).
- Checkpoints and rendered videos/images are gitignored — only commit
  scripts, docs, and your findings file (and even those, only if asked
  — by default just write files and report back).
- Save any new videos to `videos/`, not repo root or `/tmp` only.

**Write findings** to `docs/debugging/findings-focal-reweighted-loss.md`
(create it), and report back a concise summary (under 300 words) of
your conclusion.
