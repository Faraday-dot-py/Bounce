# Flownet — open issues, round 2 (2026-09-22)

Context for subagents investigating the three problems below. Read this
file first; don't ask for background, it's all here. Repo is
`/home/awebb/Research/Bounce`, run everything with
`PYTHONPATH=. python3 ...` from the repo root.

## Background — read before starting

`model/net.py`'s `BounceNextFrameModel` (dilated-conv + flow-warp head)
has had six architectural fixes landed across this session's earlier
investigations (see `docs/debugging/experiment-log.md` for the full
trail, and its docstring): windowed-attention replacement, bounded+
centered correction head, border/replicate padding (fixed the sharp
lattice-line texture + edge brightening from `findings-gridding-
artifact.md`), bicubic resampling, PROB mass renorm + soft-threshold
(fixed quilting), VX/VY mean recentering (fixed drift).

With all of those fixed, `stage2_flownet_h12_v6.pt` is the adopted
default. `stage2_flownet_h12_v7.pt` adds a peak-magnitude loss term
(`model/losses.py`, `peak_weight=0.1`) that fixes v6's uniform-blur
collapse but **root-causes to a deeper, unresolved problem**: dense
per-cell regression under `occupancy_weighted_mse` has two cheap-but-
wrong hedges available to it once a ball's true position is uncertain
(the normal state past the first few autoregressive steps) —
"blur everywhere" (v6's failure mode) and "commit to what's easy, give
up on what's uncertain" (v7's failure mode, once blur is closed off).
Full diagnosis: `docs/debugging/findings-peak-decay-dissolution.md`
(read this in full — it's the most important context for problems 1
and 2 below).

A 100-step v7 rollout render (`videos/stage2_flownet_v7_rollout_
comparison_100step.mp4`, seed 4738) was reviewed by an unbiased
subagent and found the "give up" collapse is progressive: sharp match
degrades by step 5, most content is near-black by step 20-50 except 2-3
static bright blobs, those fade out by step 70-80, and **a faint
diagonal ripple that appears around step 20-25 grows to dominate the
entire frame as a fine periodic checkerboard/lattice by step 90-100** —
never recovering. Full writeup: the last section of `docs/debugging/
experiment-log.md`. This is new evidence, not yet investigated —
problem 3 below.

Three next-step options were identified in `findings-peak-decay-
dissolution.md` but not resolved (roughly in order of effort): (1)
regional/local peak-preservation instead of one global peak, (2)
explicitly bound and report the honest rollout horizon this
architecture can support, (3) a genuinely different output
representation (out of scope for a subagent investigation — a redesign
decision, not something to investigate bottom-up). Problems 1 and 2
below correspond to options (1) and (2).

**Standing rule for all three subagents**: diagnose before recommending
a retrain (`~/.claude/CLAUDE.md` global memory,
`feedback_prefer_architecture_over_hparam_tuning` project memory). Be
concrete and quantitative — prefer computed evidence (synthetic probes,
instrumented stats, FFT/coverage analysis) over visual impression
alone. Do not edit `model/net.py`, `model/losses.py`, or `model/
train.py` — investigation only; write findings, a human/coordinator
applies fixes.

---

## Problem 1: is regional/local peak-preservation a viable fix?

`findings-peak-decay-dissolution.md`'s Part 5 ran one isolated synthetic
probe of a tile-based local-max peak term (vs. v7's single global-max
peak term) and found it did **not** show a clearly larger penalty for
the give-up strategy than the global version did, in that one probe —
inconclusive, not a real test.

**What to investigate:**
1. Design a more thorough synthetic test harness than the one-shot probe
   in Part 5: multiple ball counts/densities, multiple degrees of
   position drift (not just the one two-ball scenario), and at least 2-3
   tile sizes (e.g. 4x4, 8x8, 16x16 cells on the 50x50 grid) for the
   local peak term (`F.max_pool2d`-based, matching each tile's local max
   rather than one frame-wide max).
2. For each configuration, compute whether "give up" or "commit to
   drifted position" is loss-optimal, the way the existing probe does.
   Determine whether there's a tile size (or other formulation — e.g.
   a peak term applied per-connected-component, or top-k local maxima)
   where committing is reliably cheaper than giving up, across the
   scenarios tested.
3. If you find a promising formulation, implement it as a standalone
   loss function variant (don't touch `model/losses.py` — write it in a
   scratch/investigation script instead) and re-run the same synthetic
   comparison used in the existing findings doc so results are directly
   comparable.
4. State plainly whether regional peak-preservation is a real fix, a
   partial mitigation, or does not resolve the underlying blur-vs-
   give-up tradeoff at all (i.e., confirm or falsify option (1) from
   `findings-peak-decay-dissolution.md`).

**Write findings** to `docs/debugging/findings-regional-peak-loss.md`
(create it).

---

## Problem 2: characterize the honest rollout horizon

Rather than continuing to chase long-horizon quality, quantify the
step count at which multi-ball position uncertainty becomes
irreducible for this point-estimate-regression architecture — i.e.
find the horizon this architecture can honestly support, and how it
depends on scenario parameters.

**What to investigate:**
1. Using `stage2_flownet_h12_v6.pt` (uniform-blur failure mode) and
   `stage2_flownet_h12_v7.pt` (give-up failure mode), run rollouts
   across several ball counts/densities and at least 3 seeds each (this
   session's convention: seed 4738 first, then others as needed — see
   `~/.claude/CLAUDE.md`).
2. Define and compute a concrete, reproducible degradation metric per
   step (e.g. per-ball position error against ground truth if
   individual balls can be tracked/matched, or a structural metric like
   number of distinguishable local maxima vs. ground truth's ball
   count, or the `max0`/peak-intensity metric already used elsewhere in
   this investigation — use your judgment, but justify the choice).
3. Identify the step count (or range) at which the metric crosses a
   clear "no longer usable" threshold, for both checkpoints, and
   characterize how that threshold shifts with ball density (more balls
   → more collisions → uncertainty should compound faster).
4. Report a concrete recommended honest horizon (e.g. "N steps for
   sparse scenes, M steps for dense ones") a downstream user of this
   model should treat as the supported range, with the evidence behind
   the number — not just an impression from watching a video.

**Write findings** to `docs/debugging/findings-honest-horizon.md`
(create it).

---

## Problem 3: what is the long-horizon periodic lattice in the v7 100-step rollout?

The 100-step v7 rollout review found a faint diagonal ripple appearing
around step 20-25 that grows to dominate the frame as a fine periodic
checkerboard/lattice by step 90-100, once the "give up" collapse has
emptied out most real content. This is visually reminiscent of two
previously-diagnosed-and-addressed issues from earlier this session:
the sharp lattice-line texture from `findings-gridding-artifact.md`
(root-caused to `grid_sample` `padding_mode="zeros"` + conv zero-
padding at the border, fixed via border/replicate padding — check
`model/net.py`'s current padding mode to confirm this fix is actually
in place in the checkpoint under test) and the period-2 brightness
oscillation from `findings-period2-oscillation.md` (root-caused to
correction-head overshoot, addressed via bounded+centered correction).

**What to investigate:**
1. Regenerate or reuse the 100-step v7 rollout
   (`scripts/render_generic_flownet_rollout.py` or equivalent — check
   `scripts/` for the tool added this session, `59e1190`) and extract
   raw PROB-channel frames at steps 20, 40, 60, 80, 100 (not just the
   rendered video) for direct numerical analysis.
2. FFT each frame's interior crop (same method as `findings-gridding-
   artifact.md`'s Part 2) and determine the dominant spatial period of
   the step-90-100 lattice. Compare directly against the periods found
   in that earlier investigation (dilation-scale periods of 2/4/8px
   were ruled out there in favor of a coarser 13-40px banding).
3. Determine whether this lattice is present (even if much fainter,
   masked by real content) in the *same rollout*'s earlier steps
   (0-20) and in a v6 rollout at the same step range — i.e. is this a
   pre-existing low-amplitude bias in the architecture that both v6 and
   v7 always had, simply unmasked once give-up/blur empties the frame
   of competing real content, or something specific to v7's peak-loss
   training?
4. Check whether the same `padding_mode="border"` counterfactual used
   in `findings-gridding-artifact.md` (swap `grid_sample`'s
   `padding_mode` at inference time only, no retraining) changes this
   long-horizon lattice's amplitude or period, the same way it did for
   the earlier short-horizon lattice.
5. State plainly whether this is the *same* mechanism as one of the
   two earlier-diagnosed issues resurfacing, a novel third periodic
   bias, or fully explained by the give-up dynamics interacting with
   already-known biases (be specific about which).

**Write findings** to `docs/debugging/findings-long-horizon-lattice.md`
(create it).

---

## General notes for all three subagents

- Full test suite: `PYTHONPATH=. python3 -m pytest tests/ -q` (30
  tests as of this investigation — use to confirm you haven't broken
  anything if you write scripts that touch real code).
- Checkpoints and rendered videos/images are gitignored — only commit
  scripts, docs, and your findings file. Do not commit — write files
  and report back; the coordinator will review and commit.
- Save any new videos to `videos/`, not repo root or `/tmp` only.
- Standing tools: `scripts/render_diagnostic_grid.py`, `scripts/
  render_generic_flownet_rollout.py`, and the review prompt in
  `docs/debugging/frame-artifact-review-prompt.md` for an unbiased
  video description if useful to your own investigation.
