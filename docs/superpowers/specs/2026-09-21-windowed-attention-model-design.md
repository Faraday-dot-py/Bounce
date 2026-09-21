# Windowed-Attention Next-Frame Model — Design

Date: 2026-09-21

## Goal

A model that predicts the next frame of `bounce.py`'s grid state, trained at
small scale and able to run at much larger scale (up to 1M+ balls) without
retraining. Secondary goal: mechanistic interpretability of the learned
collision dynamics — the model's internals should be inspectable for
structure resembling the sim's actual contact-force mechanism.

## Constraints driving the design

- **Non-privileged input.** The model consumes `bounce.py`'s grid tensor
  (`G`, shape `n×n×3`), not a privileged list of ball objects. This is a
  deliberate choice: a more complex future sim may not have "balls" as a
  stable unit, but the grid representation stays valid regardless.
- **Train-small, run-large, no retraining.** Training happens at small scale
  (see Data pipeline below); inference must scale to `n` in the 1000s and
  ball counts up to 1M+. Any component whose parameter count or behavior
  depends on `n` or ball count breaks this requirement and is disqualified.
- **Local density has heavier tails at large scale.** Even with average
  ball density held constant between train and inference, local patches at
  1M balls are more likely to see extreme compaction than at training scale,
  purely from more opportunities for a rare clustering event to occur
  somewhere on the grid. Training data must deliberately include dense
  local configurations, not rely on random uniform placement to produce them.
- **Collisions are contact-only.** Interactions in this sim only happen
  between objects close enough to touch. This justifies a locality bias in
  the architecture (as opposed to full global attention) as a good match to
  the physics, not just a computational compromise.

## I/O contract

- **Input** `G_t`: `n×n×3` tensor.
  - Channel 0 (`PROB`): combined occupancy likelihood, saturating via
    `1 - exp(-sum_w)` (see `bounce.py:splat_all`) — bounded in `[0, 1)`
    regardless of local overlap density.
  - Channel 1 (`VX`), Channel 2 (`VY`): probability-weighted mean velocity
    components.
- **Output**: `G_{t+1} = G_t + ΔG`, same shape as input. The model predicts
  `ΔG` (a residual/delta), not the absolute next state — see Output head.
- Granularity: one `step()` (render-frame) worth of physics time (`dt`),
  not one of `bounce.py`'s internal integration `substeps`.

## Architecture

### 1. Tokenization

Patchify `G_t` into non-overlapping `2×2` grid-cell blocks (`2×2×3 = 12`
values per patch). Patch size is fixed in **grid cells**, not as a fraction
of `n` — this keeps a patch's physical meaning (and the statistics a
trained model has learned to expect from one) consistent as `n` grows.
Chosen relative to ball size: default `radius=0.75` → diameter ≈ 1.5 grid
cells, so a `2×2` patch is on the order of one ball's footprint, giving the
model sub-ball spatial resolution.

Each patch is projected to embedding dimension `d` via one shared linear
layer (`12 → d`), applied identically to every patch. This layer is the
**only** component coupled to channel count — the extension point if a
future channel (e.g. per-ball radius, currently a shared scalar across all
balls in `bounce.py`) is added.

### 2. Windowed self-attention body

A stack of transformer blocks. Each block:

- **Windowed multi-head self-attention**: attention is restricted to
  patches within a fixed-size window (`8×8` patches = `16×16` grid cells,
  roughly 10x a ball's diameter — enough margin to contain a local
  multi-ball cluster). Window size is fixed in grid cells, same
  resolution-independence reasoning as patch size. This is what makes
  per-frame compute scale linearly (not quadratically) with token count,
  and is what makes train-small/run-large possible at all: full
  O(tokens²) global attention cannot make the ~200x jump in token count
  between training scale and the 1M-ball target.
- **Learned relative positional bias**: attention scores get a learned
  bias term as a function of the integer offset between two patches within
  a window. Table size is bounded by window extent, so it's fixed
  regardless of `n`. Chosen over absolute positional embeddings (which
  cannot extrapolate past their trained size at all) and over continuous/
  rotary-style relative encoding (less standard for bounded 2D windows,
  and the discrete bias table is directly inspectable post-training —
  serves the interpretability goal).
- **Shifted windows (Swin-style)**: window partitioning alternates by a
  half-window offset every other block. Plain fixed, non-overlapping
  windows never let information cross what was a window boundary at any
  depth; shifting reintroduces that cross-boundary flow (the thing a
  sliding CNN kernel gets for free via overlapping receptive fields, which
  a hard window partition does not).
- Standard per-token MLP follows attention, as in a typical transformer
  block.

Depth/width (e.g. 6 blocks, `d=128`, 4 heads) are empirical starting
points, tuned during implementation — not fixed by this design.

### 3. Output head

Per-token linear "unpatchify": `d → 12`, reshaped to `2×2×3`, tiled back
into a full `n×n×3` grid (`ΔG`), added residually to `G_t` to produce the
prediction. Same weight-sharing property as tokenization: no step depends
on total grid size, preserving train-small/run-large transfer.

Chosen over a global-bottleneck encoder-decoder (pool to a fixed vector,
decode via upsampling): that approach's output resolution is
architecturally fixed at training time and has no way to decode to a grid
~200x larger at inference — a structural incompatibility with the scale
requirement, not just a worse option.

Chosen as residual/delta rather than absolute prediction: easier
optimization target for small `dt` (frames are mostly similar), and `ΔG`
is a signal directly analogous to "what changed this step" — in the same
spirit as an explicit force channel, without needing one.

## Loss

Per-channel MSE between predicted and ground-truth `ΔG`. `PROB` is bounded
`[0, 1)` while `VX`/`VY` are unbounded velocity-scale; per-channel loss
weighting/normalization will likely be needed so one channel doesn't
dominate the gradient. Specific weights are a tuning question deferred to
implementation.

## Data pipeline

Training pairs `(G_t, G_{t+1})` are generated directly from `bounce.py`'s
`step()`. Curriculum includes:

- uniform-random spawns (baseline, `init_balls`),
- pre-clustered/compact spawns (balls placed in a tight region rather than
  uniformly),
- long gravity-settle rollouts (run until balls pile against a wall/corner
  and stay there).

The clustered and settled scenarios exist specifically to expose the model
to near-saturated local patches during training, since local density tails
are heavier at large ball counts than a matched-average-density training
run would otherwise produce.

## Validation plan (staged, cheapest-first)

1. **Training scale**: `n=50`, 50–250 balls. Held-out single-step
   prediction error.
2. **Rollout stability**: autoregressive multi-step prediction at training
   scale — confirm error doesn't compound catastrophically.
3. **First scale-transfer check**: generalize to 500 balls (no retraining).
4. **Mid-scale check**: `n=300`, ~50k balls (unseen during training) —
   catch scale-transfer failures cheaply before the expensive extreme.
5. **Target scale**: up to 1M+ balls, `n` scaled to hold density
   comparable to training (exact scaling rule TBD at implementation time).

Compute: Polaris (not TIDE — see project memory).

## Open questions deferred to implementation

- Exact depth/width/head-count for the transformer body.
- Per-channel loss weighting scheme.
- Whether `n` scales strictly with `sqrt(ball_count)` to hold density
  constant at each validation stage, or some other rule.
- Whether the mid-scale (`n=300`) and target-scale (1M-ball) runs need
  further curriculum adjustments once early results are in.
