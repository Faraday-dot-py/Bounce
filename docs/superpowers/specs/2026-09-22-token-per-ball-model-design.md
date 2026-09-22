# Token-Per-Ball Next-Frame Model — Design

Date: 2026-09-22

## Goal

A next-frame prediction model that represents each ball as an explicit,
persistent token (position, velocity, hidden state) rather than as an
implicit pattern in a dense grid. Primary motivation: collision outcomes
are a chaotic/sensitive function of exact contact position and velocity,
and both prior architectures (flow-warp, windowed-attention) only ever see
a single previous grid frame, so per-ball velocity is an *inferred*
quantity, not a *tracked* one. A persistent per-token state is a direct
answer to that: velocity carried forward in the token's own state, not
re-estimated from two blurry frames each step.

## Why not extend the existing grid-based architectures

`stage2_flownet_h12_v6.pt` remains the project default; six prior
investigations (VX/VY renormalization, quilting, peak-decay, focal
reweighting, mass-conservation, richer temporal input — see
`docs/debugging/experiment-log.md`) improved or ruled out grid-space
fixes without touching the underlying representation. The recovered
windowed-attention architecture (`docs/superpowers/specs/2026-09-21-windowed-attention-model-design.md`)
independently confirmed a checkerboard/lattice artifact traced to its
window-partition boundaries, unrelated to this collision-noise question.
This design deliberately avoids windowed image-space attention and
`grid_sample`-based warping entirely — the failure modes documented for
both are architectural to pixel-space processing, not something a
token-based model inherits by construction.

## Constraints driving the design

- **Arbitrary ball count.** Must work for any N without a fixed-size
  state vector — this ruled out a naive fixed-slot per-ball design (see
  discussion below). Tokens are a variable-length set; the network
  (attention over a neighbor graph) has no dependence on set size.
- **Train-small, validate-large (tileable).** Same requirement as the
  windowed-attention design: train at small grid size / low ball count,
  validate by tiling to larger grids. This is only achievable if
  per-token updates depend on *local* neighborhood only — this is the
  main argument for proximity attention over global attention (below).
- **Must stay comparable to the existing pipeline.** Loss and evaluation
  (`docs/debugging/experiment-log.md`, honest-horizon methodology) are
  all defined in grid space against `bounce.py`'s rendered output. This
  model's output must still be a `G_{t+1}` grid tensor, produced by
  rasterizing tokens through the same formula `bounce.py` uses to
  generate ground truth, so MSE numbers stay directly comparable to
  v6/v7/v8 baselines.
- **Occlusion is a real information loss, not just a detection
  difficulty.** Per `bounce.py`'s `splat_all`, channels 1-2 hold a
  probability-weighted *average* velocity wherever two balls' disks
  overlap — the exact per-ball velocity is destroyed in the input at
  the moment of contact, which is also the moment collision dynamics
  need to be predicted correctly. The design must not assume it can
  read clean per-ball state from an overlapping frame.

## Rejected simpler alternative: fixed per-ball state vector

Considered and rejected: a fixed-size state vector (or padded max-N with
masking) per ball. Breaks arbitrary-N generalization outright — the
current grid representation's key property is that it's agnostic to
ball count, and a fixed-slot vector reintroduces a hard cap. Any
per-ball approach for this project must be set-based / variable
cardinality.

## I/O contract

- **Input**: `bounce.py`'s grid tensor `G_t` (`n×n×3`), same as the
  existing architectures — no privileged access to true ball state.
  Sequence of a few frames at initialization only (see Token
  initialization).
- **Output**: `G_{t+1}`, produced by rasterizing the token set's
  predicted next state through `bounce.py`'s existing splat/saturation
  formula (`1 - exp(-sum_w)` for `PROB`, weighted average for `VX`/`VY`).
  Not predicted directly by the network — see Rasterization head below.

## Architecture

### 1. Token initialization

On the first frame(s) of a sequence, run blob detection on `PROB`
(connected-component / local-maxima, matching the known ball radius
footprint) to establish one token per detected ball: initial (x, y),
velocity estimated by finite difference across the first 2-3 frames
(same information the existing architectures already rely on), and a
zero-initialized hidden embedding. This is the only point where a fresh
detection pass is required — after initialization, token identity is
carried by persistent recurrent state, not re-derived each frame.

Known edge case: if two balls start within 2·radius of each other,
initial detection may merge them into one token. Accepted as a rare
failure mode for v1 (balls are spawned independently via
`rng.uniform`), not solved by this design.

### 2. Persistent per-token recurrent state

Each token carries state `(x, y, vx, vy, h)` (`h` = learned embedding)
that updates every frame via a gated recurrent update — the token is a
filter, not a fresh per-frame readout. This is the direct fix for the
collision-noise motivation: velocity is tracked, corrected opportunistically
by observation, not re-inferred from scratch each step.

### 3. Occlusion gate (predictive, distance-based)

Before reading this frame's grid, compute pairwise distances between
tokens' *predicted* (not yet observed) positions. Any pair within
2·radius is flagged "occluding" for this frame — for those tokens,
observation is not trusted; state advances purely from the prediction
branch (below). All other tokens update by blending prediction with a
direct observation readout (position/velocity of their now-isolated
blob).

Chosen over a learned soft gate (network-predicted trust weight): the
occlusion condition is fully determined by known, fixed ball radius and
already-available predicted positions — no reason to make the model
learn a threshold that physics already gives for free. A learned gate
is a fallback if the hard threshold proves brittle in practice (e.g.
near-miss cases), not the starting design.

Expected occlusion duration at default sim params (`radius=0.75`,
`dt=0.15`, max per-axis speed 2.3): a head-on pair closes at up to
~4.6 cells/sec through a 1.5-cell-wide overlap zone, i.e. **~1-3 frames**
of open-loop prediction per collision — short relative to the
multi-step rollout error already characterized in
`docs/debugging/findings-honest-horizon.md` (breakdown from step 6-12).

### 4. Proximity-graph attention (collision dynamics)

A radius-based neighbor graph (edge between any two tokens within a
fixed physical distance, not a fixed k-NN count) feeds a small
graph-attention/transformer body: each token attends only to neighbors
within range, computing a predicted next state from its own state plus
neighbor states/relative geometry. This is where contact-force dynamics
gets learned, from the clean before/after frames bracketing every
occlusion window.

Chosen over global attention: matches the physical prior (contact-only
interactions, per the windowed-attention design's same reasoning) and
is what makes train-small/tile-large valid — a token's update depends
only on local neighborhood, so a bigger tiled grid is just more
repetitions of the same local pattern, not a new distribution.

Chosen over fixed k-NN: collision range is a physical distance
threshold (2·radius), not a fixed count. A radius graph keeps a token's
neighbor set consistent regardless of local density, which matters for
the tileability goal — k-NN's neighbor set composition changes with
density in a way a radius graph's doesn't.

**Tiling requirement**: the neighbor graph must be built from absolute
continuous positions (not per-tile-normalized coordinates), and a
token's edges must be allowed to cross tile boundaries. Building the
graph per-tile in isolation would reintroduce a tile-seam artifact,
the same class of bug as the bicubic tile-boundary issues in
`docs/debugging/findings-long-horizon-lattice.md`.

### 5. Rasterization head

Per-token predicted `(x, y, vx, vy)` is splatted into the output grid
using `bounce.py`'s existing `splat_ball`/`splat_all` formula, verbatim
— not a learned decoder. Two overlapping-but-individually-correct token
predictions automatically produce the correct blended `PROB`/`VX`/`VY`
values, because blending is a property of the renderer the dataset's
ground truth already uses, not something the model needs to learn
separately. This keeps the loss function and eval pipeline unchanged
from the existing grid-based architectures.

## Loss

Primary: per-channel MSE on the rasterized `G_{t+1}` against
`bounce.py`'s ground truth, same as the existing pipeline (comparable
numbers to v6/v7/v8). Secondary/auxiliary (deferred to implementation):
a direct per-token position/velocity loss on frames where ground-truth
per-ball state can be unambiguously recovered (isolated, non-occluding
balls) — may help train the recurrent/gating components faster than
grid-space MSE alone, but is not required for the model to be
evaluable against the existing baselines.

## Data pipeline

Reuses `bounce.py` directly, same as the windowed-attention design:
train at small `n` / low ball count, with curriculum including
pre-clustered spawns to expose the occlusion gate and collision-dynamics
attention to contact events often enough (uniform-random spawns at
small ball count may under-sample collisions).

## Validation plan (staged, cheapest-first)

Per this project's established finding (`findings-mass-conservation-loss.md`):
a synthetic single-step loss/behavior probe on hand-constructed
candidates is not sufficient by itself — it previously missed a
gradient-descent-discovered failure mode (v8's periodic-texture
exploit). This design's validation must include real training end to
end, not just a synthetic check of the occlusion-gate logic.

1. **Component sanity checks** (synthetic, cheap): occlusion-gate
   correctness on hand-constructed overlapping/non-overlapping token
   configurations; neighbor-graph construction correctness across a
   tile boundary.
2. **Training scale**: small `n`, low ball count — held-out single-step
   prediction error, compared against v6 baseline.
3. **Rollout stability**: autoregressive multi-step rollout at training
   scale, using the existing honest-horizon methodology
   (`findings-honest-horizon.md`) to get a directly comparable
   step-count-until-breakdown number against v6/v7.
4. **Collision-specific check**: isolate rollout error specifically in
   frames near a collision event vs. frames away from one — this is the
   metric that actually tests whether this design achieves its stated
   goal (reduced collision-time state noise), not just aggregate MSE.
5. **Unbiased video review**: per standing project practice, run the
   diagnostic-grid + fresh unbiased-subagent review on rollout video
   before drawing conclusions, not aggregate stats alone.
6. **Tile-transfer check**: validate on a tiled grid larger than
   training scale, confirming no seam artifact at tile boundaries.

Compute: Polaris (not TIDE — see project memory).

## Open questions deferred to implementation

- Exact recurrent update mechanism (GRU-style vs. simpler gated linear
  update) for per-token hidden state.
- Depth/width of the graph-attention body.
- Auxiliary per-token loss weighting, if used.
- Whether the occlusion-gate distance threshold needs slack beyond
  exactly `2·radius` (e.g. balls approaching but not yet touching may
  still benefit from suppressing observation trust slightly early).
- Behavior when initial-frame detection merges two overlapping balls at
  sequence start (accepted rare failure mode, not actively solved).
