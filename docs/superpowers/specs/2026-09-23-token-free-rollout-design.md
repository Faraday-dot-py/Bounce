# Token free-rollout design

## Problem

v9/v17 re-detect each token every step in a frame the model rendered
itself (`centroid_near` in v9, cross-attention in v17). A token whose
window loses mass stops reinforcing its own observation and cannot
recover (dropout); tokens near each other swap or merge identities.
v17 is visibly better than v9 by eye, but the last-step dropout count
(4/48 vs 3/48) does not show it.

## Goal

Remove the self-feedback loop. After `init_tokens` on frames 0 and 1,
tokens evolve in state space only; the rasterized frame is output and is
never read back. Token count is fixed by construction, so dropout cannot
occur. Success: lower long-horizon position error and fewer identity
swaps than v9, confirmed by frame review.

## Constraints

- v9 (`track_query=False, free_rollout=False`) and v17 paths stay
  byte-for-byte unchanged and reachable as baselines.
- Training on Polaris; periodic full-state checkpoints.
- Translation invariance in the interior is kept (train-small,
  evaluate-large).

## Design

### Model: `TokenModel(free_rollout=True)`

- `init_tokens` unchanged (frames 0/1 detection, Hungarian velocity
  pairing).
- New `step_free(positions, velocities, hidden)` returns
  `(final_pos, final_vel, new_hidden, next_grid)`; it takes no observed
  frame. `final_pos = positions + velocities*dt + delta_pos`,
  `final_vel = velocities + delta_vel`, same residual convention as v9.
- Incompatible with `track_query`, `territory_masking`,
  `velocity_weight` (raise `ValueError`, as `track_query` already does).

### Dynamics: `TokenFreeDynamics` (`model/token_free.py`)

Same radius-graph attention + GRU + zero-init delta head as
`TokenDynamics`, plus wall-distance node features. The existing
dynamics have no absolute-position input, so they cannot know where the
walls are; with observation correction that was hidden, without it the
model cannot learn wall bounces at all. Node features add
`clamp(x, 0, W), clamp(n-1-x, 0, W), clamp(y, 0, W), clamp(n-1-y, 0, W)`
with `W = wall_range` (default 3.0): local geometry, zero in the
interior, so interior translation invariance holds. Gravity is a
constant and is learned through the delta-head bias. `n` is passed to
the dynamics at construction.

### Training (`token_train.py --free-rollout`)

- `token_rollout_loss` free branch: init from frames 0/1, unroll
  `horizon-1` steps calling `step_free`, no self-feed coin flip and no
  observed frame.
- Loss: `token_state_loss` per step against ground truth (matched once
  after init via `match_tokens_to_state`), `boundary_loss`, and optional
  `token_grid_loss` on the rasterized output (weight default 0.1).
- Horizon curriculum: the trainer accepts `--horizon-ramp` epochs to
  grow the unroll from 4 steps to the dataset horizon (h24 dataset,
  generated with `scripts/generate_token_dataset.py`).
- Full-state checkpoint (model + optimizer + epoch) every epoch.

### Optional re-anchor (not in v1)

Every K steps, Hungarian-match tokens to detected blobs. Built only if
free-rollout error curves show drift that hurts.

## Evaluation (`scripts/eval_free_rollout.py`)

Same 48 seeds (start 4738), scenario generator as
`diagnose_token_dropout.py`, for v9 and the new model:

- Mean position error vs step at 20, 100, 300 (ground truth from
  `bounce.py`), matched once at frame 1.
- Identity swaps: count of tokens whose Hungarian match to ground truth
  changes between frame 1 and the final frame.
- Ball-count/peak sanity: fraction of tokens with own PROB peak < 0.05
  at the final step (reported for v9 only as the legacy metric; for the
  free model it is reported too, as a rendering check).
- OOD: 50x50 grid, 100 balls, 300 steps.
- Frame review: diagnostic grid + unbiased subagent (per
  `docs/debugging/frame-artifact-review-prompt.md`).

v9 in the comparison runs with its normal observation loop, i.e. its
real inference mode, and additionally with `observation_weight=0` as a
reference for how much of v9's accuracy comes from re-observation.

## Testing

Unit tests: `TokenFreeDynamics` shape, zero-init identity, interior
translation invariance (shift all tokens by a constant well inside the
grid: identical deltas), wall features nonzero only near walls;
`step_free` fixes token count; incompatibility errors; free training
loss finite and backprops at 0, 1 and N tokens; existing tests stay
green.

## Risk

Open-loop physics may diverge at 100 balls / 300 steps (collision
chaos). Mitigation is the optional re-anchor above; the eval reports
error growth so the need is measured, not assumed.
