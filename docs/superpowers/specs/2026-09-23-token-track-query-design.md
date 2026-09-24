# Token-Per-Ball Track-Query Attention — Design

Date: 2026-09-23

## Goal

Fix the token-per-ball model's remaining failure mode -- token identity
loss ("dropout") under proximity/occlusion -- with an architecture that
resolves identity implicitly through learned attention, instead of an
explicit hard occlusion gate or spatial mask.

## Background

The token-per-ball model (`docs/superpowers/specs/2026-09-22-token-per-ball-model-design.md`)
tracks each ball as a persistent `(x, y, vx, vy, h)` state, corrected
each step by a windowed observation read (`centroid_near`) that is
either fully trusted or fully skipped based on a hard, physics-derived
occlusion gate (`occluding_mask`, `TokenModel._gate_radius`). The best
checkpoint to date (v9, unmasked) still drops 3/48 seeds -- a token's
own rasterized mass falls below detection threshold during self-feed
rollout, almost always following a close-proximity/occlusion event.

Five attempts to fix this by tuning the existing structure all made it
worse (`docs/debugging/experiment-log.md`):
- v10-v12 (velocity-weighting variants): 10-30/48 dropout.
- v13 (`window_collapse_loss`, a give-up penalty): 34/48, plus a
  progressive color-bleed artifact.
- v14 (hard territory-masking, no fallback): 66/48 -- masking blocked a
  token's only recovery path once its own mass was gone.
- v15 (opt-in masking with unmasked fallback): 16/48 -- still worse
  than v9's unmasked baseline.

Research into two external literatures (TimesFM-style time-series
forecasting, and multi-object-tracking/re-identification) concluded the
former doesn't address identity assignment at all (it assumes series
identity is given and fixed), while MOT's track-query transformers
(MOTR, TrackFormer) are a structurally different idea from every prior
attempt here: identity resolution emerges from learned self- and
cross-attention weights, never from a hard spatial mask or gate. This
design adapts that idea.

## Constraints (must hold, same three tenets as every prior design in
this project)

1. **Arbitrary n** (sequence length / rollout horizon) -- no hardcoded
   horizon; must work as a per-step recurrent update regardless of how
   many steps have run.
2. **Arbitrary n_ball** -- must generalize to a token count not fixed at
   train time; no fixed-size state vector.
3. **No baked-in physics** -- must not encode gravity/collision
   equations; must be learnable purely from data for systems whose
   internal mechanism is unknown.

Additional project-specific constraint carried over from the original
token-model design: **train-small/tile-large**. A token's update must
depend only on local geometry, not absolute grid position, so training
at small `n` generalizes to a larger tiled grid without a seam
artifact.

## Scope

**Opt-in, not a replacement.** v9's exact code path (`TokenDynamics`,
`occluding_mask`, `centroid_near`, `observation_weight`/`velocity_weight`
blending) stays completely untouched and remains the default
(`track_query=False`). This is deliberate: every prior structural change
to the observation branch (v13-v15) caused the worst regressions in the
whole investigation, and v9 must stay reachable as the reference
baseline regardless of how this experiment turns out. Selected via
`TokenModel(..., track_query=True)` / `--track-query` in
`token_train.py`.

No new loss terms. `token_state_loss`, `token_grid_loss`,
`boundary_loss` apply identically to both paths, unchanged. This
isolates the architecture as the only variable under test --
`window_collapse_loss` stays available (default weight 0) but isn't
wired specially into this path. Every prior variant that changed a loss
term *and* a structural mechanism in the same experiment left the
result ambiguous when it came back mixed; this experiment changes
exactly one thing.

## Architecture

### New module: `model/token_track_query.py` -- `TrackQueryDynamics`

Drop-in alternative to `TokenDynamics` with the same call signature
(`forward(positions, velocities, hidden, observed_frame) -> delta_pos,
delta_vel, new_hidden, obs_pos`) plus the current observed frame, since
this path folds the observation read into the dynamics call instead of
doing it separately in `TokenModel.step`.

Two attention stages per step, in this order:

**1. Self-attention among tokens** (identity-shaping, done first).
Reuses `TokenDynamics`'s existing radius-graph GAT shape verbatim:
query/key/value over node state `[velocity, hidden]`, edge feature =
relative position offset (`build_radius_graph`, same
`neighbor_radius`), self-loop included so single-neighbor tokens still
get gradient. Output: `attn_out_self`, one vector per token. Rationale
for going first (not scene-first, unlike v9's structure): the
hypothesis this design tests is that a token should reason about where
its neighbors are *before* trying to read potentially-merged scene
mass, so it can use relative-position information to disambiguate "this
mass is mine vs. my neighbor's" -- the thing territory-masking tried to
hard-code and failed to generalize.

**2. Cross-attention to scene** (replaces `occluding_mask` +
`centroid_near` entirely, only in this path -- no hard gate, no
territory mask). For token `i`, gather the same bounded local window
`centroid_near` already searches: base half-width `ceil(radius +
margin)`, widening by one cell per retry up to `max_expansions` if
empty (same hyperparameter, reused for direct comparability -- not a
new tunable). Per-cell feature = `[PROB, VX, VY, relative dx, relative
dy]` (relative to the token's own current position -- no absolute
position enters, preserving translation invariance/tileability). Query
= linear projection of the token's post-self-attention hidden state.
Softmax attention over the window's cells produces:
  - a weighted feature readout, fed into the update below;
  - a byproduct weighted centroid (`obs_pos`) -- kept only for
    diagnostic logging and to preserve `TokenModel.step`'s existing
    5-tuple return interface (`prev_obs_pos` bookkeeping in
    `token_train.py` needs no changes), not used for any hard blend.

If every expansion's window is empty (token has drifted fully off any
mass, including off-grid), cross-attention falls back to an
all-zero/uniform readout -- there is no masked-then-unmasked-retry
logic to replicate here, because there is no mask to begin with; the
model either finds mass in its unmasked local window or it doesn't.

**3. Update.** `attn_out_self` and the cross-attention feature readout
concatenate into one GRU update -> `new_hidden`. `delta_head(new_hidden)`
produces `(delta_pos, delta_vel)`, zero-initialized (same "coast at
current velocity" residual convention as `TokenDynamics`).
`final_pos = positions + velocities * dt + delta_pos` -- computed from
the *original* input positions/velocities, not an explicitly
blended/corrected value. Correction is implicit, carried entirely
inside the hidden state through the two attention stages, which is the
actual mechanism under test (soft/implicit vs. hard/explicit trust).

### `TokenModel.step` change

One conditional branch: if `self.track_query`, call
`TrackQueryDynamics` directly on `(positions, velocities, hidden,
observed_frame)` and skip the `occluding_mask`/`centroid_near`/
blending block entirely. Otherwise, byte-for-byte unchanged. Return
signature stays the same 5-tuple in both branches.

## Diagnostics

New script `scripts/visualize_track_query_attention.py`: for a rollout,
dumps per-step, per-token (a) the self-attention weight row (which
neighbors it attended to and how strongly) and (b) the cross-attention
weight heatmap over its local window, aligned against
`diagnose_token_dropout.py`'s existing per-step trace output (window
total, occlusion state, ground-truth error). Built alongside the
architecture, not added after a result comes back ambiguous -- per this
session's interpretability discussion, this is the known cost of moving
from an explicit hard gate/match (directly inspectable) to soft
attention (diffused across layers/heads), and the mitigation is
treating attention-weight logging as a first-class diagnostic output
from day one rather than eyeballing a bad result after the fact.

## Testing

New `tests/test_token_track_query.py`:
- Shape/gradient tests for both attention stages (output shapes match
  input token count; gradients reach all trainable parameters).
- **Permutation test**: reordering the input token set does not change
  any individual token's output (up to the corresponding reorder) --
  confirms arbitrary-n_ball (tenet 2), since nothing in the module may
  depend on a token's index or position within the batch.
- **Translation-invariance test**: mirrors `TokenDynamics`'s existing
  one -- translating every token's position by the same offset does not
  change any output. Confirms no absolute-position leakage, since
  cross-attention only ever sees relative offsets and cell values, and
  self-attention only sees relative position edges, never absolute
  position (tileability requirement).
- Empty-token-set guard (0 tokens in, 0 tokens out, no NaN) matching
  the existing convention in `token_losses.py`/`token_model.py`.

## Validation plan

Same staged, cheapest-first order this project uses for every
architecture change (`docs/superpowers/specs/2026-09-22-token-per-ball-model-design.md`'s
own validation plan):

1. Unit tests above (synthetic, cheap).
2. Train at existing default scale (`n=20`, 2-6 balls, gravity=9,
   50 epochs -- same recipe as v9) via `--track-query`, on Polaris.
3. `diagnose_token_dropout.py` (48 seeds) -- the direct, established
   metric this whole investigation has used throughout. Primary success
   criterion: dropout count below v9's 3/48. Any result at or above
   3/48 means this architecture did not beat the existing baseline,
   regardless of other qualitative improvements.
4. Diagnostic grid + rollout video + **unbiased fresh-subagent review**
   (standing project practice) before drawing conclusions -- aggregate
   dropout count alone has repeatedly missed real artifacts in this
   project's history (v13's color-bleed was caught this way).
5. Attention-weight visualization (`visualize_track_query_attention.py`)
   on any seed that still drops out, to see whether the failure mode
   changed in kind (e.g., attention diffusing across both tokens during
   occlusion) even if the count didn't improve -- this is the
   diagnostic payoff for accepting the interpretability cost.

## Open questions deferred to implementation

- Number of self-/cross-attention layers (single layer per stage,
  matching `TokenDynamics`'s current depth, is the starting point --
  not stacking multiple decoder blocks MOTR-style unless a single layer
  proves insufficient).
- Whether `hidden_dim` needs to grow to give the two attention stages
  enough capacity, given `TrackQueryDynamics` does strictly more work
  per step than `TokenDynamics`.
- Exact behavior at `max_expansions` exhaustion (all-zero vs. uniform
  fallback readout) -- pick whichever tests cleaner in the unit tests,
  no behavioral requirement distinguishes them a priori.
