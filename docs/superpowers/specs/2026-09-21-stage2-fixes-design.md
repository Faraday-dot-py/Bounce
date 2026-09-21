# Stage-2: Windowed-Attention Model Fixes — Design

Date: 2026-09-21

## Goal

Fix the three issues explicitly parked at the end of Stage-1 (see
[[project-windowed-attention-stage1-done]] / `docs/superpowers/specs/2026-09-21-windowed-attention-model-design.md`):

1. Unmasked padding tokens in windowed attention.
2. Target-only occupancy mask in the training loss (stale mass not penalized).
3. Autoregressive checkerboard/hallucination artifact under rollout.

No new capability or scale target — this stage is corrective, not additive.
Validation gate is a Polaris retrain compared against `stage1.pt`'s rollout
divergence.

## 1. Padding mask

**Problem.** `model/windows.py::pad_to_multiple` pads `H`/`W` up to a
multiple of `window_size`. `SwinBlock.forward` (`model/block.py`) only
builds an attention mask when `shift_size > 0` (via `compute_shift_mask`).
Padding tokens are otherwise fully unmasked — they participate in attention
like real tokens, in both the unshifted and shifted case. Padding fraction
is `n`-dependent (larger at small `n`), so boundary-window behavior differs
between training scale and inference scale.

**Fix.** Add `compute_validity_mask(Hp, Wp, orig_H, orig_W, window_size,
device)` to `model/windows.py`: build a `(1, Hp, Wp, 1)` tensor that is `1`
for cells inside `[0, orig_H) x [0, orig_W)` and `0` elsewhere, partition it
into windows the same way `compute_shift_mask` does, and produce an additive
`-100`/`0` bias mask from pairwise validity (a window-pair gets `-100` if
either token is padding).

`SwinBlock.forward` always computes this validity mask when `Hp != orig_H or
Wp != orig_W` (regardless of `shift_size`), and combines it with the shift
mask (when present) via elementwise `torch.maximum` (more negative wins —
either condition is enough to suppress attention). When there's no padding
and no shift, `mask` stays `None` as today.

## 2. Target-only occupancy mask

**Problem.** `model/losses.py::occupancy_weighted_mse` builds its spatial
weight from `target[:, 0:1]` (frame `t+1`) occupancy only. A cell occupied
at `t` but vacated by `t+1` lands in the `bg_weight`-discounted background,
so the model is barely penalized for failing to predict that the mass
should clear. Suspected contributor to the hallucination artifact (fix #3).

**Fix.** Change signature to
`occupancy_weighted_mse(pred, target, source, weights, bg_weight=0.05)`.
Spatial weight mask becomes the union:
`(target[:, 0:1] > 1e-6) | (source[:, 0:1] > 1e-6)`. All call sites
(training loop, tests) pass the step's source frame alongside target.

## 3a. Checkerboard artifact — ICNR init

**Problem.** `model/patchify.py::Unpatchify` projects each patch's embedding
through one shared `Linear(embed_dim, patch_size*patch_size*out_channels)`.
Each of the `patch_size*patch_size` sub-pixel output positions within a
channel has independent, randomly-initialized weights — the standard setup
that produces sub-pixel/pixel-shuffle checkerboard artifacts (Aitken et al.
2017, "Checkerboard artifact free sub-pixel convolution").

**Fix.** Add `icnr_init(weight, patch_size, out_channels)` to
`model/patchify.py`: reinitialize `Unpatchify.proj.weight` so that, for each
output channel, all `patch_size*patch_size` sub-pixel positions start with
identical weight rows (copy one initialized sub-filter across the group,
per Aitken et al.). Applied once in `Unpatchify.__init__` after the
`nn.Linear` is constructed. No forward-pass, shape, or parameter-count
change — training can still break the symmetry as needed.

## 3b. Checkerboard artifact — multi-step rollout training

**Problem.** Training only ever supervises real→real single steps. Any
per-step artifact (including whatever ICNR init doesn't fully eliminate)
compounds unchecked across an autoregressive rollout, since nothing in
training penalizes multi-step drift.

**Fix — data.** Replace `BouncePairDataset` with `BounceSequenceDataset` in
`model/dataset.py`:
- `generate_pair(...)` becomes `generate_sequence(balls, n, dt, gravity,
  radius, stiffness, substeps, horizon)`, returning a list of `horizon + 1`
  grid frames (`[g_0, g_1, ..., g_horizon]`) via repeated `bounce.step`
  calls, splatting a fresh grid at each step.
- Scenario generation (`uniform`/`clustered`/`settled`) and sample-index
  bookkeeping are unchanged.
- `__getitem__` returns a single `(horizon+1, C, H, W)` tensor per sample
  instead of a `(g_t, g_t1)` pair.
- Cache format changes to store the full sequence array; old pair-format
  `.npz` caches are incompatible (acceptable — caches/checkpoints are
  gitignored and regenerated on Polaris, per Stage-1 practice). Cache
  validation config gains `horizon`.

**Fix — training loop (`model/train.py`).** New `--horizon` (default `3`)
and `--sampling-ramp-epochs` (default: `args.epochs`) CLI args. Per batch:
- `frame = sequence[:, 0]` (ground truth `g_0`).
- For `k` in `1..horizon`: `pred = model(frame)`; compute
  `occupancy_weighted_mse(pred, sequence[:, k], frame, weights, bg_weight)`
  and accumulate; then set `frame` for the next step to `sequence[:, k]`
  (ground truth) with probability `1 - p`, or `pred.detach()` (self-fed)
  with probability `p`, sampled once per batch. `p` ramps linearly from `0`
  at epoch `0` to `1` at `sampling_ramp_epochs` (clamped to `1` after), i.e.
  `p = min(1.0, epoch / sampling_ramp_epochs)`.
- Batch loss is the mean of the `horizon` per-step losses. Backward/step
  happens once per batch on the summed/mean loss (not per-step), matching
  the existing single optimizer-step-per-batch structure.
- `pred.detach()` on self-fed steps is deliberate: gradients still flow
  through the per-step loss into the model that produced each prediction,
  but not back through the chain of prior self-fed steps — keeps memory
  bounded and matches standard scheduled-sampling practice (only the
  current step's prediction is differentiated).

## Testing

Per Stage-1's test-per-module pattern:
- `compute_validity_mask`: shape, all-valid case (no padding) is all-zero
  bias, partial-padding case marks the correct window pairs.
- `occupancy_weighted_mse` with `source`: synthetic vacate case (occupied
  at source, empty at target) gets full weight, not `bg_weight`.
- `icnr_init`: post-init weight rows within a sub-pixel group are equal;
  post-one-optimizer-step they're allowed to diverge (confirms it's an init,
  not a constraint).
- `BounceSequenceDataset`: sequence length/shape, cache save/load roundtrip
  including `horizon` in the validated config.
- Scheduled-sampling schedule: `p` values at epoch `0`, mid-ramp, and
  post-ramp; rollout loop wiring (source/target pairing per step, self-feed
  vs teacher-force branch taken per the sampled probability).

## Validation plan

1. Unit tests (above) green.
2. Polaris retrain at Stage-1 settings (`n=50`, 50–250 balls, `horizon=3`)
   reusing/regenerating the dataset cache.
3. Compare autoregressive rollout divergence against `stage1.pt` — gate is
   materially reduced occupied-cell ballooning within the first 10-20
   self-fed steps (the failure mode diagnosed in Stage-1), not just a
   lower single-step held-out loss.
4. If rollout divergence is still high, that's new information for a
   further stage, not a reason to keep iterating past this spec's scope.

## Open questions deferred to implementation

- Exact `sampling-ramp-epochs` default may need tuning against the
  existing 20-epoch training budget (a linear ramp over 20 epochs may
  under- or over-shoot).
- Whether `horizon=3`'s memory/compute cost forces a smaller batch size on
  Polaris — adjust `--batch-size` empirically if needed, not a design change.
