# Stage-1 vs Stage-2 rollout comparison

Job 2816 (Stage-2 train, 20 epochs, horizon=3, scheduled sampling ramp
0→0.95): loss 13.12 → 10.24. (Supersedes job 2815 — that run predates a
final-review fix to `model/block.py`'s mask combination, which had made
the padding fix inert in the 3 shifted `SwinBlock`s; see
`.superpowers/sdd/2026-09-21-stage2-fixes/progress.md` for the ruling.
Dataset cache was reused, so this run only needed the ~2 min training
step, not a full regeneration.)

Autoregressive rollout at `n=50`, 150 balls, seed 4738, 20 self-fed steps,
comparing `stage1.pt` (Stage-1's occupancy-loss-fixed checkpoint) against
`stage2.pt` (this stage's checkpoint — padding mask + union occupancy loss
+ ICNR init (weight and bias) + multi-step scheduled-sampling training).

| step | stage1 MSE | stage2 MSE | stage1 occupied cells | stage2 occupied cells |
|-----:|-----------:|-----------:|-----------------------:|-----------------------:|
| 1 | 1.00 | 0.99 | 2484 | 2440 |
| 5 | 16.31 | 2.49 | 2496 | 2193 |
| 10 | 23.10 | 6.42 | 2433 | 980 |
| 15 | 21.47 | 9.02 | 2392 | 768 |
| 20 | 19.70 | 8.00 | 2450 | 664 |

True occupied cells at t=0: ~150 (one per ball, before splat overlap).
Grid is 50×50 = 2500 cells.

**Stage-1**: occupancy balloons to ~2400-2500 cells within a few steps and
stays there — near-total grid saturation, the hallucination artifact
described in the Stage-1 parked issues.

**Stage-2**: occupancy rises initially but then *declines* steadily from
step ~6 onward (2193 → 980 → 768 → 664 by step 20), continuing to trend
toward the true ~150 rather than plateauing at a fixed fraction of the
grid. MSE stays 2-3x lower than Stage-1's at every step past the first.

**Verdict**: gate met — materially reduced occupied-cell ballooning
within the first 10-20 self-fed steps, not just a lower single-step
held-out loss (per the spec's validation plan), and the fully-applied
padding mask (post final-review fix) gives a clearly better trajectory
than the partially-applied version did. The artifact is substantially
mitigated, not eliminated — a further stage would be new scope, not a
reason to keep iterating here.
