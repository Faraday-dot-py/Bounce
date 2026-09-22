# Stage-1 vs Stage-2 rollout comparison

Job 2815 (Stage-2 train, 20 epochs, horizon=3, scheduled sampling ramp
0→0.95): loss 13.34 → 10.29.

Autoregressive rollout at `n=50`, 150 balls, seed 4738, 20 self-fed steps,
comparing `stage1.pt` (Stage-1's occupancy-loss-fixed checkpoint) against
`stage2.pt` (this stage's checkpoint — padding mask + union occupancy loss
+ ICNR init + multi-step scheduled-sampling training).

| step | stage1 MSE | stage2 MSE | stage1 occupied cells | stage2 occupied cells |
|-----:|-----------:|-----------:|-----------------------:|-----------------------:|
| 1 | 1.00 | 0.99 | 2450 | 2438 |
| 5 | 17.38 | 2.51 | 2485 | 1551 |
| 10 | 22.68 | 6.31 | 2333 | 1286 |
| 15 | 23.09 | 8.94 | 2297 | 1265 |
| 20 | 21.77 | 7.99 | 2364 | 1262 |

True occupied cells at t=0: ~150 (one per ball, before splat overlap).
Grid is 50×50 = 2500 cells.

**Stage-1**: occupancy balloons to ~2300-2500 cells within 3-5 steps and
stays there — near-total grid saturation, the hallucination artifact
described in the Stage-1 parked issues.

**Stage-2**: occupancy still balloons past the true ~150, but plateaus
around 1260-1320 cells by step ~8 (roughly half the grid, not
near-total saturation) and MSE stays 2-3x lower than Stage-1's at every
step past the first.

**Verdict**: gate met — materially reduced occupied-cell ballooning
within the first 10-20 self-fed steps, not just a lower single-step
held-out loss (per the spec's validation plan). The artifact is
substantially mitigated, not eliminated — a further stage would be new
scope, not a reason to keep iterating here.
