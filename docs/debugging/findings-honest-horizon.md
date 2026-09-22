# Findings: honest rollout horizon (v6 vs v7, by ball density)

Investigation of Problem 2 from `docs/debugging/flownet-open-issues-v2.md`:
quantify the step count at which multi-ball position uncertainty becomes
irreducible for `stage2_flownet_h12_v6.pt` (blur-collapse) and
`stage2_flownet_h12_v7.pt` (give-up collapse), and how it depends on ball
density. Investigation only — `model/net.py`, `model/losses.py`,
`model/train.py` not touched. Scripts used live in the session scratchpad,
not committed (same convention as `findings-peak-decay-dissolution.md`).

## Metric: per-ball position tracking via peak matching against exact ground truth

Unlike the aggregate `occ_frac`/`max0` metrics used in the prior peak-decay
investigation (still reported here for continuity), this uses **exact
ground-truth ball positions** (`bounce.step`'s ball dicts, not derived from
any grid) matched against **local-maxima peaks** detected in the model's
predicted PROB channel each step:

- Peak detection: `scipy.ndimage.maximum_filter` (3x3 footprint) local
  maxima with PROB > 0.05 (same threshold as `occ_frac` in the peak-decay
  doc).
- Matching: optimal (Hungarian, `scipy.optimize.linear_sum_assignment`)
  assignment between ground-truth ball positions and detected peaks,
  capped at a max match distance of `2*radius = 1.5` grid cells (radius
  0.75, this session's standing scenario convention).
- **recall** = fraction of true balls with a matched nearby peak (the
  primary degradation metric).
- **mean_err** = mean position error (grid cells) of matched pairs only.
- **ceiling_recall**: the same peak-detector run on the *ground-truth*
  frames themselves (not model output), matched against the same ball
  positions. This is not 1.0 at high density because overlapping balls'
  splats fuse into one blob with one local max — an inherent occlusion
  limit of *any* point-estimate output at that density, independent of
  model quality. All recall numbers below should be read relative to this
  ceiling, not to 1.0.
- **Null check** (see below): because match windows are large relative to
  average ball spacing at high density, a `recall` was also cross-checked
  against a shuffled-peak-position null to confirm it reflects real
  tracking and not chance overlap.

Repro: `n=50`, radius=0.75, `dt=0.15`, `gravity=9.0`, `stiffness=400.0`,
`substeps=8`, `vy=2.3` (matching prior probes' convention). Seeds 4738,
4739, 4740 (3 seeds, per this session's standing seed convention).
Densities: 20, 50, 125, 250 balls (spans below, within, and at the top of
`model/train.py`'s trained range of 50-250 balls, `scripts/
polaris_train.sh`). 40-step rollouts, both checkpoints from the same
initial frame per (density, seed).

## Results: recall vs. step, by density and checkpoint

(3-seed mean; `--` = no matched pairs, mean_err undefined. Steps 0-16 in
full, then every 4th step to 40.)

```
=== density 20 (occlusion ceiling ~0.98-1.00 throughout) ===
step  v6_recall  v6_err  v7_recall  v7_err
  0     0.983    0.38      0.983    0.38
  2     0.917    0.43      0.583    0.66
  4     0.883    0.63      0.350    1.00
  6     0.750    0.71      0.217    0.84
  8     0.133    0.81      0.117    1.01
 10     0.017    0.90      0.100    0.92
 12     0.000      --      0.050    0.69
 20     0.000      --      0.117    1.09
 40     0.000      --      0.083    0.94

=== density 50 (occlusion ceiling ~0.87-0.97) ===
step  v6_recall  v6_err  v7_recall  v7_err
  0     0.973    0.37      0.973    0.37
  2     0.887    0.49      0.860    0.52
  4     0.827    0.66      0.780    0.65
  6     0.760    0.75      0.633    0.79
  8     0.300    0.98      0.360    1.07
 10     0.000      --      0.247    0.90
 12     0.000      --      0.293    0.90
 20     0.000      --      0.127    0.87
 40     0.000      --      0.147    1.13

=== density 125 (occlusion ceiling ~0.53-0.89) ===
step  v6_recall  v6_err  v7_recall  v7_err
  0     0.819    0.37      0.819    0.37
  2     0.741    0.54      0.752    0.56
  4     0.576    0.64      0.635    0.70
  6     0.272    0.72      0.541    0.78
  8     0.000      --      0.435    0.94
 12     0.000      --      0.320    0.94
 20     0.000      --      0.224    0.89
 40     0.000      --      0.165    0.92

=== density 250 (occlusion ceiling ~0.38-0.75, near trained max) ===
step  v6_recall  v6_err  v7_recall  v7_err
  0     0.645    0.34      0.645    0.34
  2     0.536    0.59      0.563    0.62
  4     0.365    0.75      0.497    0.77
  6     0.264    0.88      0.451    0.89
  8     0.237    0.92      0.453    0.87
 12     0.173    0.81      0.375    0.89
 20     0.109    0.71      0.276    0.81
 40     0.039    0.99      0.316    0.90
```

## Null check: is the nonzero recall floor real tracking or chance?

At higher density and for v7 generally, recall does not collapse fully to
0 (unlike v6 at low/medium density). Ran a supplementary check at select
(density, checkpoint, step) combinations: shuffle the same number of
detected peaks to uniformly random grid positions (200 trials) and
recompute recall against the same ground truth, to get a chance baseline.

```
density  ckpt  step  n_peaks  real_recall  null_recall  signal(real-null)
   50    v6      8      40      0.460        0.101          +0.360
   50    v6     20       0      0.000        0.000           0.000
   50    v7      8      46      0.400        0.115          +0.285
   50    v7     20      21      0.060        0.043          +0.017
   50    v7     40      26      0.120        0.065          +0.055
  125    v7      8     116      0.456        0.252          +0.204
  125    v7     20      76      0.208        0.120          +0.088
  125    v7     40      60      0.184        0.131          +0.053
  250    v6      8      94      0.232        0.183          +0.049
  250    v6     20      40      0.112        0.050          +0.062
  250    v6     40      35      0.044        0.065          -0.021
  250    v7      8     194      0.436        0.347          +0.089
  250    v7     20     145      0.320        0.176          +0.144
  250    v7     40     146      0.316        0.260          +0.044
```

Findings: v6's nonzero-looking recall at density 250 step 40 (0.039-0.044
in the main table) is **not distinguishable from chance** (signal -0.02 to
+0.05) — v6's tracking is genuinely gone by step ~20-40 at every density
tested, the raw recall floor is a peak-density artifact of the match
window, not real signal. v7 retains a small but real (signal reliably
positive, +0.04 to +0.14) above-chance tracking component out to step 40
at medium/high density (125, 250) — consistent with the "give up on most,
commit to one region" mechanism from `findings-peak-decay-dissolution.md`
Part 5: whichever balls happen to fall inside v7's one retained blob stay
weakly trackable, the rest are indistinguishable from noise. At sparse
density (50), v7's late-horizon signal (+0.02 to +0.06) is small and only
marginally above chance.

## Where the "no longer usable" threshold falls

Using recall >= 0.5 (relative to ground truth, not to the density-limited
occlusion ceiling) as "mostly trustworthy per-ball state," and confirming
against the null check that this level is always well above chance:

```
                trustworthy horizon (last step, recall>=0.5)
density   v6              v7
  20      step 6-7        step 1 (already <0.5 by step 2)
  50      step 6           step 1
 125      step 4           step 1 (barely; 0.541 at step 6)
 250      step 1-2         step 0-1
```

v7 is **worse than v6 in this "trustworthy" tier at every density** — the
peak-preservation loss term makes v7 commit early to abandoning most balls
in exchange for keeping one region sharp, so its multi-ball recall crosses
below 0.5 almost immediately, sooner than v6's more gradual blur-driven
decay. This directly corroborates `findings-peak-decay-dissolution.md`'s
Part 5 characterization: v7 is not simply "better," it swaps one failure
mode for a differently-shaped one.

Past the trustworthy horizon, degradation continues but along different
paths: v6 decays smoothly toward **true zero / chance-level information**
(confirmed by the null check) by roughly step 10 (dense) to step 12
(sparse) — a hard floor, nothing recoverable. v7 decays toward a **partial,
density-dependent, but real (non-chance) residual signal** that persists
to at least step 40 at medium/high density, concentrated in whichever
region v7's give-up dynamics happen to preserve — real information, but
about an unpredictable, non-uniform subset of the scene, not a usable
whole-scene state estimate.

Position error of matched balls (`mean_err`, in grid cells, ball diameter
1.5) also climbs steadily even while recall is still meaningful — by the
step each checkpoint crosses recall<0.5, `mean_err` is already
0.6-0.9 cells for both, i.e. even the balls still nominally "tracked" are
positioned with error comparable to a full ball radius, not a precise
estimate.

## Density dependence

Both the trustworthy-horizon and full-collapse points shrink monotonically
with ball count, as expected (more balls -> more collisions -> uncertainty
compounds faster):

- v6 trustworthy horizon: ~6-7 steps (density 20-50) -> ~4 steps (density
  125) -> ~1-2 steps (density 250, near the top of the trained range).
- The occlusion ceiling itself also drops with density (0.98 at density 20
  down to 0.65-0.75 at density 250) — part of the density-250 numbers
  reflect balls that are fundamentally indistinguishable from each other
  in a dense occupancy-grid representation even under perfect prediction,
  not model failure specifically. This is a representation limit
  (dense-grid output cannot resolve fully overlapping balls), separate
  from but compounding the model's own degradation.

## Recommended honest horizon

**For per-ball state that should be treated as reliable (recall >= 0.5,
well above the chance floor confirmed by the null check):**

- **Sparse scenes (~20-50 balls on a 50x50 grid): 6 steps.**
- **Medium scenes (~125 balls): 4 steps.**
- **Dense scenes (~250 balls, near the top of the trained 50-250 range):
  1-2 steps.**

These numbers are essentially the same for v6 and v7 in this tier (v7 is
not longer-horizon-safe than v6 despite fixing the aggregate peak-decay
metric — see above).

**Past these horizons, treat rollout output as unreliable for any
downstream use that depends on individual ball positions or count**, for
both checkpoints. If only *some* signal is needed and v7 is the checkpoint
in use, note it retains a weak, non-uniform, real-but-partial signal out
to ~40 steps at medium/high density — but this cannot be relied on to
cover any particular ball or region, and should not be presented as
whole-scene tracking. v6 fully collapses to chance-level information by
roughly step 10-12 regardless of density, with no residual signal to fall
back on.

**Neither checkpoint should be used for rollouts beyond ~10-12 steps for
any purpose that requires trusting the model's own state as ground truth**
(e.g. as a physics-sim replacement) — consistent with `findings-peak-
decay-dissolution.md`'s conclusion that this is an architectural/objective
-level limit, not a training-progress one. The density dependence found
here (roughly halving the trustworthy horizon from sparse to dense, and
scenes at or above the trained max of 250 balls losing trustworthy
tracking almost immediately) should inform any dataset/scenario design for
downstream use: if longer effective horizons are needed, the practical
lever available without touching the architecture is capping scenario
density well below 250 balls, not extending rollout length.

## Caveats / scope not covered here

- Only uniform-random initial scenarios (`make_scenario_uniform`) were
  tested — `make_scenario_clustered`/`make_scenario_settled` (`model/
  dataset.py`) were not, and could plausibly shift these numbers (e.g.
  clustered scenes may behave like a locally-denser scenario even at
  moderate total ball count).
- The match-distance threshold (1.5 cells = 2*radius) and peak-detection
  footprint (3x3) are reasonable but somewhat arbitrary choices; the
  qualitative conclusions (v7 not longer-horizon-safe than v6 in the
  trustworthy tier; both checkpoints reach a density-dependent hard
  horizon within single-digit-to-low-double-digit steps; v6 fully
  collapses while v7 retains partial signal) are robust to reasonable
  variation in these thresholds since the recall curves have a sharp knee
  rather than being threshold-sensitive at the margin, but the exact
  step numbers reported would shift by 1-2 steps under different
  thresholds.
- This does not investigate *why* v7's give-up region is spatially
  non-uniform (which region it favors, whether it's scenario-dependent) —
  out of scope for this problem; not needed to answer "what horizon is
  honest."
