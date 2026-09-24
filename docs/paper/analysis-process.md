# Bounce: analysis process (for the paper)

Record of how results are produced and checked, so the method section can be
written without reconstructing it. Living document; append as the process
evolves. Numbers live in `docs/debugging/experiment-log.md` and commits.

## Model under study
Token-per-ball free-rollout model (soup B = weight average of the v30-v33
fine-tune lineage). Judged at rollout steps 3-20 (chaos limit beyond ~20).
Gravity = 9 on. Training on Polaris, 50 epochs, seed 4738.

## Evaluation protocol
- Free rollout scored by `scripts/eval_free_rollout.py` against ground-truth sim.
- Report err@5/10/20 (pos error). Headline soup B ~0.15/0.45/2.5 vs v18 1.15/1.95/4.9.
- Select checkpoints on held-out seed sets (4738, 9000, 12000; 48 seeds each);
  fine-tunes differ ~10-15% by noise, so lineage soups are used, not single picks.
- Trivial baselines (`scripts/eval_trivial_baselines.py`) always reported.
- Break errors down by channel / mass / saturation, not aggregate MSE only.
- Synthetic loss probes must be validated by a real retrain before being trusted.

## Visual review (unbiased)
After every run producing rollouts: render (`scripts/render_token_diagnostic_grid.py`,
videos in `videos/`), then a fresh subagent with no hypothesis primed describes the
output using `docs/debugging/frame-artifact-review-prompt.md`. Aggregate stats alone
have repeatedly hidden per-frame artifacts.

## Root-cause method
Ablate one component at a time (init/readout, dynamics head, hidden state, pair head),
teacher-forced vs free-rollout, per-class (wall / pair / free-flight) accounting.
Prefer architectural explanations over hparam sweeps.
Example, energy audit 2026-09-24 (soup B, 3 seed sets x 48, 4 balls, 100 steps):
truth E flat (-83 -> -85), model -83 -> -116. Cause: MSE-trained update with no
conservation structure under-transfers contact impulses (dv slope 0.66-0.81 pair,
0.88-0.95 wall; vy channel worst), hidden norm drifts OOD after step ~20. Init/readout
negligible; free-flight gravity exact. Proposed fix: conservative antisymmetric
impulse + fixed symplectic integrator, GRU residual only. Not yet implemented.

## Scaling benchmark (time complexity)
- Sim step made O(N): cell-list graph (`build_radius_graph_cells`), `render=False`
  skips rasterization, optional `local_softmax` (per-destination max).
- `scripts/bench_scaling.py`, `scripts/polaris_bench_scaling.sh`, `scripts/tiled_sim.py`,
  `scripts/plot_bench_scaling.py`. Results `results/bench_scaling_cuda.json`,
  `results/bench_scaling_big.json`, plot `results/bench_scaling.png`.
- Setup: Polaris H200 GPU, 300 ticks, density 0.01, grid 10000 to 1M balls then
  10*sqrt(N) (density held). Median ms/tick reported.
- Results: 1M = 4.39 s (14.2 ms/tick, 4.1 GB); 10M 40 s; 100M 384 s;
  500M 6.33 s/tick (1898.9 s total, 103 GB peak, ~70.7 GB resident state);
  slope 0.96-0.99 => O(N); flat ~0.33 s below 100k (launch overhead).
  ~13-14 ns/ball/tick constant from 1M to 500M.
- Beyond 10M the run is tiled (strip tiling with halos): exact per-step vs global
  (~3e-5) but uses local softmax max; full rollouts diverge chaotically in dense
  piles (expected, not a benchmark artifact).
- 1B run (job 2917, ~12.4-12.9 s/tick so far): hidden state fp16 because fp32 = 144 GB
  > 141 GB GPU. CAVEAT: fp16 hidden differs from the fp32 used in all smaller runs;
  report it. Fill final 1B total/ms per tick/peak mem here when job finishes.
- Comparison to physics sims: same O(N) class; ~13 ns/ball/tick is CPU-sim range,
  est. 3-10x slower than tuned analytic GPU codes (recollection, NOT measured).
  A fair baseline needs an analytic GPU sim at equal N on same hardware (not run).
- The interactive demo cap (`--max-balls 300`) is a demo setting, not a model limit.

## Generality
Sim-specific assumptions audit: `docs/paper/sim-specific-assumptions.md`. Goal: emulate any similar particle sim (e.g. N-body gravity) with no sim-specific traits.

## Reproducibility notes
Seed 4738 default; commit scripts+results with key metric after each experiment;
checkpoints saved for long runs; Polaris logs at ~/<name>-<id>.log, 1 GPU job at a time.
