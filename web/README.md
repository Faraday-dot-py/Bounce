# Bounce token model in the browser

Static page: the conservative-contact token model (`results/cons_pure.pt`, 8.7k floats: distance-only pair force, wall force, learned gravity, 8 velocity-Verlet substeps) runs a live 100x100 bouncing-ball sim in plain JavaScript. The page renders the arena as a field of spatial tokens (one column of pos/vel channel cubes per ball) in Three.js (loaded from jsDelivr), and unfolds the selected ball's computation beside it: pair-force MLP activations per neighbour, wall-force MLPs, learned gravity, the 8 substeps and the output token.

Open `index.html` through any static server (`python3 -m http.server` in this directory). Published at the Pages root, which redirects here.

## Files

- `js/model.js`: port of `TokenFreeDynamics._conservative` + `TokenModel.step_free` and the `contain_state` guard from `scripts/realtime_sim.py`. `step(pos, vel, hidden, count, traceIdx)` steps in place (hidden passes through unchanged); `traceIdx >= 0` also returns a trace for that ball.
- `js/sim.js`: ball state, spawning (same rules as `realtime_sim.py`), ticking.
- `js/physics.js`: port of the `bounce.py` ground truth, for the ghost overlay.
- `js/scene.js`: Three.js scene: arena, token cubes, neighbour links, contact glows, ground-truth ghosts, camera presets, picking.
- `js/arch.js`: 3D architecture diagram of one trace (cell layout, labels, flow animation) and the diverging colour map.
- `js/plots.js`: pair/wall force curves sampled from the MLPs, energy plot.
- `js/app.js`: UI wiring, controls, keyboard shortcuts (space, `.`, r, b, v, g, e, t, 1-4, +/-).
- `weights.bin` / `weights.json`: fp32 weights + manifest/config (13100 floats).

## Re-export weights

    PYTHONPATH=. python3 scripts/export_web_weights.py --checkpoint results/cons_pure.pt

Config (radius 0.75, dt 0.15, force_scale 100, neighbour radius 4.0, arena 100) is baked into the export script; substeps come from the checkpoint flags.

## Verify against PyTorch

    PYTHONPATH=. python3 scripts/export_web_testvectors.py
    node web/tests/verify.mjs
    PYTHONPATH=. python3 scripts/dump_web_physics_ref.py
    node web/tests/physics.test.mjs web/tests/physics_ref.json

`verify.mjs` gates 1-step max abs diff (< 1e-4) against a float64 PyTorch step from the same float32 states (~2e-6), and reports diffs against the float32 PyTorch rollout (up to ~4e-3 in dense contact, where torch's own float32 rounding is amplified by the stiff force) and free-run diffs at step 20 (chaotic in contact-heavy scenes) and 60. It also checks the per-substep trace of one ball.

## Trace

`net.step(..., traceIdx)` returns (arrays are Float64Array unless noted; vectors are [x, y], x is the gravity axis):

- `index`, `pos`, `vel`: ball state at the start of the step. `hidden`: Float32Array(32) passthrough (unused, zero).
- `wallDist`[4]: distances to walls in order x=0, x=n-1, y=0, y=n-1. `wall`[4]: `{pen, out, force, h1[64], h2[64]}` per wall (pen = relu(r - d)/r, `out` = raw MLP output, `force` = pen * out * 100, h1/h2 = tanh activations). `wallForce`: net wall acceleration [f0 - f1, f2 - f3]. `gravity`: learned acceleration.
- `edges`: one per other ball within neighbor_radius (4.0): `{src (index), dist, pen, unit (from src to this ball), contact (dist <= 2r), out, mag, force, h1, h2}`. `out`, `mag`, `force` (vector on this ball), `h1`, `h2` are set only when `contact`; otherwise 0/null. `app.js` adds `srcId`.
- `subPos`, `subVel`, `subAcc`, `subAccWall`, `subAccPair`: (substeps + 1) x 2 flat, row 0 = start state and acceleration there, row s = after substep s. `subAcc` = wall + pair + gravity.
- `newPos`, `newVel`, `dp`, `dv` (dp relative to pos + vel * dt, as in PyTorch).

`net.edgeList(pos, count, radius)` returns a flat [i, j, ...] pair list.
