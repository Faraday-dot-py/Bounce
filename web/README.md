# Bounce token model in the browser

Static page: the soup B token-per-ball model (`checkpoints/token_model_soup_b.pt`, ~13k parameters) runs a live 100x100 bouncing-ball sim in plain JavaScript, next to a 3D cube view of the selected ball's activations (WebGL2, no libraries).

Open `index.html` through any static server (`python3 -m http.server` in this directory). Published at the Pages root, which redirects here.

## Files

- `js/model.js`: port of `TokenFreeDynamics._core` + `TokenModel.step_free` (cell graph, global-max softmax, wall head, pair head) and the `contain_state` guard from `scripts/realtime_sim.py`. `step(..., traceIdx)` also returns every intermediate for one ball.
- `js/sim.js`: ball state, spawning (same rules as `realtime_sim.py`), ticking.
- `js/physics.js`: port of the `bounce.py` ground truth, for the side-by-side view.
- `js/viz3d.js`: instanced-cube renderer; layout of tensors lives in `buildLayout`.
- `js/app.js`: UI.
- `weights.bin` / `weights.json`: fp32 weights + manifest/config (13100 floats).

## Re-export weights

    PYTHONPATH=. python3 scripts/export_web_weights.py --checkpoint checkpoints/token_model_soup_b.pt

The flags baked into the config (`wall_lookahead`, `wall_head`, `pair_impulse`, hidden 32, neighbour radius 4.0, arena 100) must match the checkpoint.

## Verify against PyTorch

    PYTHONPATH=. python3 scripts/dump_web_testvectors.py
    node web/tests/verify.mjs
    PYTHONPATH=. python3 scripts/dump_web_physics_ref.py
    node web/tests/physics.test.mjs web/tests/physics_ref.json

`verify.mjs` reports 1-step and free-run max abs differences against PyTorch rollouts (fp32 rounding at speed 30 gives ~4e-5 per step; the dynamics amplify it over tens of steps).

## Reading the 3D view

Rows are tensors, top to bottom in data-flow order: input node state, q, per-edge K/V with score and softmax, attention output, GRU gates and new hidden state, delta head, wall head, pair-impulse head, final dp/dv, then the weight matrices below. Colour is a blue/orange diverging map, symmetric about zero (about 0.5 for the gates); height is |value|. Values are shown at the scale of the running maximum of each group. Hover a cell for its value.
