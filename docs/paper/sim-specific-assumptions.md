# Audit: sim-specific assumptions in the token model (2026-09-24)

Scope: model/token_*.py, bounce.py. Goal: what must become configurable/learned
to emulate another particle sim (e.g. N-body gravity). Static read, not run.

## A. Bounce-physics baked into the dynamics (blocks other sims)
- `wall_features`, `wall_contact_features`, `wall_head` (token_free.py:7-27,80-89,154): four axis-aligned walls of a square box at 0 and n-1. No walls or a different domain (periodic, open) breaks it. Off-switch exists (wall_lookahead/wall_head flags), but base `wall_features` is always on.
- `pair_invariants` / `pair_head` (token_free.py:30-45,157): contact features use `2*radius` as contact distance and penetration = relu(2r - d). Assumes finite-radius hard/penalty contact. Antisymmetric normal+tangent form assumes reciprocal, central-plus-tangential forces. Opt-in flag.
- `mirror_sym` (token_free.py:99-108): mirrors y about the box (n-1-y) and flips vy; assumes y-mirror symmetry. Gravity breaks it unless handled; opt-in.
- Gravity is NOT hard-coded in the network: it is learned through the delta head and hidden state (the y-bias diagnosis found it in delta_head.bias). Good for generality, but also why it is fragile off-distribution.
- Fixed integrator form final_pos = pos + vel*dt + delta_pos, vel += delta_vel (token_model.py:step_free): generic for 2nd-order dynamics; the network only supplies corrections.

## B. Interaction locality (blocks long-range sims)
- Radius graph / cell list with `neighbor_radius` (default 4.0; token_graph.py): only neighbours within a fixed radius interact. Correct for contact; wrong for gravity/Coulomb. The O(N) scaling result depends on this.
- Softmax-normalized attention (weights sum to 1 per token): cannot represent forces that scale with neighbour count or 1/r^2; the pair_head (summed, not normalized) is the workaround.
- `local_softmax` vs scene-wide max: global max makes updates depend on scene size (needed off for tiling).

## C. Perception / init pipeline tied to the grid rendering (blocks non-grid inputs)
- Observation is a 3-channel grid (PROB, VX, VY) rendered by bounce.py splat: linear-falloff disk of `radius`, PROB=1-exp(-sum w), VX/VY weighted mean (token_rasterize.py). Detection (`find_token_positions`, `detect_balls`, `refine_positions`, `read_token_velocities`, `centroid_near`) all assume this encoding and disc radius 0.75.
- `detect_balls` max_tokens=12, `greedy_fit` max_balls=8 (token_split.py): init from images limited to <=12 balls. Free rollout itself has no cap.
- `max_init_speed` default 20 cells/s tied to "free-fall across 20-cell grid at gravity 9" (token_model.py:229 comment).
- Hard-coded defaults radius=0.75, dt=0.15, detect_margin=1.0, `_gate_radius`=radius+margin. Passed as ctor args (configurable) but defaults reflect bounce.
- Init-from-frames is only needed if the input is images; feeding true (pos, vel) skips section C entirely (that path is what free-rollout evaluation effectively uses).

## D. State assumptions
- Token state = (x, y, vx, vy, hidden 32). No mass, charge, radius per particle; all balls identical. Other sims with per-particle attributes need extra node features.
- 2D only (unit vectors, tangent as rel_vel minus normal component, 4 walls).
- Box size `n` is a ctor constant used for walls and rasterization.

## E. Data / training
- token_dataset.py / generate_token_dataset.py generate data from bounce.py (gravity 9, stiffness, radius, box) and spawn speed ~3.25 cells/s. Held-out seeds vary initial conditions only, not physics params.
- Losses: token_grid_loss (grid-space), speed loss, contact-weighted loss: grid-space terms and contact weighting assume disc rendering / contact events.

## Minimal changes for a gravity N-body sim
1. Make walls optional (flag default off); pass domain/periodicity as config, not `n-1`.
2. Replace the radius graph with a long-range mechanism (Barnes-Hut/FMM aggregation or global attention); expect O(N log N) or O(N) with FMM.
3. Use summed, antisymmetric pair messages with invariants (r, |v_rel|, and optionally learned pair features) not tied to 2*radius contact.
4. Add per-node features (mass, charge) to node_state.
5. Drop grid rendering/detection from the training path; train on (pos, vel) trajectories with the multi-step unroll loss.
6. Make mirror_sym off (already opt-in) and don't rely on y-symmetry.
