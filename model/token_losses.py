import math

import torch

from model.losses import occupancy_weighted_mse
from model.token_detect import territory_mask


def token_grid_loss(pred_grid, target_grid, source_grid, weights, bg_weight=0.05,
                     peak_weight=0.1, mass_weight=0.0, mass_tile=16):
    """Grid-space loss for the token model's rasterized output, reusing
    the existing occupancy-weighted MSE (model.losses) -- not the plain
    per-pixel weighted_channel_mse -- so numbers stay directly comparable
    to the flow-warp/windowed-attention baselines' actual training loss,
    not just their eval metric.

    This choice is load-bearing, not cosmetic: a first real training run
    (job 2840) using plain weighted_channel_mse converged to a "give up"
    degenerate solution -- delta_head learned a large constant offset
    that flings every token off-grid within one step, producing an
    all-background prediction. Under a plain, unweighted per-pixel MSE
    over a grid that's ~97% background, "predict nothing" scores better
    than a present-but-imperfectly-positioned ball, because VX/VY error
    at a wrong-but-occupied cell is far more expensive than matching zero
    background everywhere. This is the same failure mode already
    characterized for the flow-warp architecture (see
    docs/debugging/findings-peak-decay-dissolution.md) -- occupancy_weighted_mse's
    bg_weight down-weighting and peak/mass terms exist specifically to
    close off that shortcut, which is why this reuses it rather than the
    plainer weighted_channel_mse the spec's Loss section originally
    described.

    `source_grid` is the frame the model stepped from (this call's
    `observed_frame`), used identically to model.train.rollout_loss's
    `frame` argument to build the occupancy union mask. Operates on a
    single (3, n, n) frame -- the token pipeline has no batch dimension
    (see model.token_dataset)."""
    return occupancy_weighted_mse(
        pred_grid.unsqueeze(0), target_grid.unsqueeze(0), source_grid.unsqueeze(0),
        weights, bg_weight=bg_weight, peak_weight=peak_weight,
        mass_weight=mass_weight, mass_tile=mass_tile,
    )


def token_state_loss(final_pos, final_vel, target_state, match_idx, vel_weight=0.1, speed_weight=0.0):
    """Direct per-token MSE against the dataset's ground-truth ball state
    (model.token_dataset's state_seq entries), using the correspondence
    `match_idx` from model.token_match.match_tokens_to_state. Unlike
    token_grid_loss, this never rasterizes -- there is no background
    cell for a token to hide behind, so "predict nothing" has no lower
    cost here regardless of grid occupancy, which is what closes off the
    give-up degenerate solution structurally rather than through an
    additive penalty (see docs/debugging/experiment-log.md).

    vel_weight down-weights the velocity term since VX/VY and X/Y are on
    different natural scales (velocities are the same order of magnitude
    as one grid cell per dt, positions span the whole grid).

    speed_weight adds a direction-agnostic |v| term: once phase is chaotic
    the MSE is minimized by shrinking velocity toward the conditional mean,
    which bleeds kinetic energy out of long rollouts (see docs/debugging/
    experiment-log.md, energy audit); penalizing |v_pred| vs |v_true|
    directly does not reward that shrinkage. Default 0 (off)."""
    if final_pos.shape[0] == 0:
        return final_pos.new_zeros(())
    target_pos = torch.stack([target_state["x"], target_state["y"]], dim=1)[match_idx]
    target_vel = torch.stack([target_state["vx"], target_state["vy"]], dim=1)[match_idx]
    pos_loss = ((final_pos - target_pos) ** 2).mean()
    vel_loss = ((final_vel - target_vel) ** 2).mean()
    loss = pos_loss + vel_weight * vel_loss
    if speed_weight > 0.0:
        speed_err = torch.sqrt((final_vel ** 2).sum(dim=-1) + 1e-8) - torch.sqrt((target_vel ** 2).sum(dim=-1) + 1e-8)
        loss = loss + speed_weight * (speed_err ** 2).mean()
    return loss


def window_collapse_loss(prob, positions, radius, margin=1.0, floor=0.3,
                          all_positions=None, self_idx_offset=0):
    """Penalizes a token whose own rasterized PROB mass near its tracked
    position falls below `floor`, closing the give-up shortcut identified
    in docs/debugging/experiment-log.md's v9-v12 synthesis: neither
    token_state_loss nor token_grid_loss ever looks at whether a token's
    own rasterization deposits detectable mass at the position it
    reports, so a token that drifts its window empty (dropout) costs
    nothing beyond whatever state-space error that drift happens to
    cause -- vanishing is free. This reads `prob` (rasterize_tokens'
    PROB channel for THIS step's own prediction, not the observed/source
    frame) at each token's own `positions`, so the gradient flows back
    through the same forward pass that produced both, unlike the
    observation branch's `centroid_near` reads which are detached during
    self-feed.

    Window size mirrors centroid_near's un-widened base window
    (`ceil(radius + margin)`) -- deliberately not its widened retry, so a
    token doesn't get this penalty's credit for centroid_near's separate
    recovery leniency. `floor` is calibrated against a single
    well-centered ball's own window mass (~0.63 on-lattice, ~0.22 at a
    half-integer offset -- see scripts/calibrate_window_total.py-style
    check) so normal sub-pixel jitter stays well clear of it while an
    actually-collapsed token's near-zero mass triggers it hard.

    The window's location is derived from a detached, rounded position
    (matching centroid_near) -- this penalizes the network for the mass
    it puts down, not for moving the position to chase existing mass.

    When `all_positions` is given (the full tracked-token tensor this
    call's `positions` is drawn from, or identical to `positions` itself
    when every token is being penalized in one call), each token i's
    window sum excludes mass outside its territory (see
    model.token_detect.territory_mask) before comparing against `floor`
    -- otherwise a token near a healthy neighbor could get credit for
    mass that was never its own, reintroducing the incentive this loss
    exists to remove (see docs/superpowers/specs/2026-09-23-token-territory-masking-design.md).
    `self_idx_offset` lets a caller penalize a subset of `positions` that
    starts partway through `all_positions` (0 in every current call
    site)."""
    if positions.shape[0] == 0:
        return positions.new_zeros(())
    n = prob.shape[0]
    half = int(math.ceil(radius + margin))
    totals = []
    for i in range(positions.shape[0]):
        cx = int(round(float(positions[i, 0].detach())))
        cy = int(round(float(positions[i, 1].detach())))
        i_lo, i_hi = max(0, cx - half), min(n - 1, cx + half)
        j_lo, j_hi = max(0, cy - half), min(n - 1, cy + half)
        if i_lo > i_hi or j_lo > j_hi:
            totals.append(prob.new_zeros(()))
            continue
        window = prob[i_lo:i_hi + 1, j_lo:j_hi + 1]
        if all_positions is not None:
            ii = torch.arange(i_lo, i_hi + 1, device=prob.device, dtype=prob.dtype).view(-1, 1)
            jj = torch.arange(j_lo, j_hi + 1, device=prob.device, dtype=prob.dtype).view(1, -1)
            ii_grid = ii.expand(window.shape[0], window.shape[1])
            jj_grid = jj.expand(window.shape[0], window.shape[1])
            mask = territory_mask(ii_grid, jj_grid, all_positions, self_idx_offset + i)
            window = window * mask.to(window.dtype)
        totals.append(window.sum())
    total = torch.stack(totals)
    deficit = torch.relu(floor - total)
    return (deficit ** 2).mean()


def boundary_loss(positions, n, margin=0.0):
    """Penalizes token positions outside [margin, n - margin) on either
    axis. token_grid_loss only sees the rasterized grid, so it can only
    react to a token vanishing off-grid after the fact; this operates on
    the tracked (x, y) positions themselves (model.token_model.TokenModel.step's
    `final_pos`), the same quantity that actually drifts, so it fires the
    moment a token starts leaving rather than once it's already gone.

    Grows quadratically with distance past the boundary, zero otherwise --
    same shape as occupancy_weighted_mse's peak/mass terms, so it composes
    the same way as an additive penalty.

    A zero-token rollout step (init_tokens detected no balls) previously
    hit `.mean()` on an empty tensor, which is NaN in PyTorch -- and
    `weight * nan` is still nan even at weight 0.0, so this leaked into
    training logs regardless of boundary_weight (job 2857/v13, see
    docs/debugging/experiment-log.md). Harmless to trained weights in
    practice (backward through a zero-cardinality path can't actually
    multiply a real nan into any parameter's gradient), but the loss
    value itself should be a real zero, matching token_state_loss's and
    window_collapse_loss's existing empty-token guards."""
    if positions.shape[0] == 0:
        return positions.new_zeros(())
    lower = torch.relu(margin - positions)
    upper = torch.relu(positions - (n - margin))
    return (lower ** 2 + upper ** 2).mean()
