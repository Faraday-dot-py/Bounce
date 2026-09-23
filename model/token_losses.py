import torch

from model.losses import occupancy_weighted_mse


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


def boundary_loss(positions, n, margin=0.0):
    """Penalizes token positions outside [margin, n - margin) on either
    axis. token_grid_loss only sees the rasterized grid, so it can only
    react to a token vanishing off-grid after the fact; this operates on
    the tracked (x, y) positions themselves (model.token_model.TokenModel.step's
    `final_pos`), the same quantity that actually drifts, so it fires the
    moment a token starts leaving rather than once it's already gone.

    Grows quadratically with distance past the boundary, zero otherwise --
    same shape as occupancy_weighted_mse's peak/mass terms, so it composes
    the same way as an additive penalty."""
    lower = torch.relu(margin - positions)
    upper = torch.relu(positions - (n - margin))
    return (lower ** 2 + upper ** 2).mean()
