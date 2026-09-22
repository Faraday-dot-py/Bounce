from model.losses import weighted_channel_mse


def token_grid_loss(pred_grid, target_grid, weights):
    """Grid-space loss for the token model's rasterized output, reusing
    the existing per-channel MSE weighting (model.losses) so numbers stay
    directly comparable to the flow-warp/windowed-attention baselines.
    Operates on a single (3, n, n) frame -- the token pipeline has no
    batch dimension (see model.token_dataset)."""
    return weighted_channel_mse(pred_grid.unsqueeze(0), target_grid.unsqueeze(0), weights)
