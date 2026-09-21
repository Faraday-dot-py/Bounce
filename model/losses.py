import torch


def weighted_channel_mse(pred, target, weights):
    per_channel = ((pred - target) ** 2).mean(dim=(0, 2, 3))
    return (per_channel * weights).sum()


def occupancy_weighted_mse(pred, target, weights, bg_weight=0.05):
    spatial_weight = torch.where(target[:, 0:1, :, :] > 1e-6, 1.0, bg_weight)
    weighted_sq_err = (pred - target) ** 2 * spatial_weight
    per_channel = weighted_sq_err.mean(dim=(0, 2, 3))
    return (per_channel * weights).sum()
