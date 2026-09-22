import torch


def weighted_channel_mse(pred, target, weights):
    per_channel = ((pred - target) ** 2).mean(dim=(0, 2, 3))
    return (per_channel * weights).sum()


def occupancy_weighted_mse(pred, target, source, weights, bg_weight=0.05):
    target_occ = target[:, 0:1, :, :] > 1e-6
    source_occ = source[:, 0:1, :, :] > 1e-6
    occ = target_occ | source_occ
    prob_spatial_weight = torch.where(occ, 1.0, bg_weight)
    spatial_weight = torch.ones_like(target)
    spatial_weight[:, 0:1, :, :] = prob_spatial_weight
    weighted_sq_err = (pred - target) ** 2 * spatial_weight
    per_channel = weighted_sq_err.mean(dim=(0, 2, 3))
    return (per_channel * weights).sum()
