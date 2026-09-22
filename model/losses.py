import torch


def weighted_channel_mse(pred, target, weights):
    per_channel = ((pred - target) ** 2).mean(dim=(0, 2, 3))
    return (per_channel * weights).sum()


def occupancy_weighted_mse(pred, target, source, weights, bg_weight=0.05):
    target_occ = target[:, 0:1, :, :] > 1e-6
    source_occ = source[:, 0:1, :, :] > 1e-6
    occ = (target_occ | source_occ).float()
    bg = 1.0 - occ

    sq_err = (pred - target) ** 2
    prob_err = sq_err[:, 0:1, :, :]

    # Background is normalized by its own pixel count rather than the total
    # grid size, so bg_weight controls its relative contribution directly
    # instead of being further diluted by grid sparsity (a growing periodic
    # error spread across mostly-background cells was otherwise invisible
    # to the loss regardless of bg_weight).
    occ_count = occ.sum().clamp(min=1.0)
    bg_count = bg.sum().clamp(min=1.0)
    weighted_count = occ_count + bg_weight * bg_count
    prob_loss = ((prob_err * occ).sum() + bg_weight * (prob_err * bg).sum()) / weighted_count

    other_loss = sq_err[:, 1:, :, :].mean(dim=(0, 2, 3))
    per_channel = torch.cat([prob_loss.reshape(1), other_loss])
    return (per_channel * weights).sum()
