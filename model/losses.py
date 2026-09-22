import torch


def weighted_channel_mse(pred, target, weights):
    per_channel = ((pred - target) ** 2).mean(dim=(0, 2, 3))
    return (per_channel * weights).sum()


def occupancy_weighted_mse(pred, target, source, weights, bg_weight=0.05, peak_weight=0.1):
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

    # Per-pixel occupancy MSE alone, even occupancy-weighted, prefers a
    # diffuse mass-matched blob over a sharp-but-slightly-mispositioned
    # ball as soon as position uncertainty exceeds about one ball radius
    # (confirmed directly on this exact loss formula, not inferred --
    # see docs/debugging/findings-peak-decay-dissolution.md) -- it has no
    # mechanism to penalize spread once mass is inside the occupied mask.
    # This term penalizes the predicted frame's global peak PROB value
    # diverging from the target's, which is cheap to compute (no ball-
    # level ground truth needed) and directly counteracts the observed
    # autoregressive-rollout collapse to a near-uniform low-magnitude
    # field, without overriding the existing per-pixel position signal.
    pred_peak = pred[:, 0:1].amax(dim=(2, 3))
    target_peak = target[:, 0:1].amax(dim=(2, 3))
    peak_loss = ((pred_peak - target_peak) ** 2).mean()

    return (per_channel * weights).sum() + peak_weight * peak_loss
