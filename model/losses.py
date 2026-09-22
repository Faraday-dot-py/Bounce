import torch
import torch.nn.functional as F


def weighted_channel_mse(pred, target, weights):
    per_channel = ((pred - target) ** 2).mean(dim=(0, 2, 3))
    return (per_channel * weights).sum()


def _tile_mass_loss(pred_prob, target_prob, tile):
    # Zero-pad to a multiple of `tile` (rather than relying on
    # avg_pool2d's ceil_mode, whose partial-window divisor would
    # otherwise overcount a boundary tile's recovered sum) so every
    # tile's mass is computed over an honest tile*tile denominator.
    h, w = pred_prob.shape[-2:]
    pad_h = (tile - h % tile) % tile
    pad_w = (tile - w % tile) % tile
    pred_padded = F.pad(pred_prob, (0, pad_w, 0, pad_h))
    target_padded = F.pad(target_prob, (0, pad_w, 0, pad_h))
    pred_mass = F.avg_pool2d(pred_padded, kernel_size=tile, stride=tile) * (tile * tile)
    target_mass = F.avg_pool2d(target_padded, kernel_size=tile, stride=tile) * (tile * tile)
    return ((pred_mass - target_mass) ** 2).mean()


def occupancy_weighted_mse(pred, target, source, weights, bg_weight=0.05, peak_weight=0.1,
                            mass_weight=0.0, mass_tile=16):
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

    # The peak term above closes off "blur everywhere" as a cheap hedge,
    # but exposes a second one: once a ball's predicted position has
    # drifted, giving up on it entirely is cheaper than committing to any
    # specific nearby position (a confident-but-wrong prediction is
    # double-penalized: false-positive + missed-cell, vs. give-up's single
    # missed-cell cost). A per-cell max-based regional peak check doesn't
    # fix this either -- a confidently-correct neighboring ball can mask a
    # give-up ball sharing its region, and the masking gets worse exactly
    # as density rises. A per-tile *summed* mass term doesn't have this
    # failure mode: a tile's total mass is directly short by a missing
    # ball's mass regardless of what else is in the tile, so a
    # confidently-correct neighbor can't cancel out a give-up ball's
    # absence. See docs/debugging/findings-mass-conservation-loss.md.
    mass_loss = _tile_mass_loss(pred[:, 0:1, :, :], target[:, 0:1, :, :], mass_tile)

    return (per_channel * weights).sum() + peak_weight * peak_loss + mass_weight * mass_loss
