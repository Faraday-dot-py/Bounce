import math

import torch
import torch.nn.functional as F


def centroid_near(prob, position, radius, margin=1.0):
    """Intensity-weighted centroid of `prob` (n, n) within a square window
    around `position` (x, y) -- refines a coarse detection (or a token's
    predicted position) into a sub-pixel observation. Differentiable in
    `prob`'s values (not in the window's location, which is derived from
    a detached rounded position). Returns `position` unchanged if the
    window has no mass, e.g. the predicted position and the true ball
    have drifted further apart than the window covers."""
    n = prob.shape[0]
    half = int(math.ceil(radius + margin))
    cx = int(round(float(position[0].detach())))
    cy = int(round(float(position[1].detach())))
    # Bail out before clamping if the raw window misses the grid entirely:
    # clamping an off-grid window can leave i_lo > i_hi, which both slices
    # nonsense (negative indices wrap) and makes the arange below raise.
    i_lo_raw, i_hi_raw = cx - half, cx + half
    j_lo_raw, j_hi_raw = cy - half, cy + half
    if i_hi_raw < 0 or i_lo_raw > n - 1 or j_hi_raw < 0 or j_lo_raw > n - 1:
        return position
    i_lo, i_hi = max(0, i_lo_raw), min(n - 1, i_hi_raw)
    j_lo, j_hi = max(0, j_lo_raw), min(n - 1, j_hi_raw)
    window = prob[i_lo:i_hi + 1, j_lo:j_hi + 1]
    total = window.sum()
    if total <= 1e-6:
        return position

    ii = torch.arange(i_lo, i_hi + 1, device=prob.device, dtype=prob.dtype).view(-1, 1)
    jj = torch.arange(j_lo, j_hi + 1, device=prob.device, dtype=prob.dtype).view(1, -1)
    x = (window * ii).sum() / total
    y = (window * jj).sum() / total
    return torch.stack([x, y])


def find_token_positions(prob, radius, threshold=0.1):
    """Detect one token per isolated ball via non-max suppression over a
    window sized to one ball's footprint, refined by `centroid_near`.
    Used only for sequence-start initialization -- per-step tracking uses
    `centroid_near` directly against each token's predicted position
    instead of re-running full-grid detection. Two balls closer together
    than the window are detected as a single peak, an accepted rare
    failure mode (see design spec's Token initialization section)."""
    n = prob.shape[0]
    k = max(1, int(round(radius)) * 2 + 1)
    padded = prob.unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(padded, kernel_size=k, stride=1, padding=k // 2)
    pooled = pooled[0, 0, :n, :n]
    is_peak = (prob == pooled) & (prob > threshold)
    coords = torch.nonzero(is_peak, as_tuple=False).to(prob.dtype)
    if coords.shape[0] == 0:
        return torch.zeros((0, 2), dtype=prob.dtype, device=prob.device)
    return torch.stack([centroid_near(prob, c, radius) for c in coords])
