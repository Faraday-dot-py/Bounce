import math

import torch
import torch.nn.functional as F


def centroid_near(prob, position, radius, margin=1.0, max_expansions=3):
    """Intensity-weighted centroid of `prob` (n, n) within a square window
    around `position` (x, y) -- refines a coarse detection (or a token's
    predicted position) into a sub-pixel observation. Differentiable in
    `prob`'s values (not in the window's location, which is derived from
    a detached rounded position).

    If the base window (radius `ceil(radius + margin)`) has no mass, the
    search widens by one cell per retry, up to `max_expansions` times,
    before giving up and returning `position` unchanged -- a token whose
    predicted position has drifted a few cells from its ball otherwise
    hits a permanent dead end (see docs/debugging/experiment-log.md,
    "Bailout confirmed directly": the un-widened version never recovers
    once the base window goes empty). The widening is intentionally
    modest, not unbounded: past a few cells it risks pulling in a
    different ball's mass entirely (an identity swap) rather than
    recovering the token's own ball, which is worse than staying lost."""
    n = prob.shape[0]
    cx = int(round(float(position[0].detach())))
    cy = int(round(float(position[1].detach())))
    step = max(1, int(math.ceil(radius)))
    base_half = int(math.ceil(radius + margin))
    for expansion in range(max_expansions + 1):
        half = base_half + expansion * step
        # Bail out before clamping if the raw window misses the grid
        # entirely: clamping an off-grid window can leave i_lo > i_hi,
        # which both slices nonsense (negative indices wrap) and makes
        # the arange below raise.
        i_lo_raw, i_hi_raw = cx - half, cx + half
        j_lo_raw, j_hi_raw = cy - half, cy + half
        if i_hi_raw < 0 or i_lo_raw > n - 1 or j_hi_raw < 0 or j_lo_raw > n - 1:
            continue
        i_lo, i_hi = max(0, i_lo_raw), min(n - 1, i_hi_raw)
        j_lo, j_hi = max(0, j_lo_raw), min(n - 1, j_hi_raw)
        window = prob[i_lo:i_hi + 1, j_lo:j_hi + 1]
        total = window.sum()
        if total <= 1e-6:
            continue

        ii = torch.arange(i_lo, i_hi + 1, device=prob.device, dtype=prob.dtype).view(-1, 1)
        jj = torch.arange(j_lo, j_hi + 1, device=prob.device, dtype=prob.dtype).view(1, -1)
        x = (window * ii).sum() / total
        y = (window * jj).sum() / total
        return torch.stack([x, y])
    return position


def _suppress_tied_peaks(coords, prob, radius):
    """`prob == pooled` non-max suppression doesn't break ties: a ball
    centered near a half-integer coordinate splats equal mass onto its
    two straddled cells, so both satisfy `prob == pooled` and register
    as separate peaks for what is one ball (see
    docs/debugging/experiment-log.md -- reproduced directly with a
    ball at x=10.5, radius=1.5: cells x=10 and x=11 tie at 0.487 and
    both pass). Keeps the highest-value peak in any cluster of
    candidates within `radius` of each other, discarding the rest --
    the same physical-footprint-sized threshold `find_token_positions`
    already uses to accept merging close-together balls into one peak,
    so this doesn't introduce a new merge distance, just makes the
    existing one deterministic under ties."""
    if coords.shape[0] <= 1:
        return coords
    vals = prob[coords[:, 0].long(), coords[:, 1].long()]
    order = torch.argsort(vals, descending=True)
    kept = []
    for idx in order.tolist():
        c = coords[idx]
        if all(float(torch.norm(c - coords[k])) > radius for k in kept):
            kept.append(idx)
    kept = sorted(kept)
    return coords[torch.tensor(kept, dtype=torch.long, device=coords.device)]


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
    coords = _suppress_tied_peaks(coords, prob, radius)
    return torch.stack([centroid_near(prob, c, radius) for c in coords])
