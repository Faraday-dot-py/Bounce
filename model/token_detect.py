import math

import torch
import torch.nn.functional as F

from model.token_gate import occluding_mask


def territory_mask(ii, jj, positions, self_idx):
    """Boolean mask, same shape as ii/jj (a window's per-cell coordinate
    grids, broadcastable to (h, w)): True where a cell is at least as
    close to positions[self_idx] as to every other row of positions --
    a Voronoi partition over currently-tracked token positions. Ties
    favor self_idx (a cell exactly equidistant from two tokens belongs
    to both of their masks, never to neither), so the total masked area
    across all tokens never has a gap. Distance is plain Euclidean in
    grid-cell space; this is not derived from `radius` or any other
    physical constant -- it only asks "which token is nearest," using
    whatever positions are currently tracked.

    See docs/superpowers/specs/2026-09-23-token-territory-masking-design.md."""
    self_pos = positions[self_idx]
    self_dist_sq = (ii - self_pos[0]) ** 2 + (jj - self_pos[1]) ** 2
    mask = torch.ones_like(ii, dtype=torch.bool)
    for k in range(positions.shape[0]):
        if k == self_idx:
            continue
        other = positions[k]
        other_dist_sq = (ii - other[0]) ** 2 + (jj - other[1]) ** 2
        mask &= self_dist_sq <= other_dist_sq
    return mask


def centroid_near(prob, position, radius, margin=1.0, max_expansions=3,
                   all_positions=None, self_idx=None):
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
    recovering the token's own ball, which is worse than staying lost.

    When `all_positions` (all currently-tracked token positions) and
    `self_idx` (this token's row in that tensor) are both given, window
    mass outside this token's territory (see `territory_mask`) is
    excluded before computing the centroid -- a token's window can never
    read a cell that rightfully belongs to a different tracked token.
    Both default to None, which reproduces the prior unmasked behavior
    exactly (only `find_token_positions`, called before any persistent
    tokens exist, relies on this default)."""
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

        ii = torch.arange(i_lo, i_hi + 1, device=prob.device, dtype=prob.dtype).view(-1, 1)
        jj = torch.arange(j_lo, j_hi + 1, device=prob.device, dtype=prob.dtype).view(1, -1)

        if all_positions is not None and self_idx is not None:
            ii_grid = ii.expand(window.shape[0], window.shape[1])
            jj_grid = jj.expand(window.shape[0], window.shape[1])
            mask = territory_mask(ii_grid, jj_grid, all_positions, self_idx)
            window = window * mask.to(window.dtype)

        total = window.sum()
        if total <= 1e-6:
            continue

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


def find_token_positions(prob, radius, threshold=0.1, margin=1.0):
    """Detect one token per isolated ball via non-max suppression over a
    window sized to one ball's footprint, refined by `centroid_near`.
    Used only for sequence-start initialization -- per-step tracking uses
    `centroid_near` directly against each token's predicted position
    instead of re-running full-grid detection. Two balls closer together
    than the window are detected as a single peak, an accepted rare
    failure mode (see design spec's Token initialization section).

    Raw peaks within `centroid_near`'s own contamination distance of
    each other (`radius + margin`, the same threshold `TokenModel`'s
    occlusion gate uses during tracking -- see `TokenModel._gate_radius`)
    skip refinement and keep their coarse integer coordinate instead.
    Without this, a weaker ball's peak can get its centroid pulled
    entirely into a stronger neighbour's mass -- reproduced directly
    (docs/debugging/experiment-log.md): two true balls 2.85 cells apart
    each detected correctly at the raw-peak stage, but the weaker one's
    `centroid_near` window reached far enough to catch the stronger
    ball's much larger mass in one corner, dragging its refined position
    on top of the stronger ball's and silently losing its own detection.
    A coarse (integer-cell) position is less precise but can't be
    hijacked this way, and per-step tracking's own occlusion gate already
    accepts the same tradeoff (skipping the correction rather than
    risking a contaminated one) once dynamics take over."""
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
    occluding = occluding_mask(coords, radius + margin)
    return torch.stack([
        coords[i] if occluding[i] else centroid_near(prob, coords[i], radius, margin=margin)
        for i in range(coords.shape[0])
    ])
