import torch


def occluding_mask(positions, radius):
    """Per-token boolean mask: True if this token's position is within
    2*radius of any other token -- the same contact-overlap condition as
    bounce.py's ball_pair_forces (`2 * radius - dist`). Intended to be
    called on *predicted* (not yet observed) positions, per the design
    spec's Occlusion gate section, so the gate anticipates an upcoming
    overlap before it's observed rather than reacting to already-blended
    pixels."""
    n = positions.shape[0]
    if n < 2:
        return torch.zeros(n, dtype=torch.bool, device=positions.device)
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)
    dist = dist + torch.eye(n, device=positions.device, dtype=positions.dtype) * 1e6
    min_dist, _ = dist.min(dim=1)
    return min_dist < (2.0 * radius)
