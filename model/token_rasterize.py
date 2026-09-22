import torch


def rasterize_tokens(positions, velocities, n, radius):
    """Differentiable equivalent of bounce.py's splat_all for a single
    frame's worth of tokens (no batch dimension -- token count varies per
    sample, so the token pipeline processes one sample at a time).

    positions: (N, 2) tensor, columns (x, y) in grid-cell coordinates.
    velocities: (N, 2) tensor, columns (vx, vy).
    Returns a (3, n, n) tensor in bounce.py's channel order
    (PROB, VX, VY). Matches bounce.py's splat_ball/splat_all exactly:
    each token contributes a linear-falloff disk weight
    `w = max(0, 1 - d/radius)`; the *raw* weight sum is used as the
    VX/VY averaging denominator before PROB is saturated via
    `1 - exp(-sum_w)` -- computing PROB from the saturated value instead
    would double-correct the averaging and not match ground truth.
    """
    device = positions.device
    dtype = positions.dtype
    ii, jj = torch.meshgrid(
        torch.arange(n, device=device, dtype=dtype),
        torch.arange(n, device=device, dtype=dtype),
        indexing="ij",
    )
    x = positions[:, 0].view(-1, 1, 1)
    y = positions[:, 1].view(-1, 1, 1)
    vx = velocities[:, 0].view(-1, 1, 1)
    vy = velocities[:, 1].view(-1, 1, 1)

    d = torch.sqrt((ii.unsqueeze(0) - x) ** 2 + (jj.unsqueeze(0) - y) ** 2 + 1e-12)
    w = torch.clamp(1.0 - d / radius, min=0.0)

    sum_w = w.sum(dim=0)
    vx_sum = (w * vx).sum(dim=0)
    vy_sum = (w * vy).sum(dim=0)
    denom = sum_w.clamp(min=1e-6)

    prob = 1.0 - torch.exp(-sum_w)
    return torch.stack([prob, vx_sum / denom, vy_sum / denom], dim=0)
