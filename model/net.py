import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, padding_mode="replicate")
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, padding_mode="replicate")
        self.act = nn.SiLU()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


# Dilation schedule grows the receptive field (up to ~33px at dilation=8)
# without any downsampling, so the backbone stays translation-equivariant
# at every layer -- no window grid or fixed patch alignment to leak into
# the output, unlike the windowed-attention architecture this replaces.
DILATIONS = (1, 2, 4, 8, 4, 2, 1)


class BounceNextFrameModel(nn.Module):
    """Predicts g_{t+1} from g_t by warping the grid along a predicted
    per-cell flow field (advection) plus a small local correction, instead
    of regressing new pixel values from scratch. Direct regression under
    an MSE loss blurs sharp ball positions into a diffuse average; a warp
    can represent a sharp translated disk exactly.

    Flow and correction heads are zero-initialized so the model starts as
    an exact identity (out == g_t), matching the fact that "copy the last
    frame" is already a strong single-step baseline.

    correction is tanh-bounded by max_correction, mirroring how flow is
    tanh-bounded by max_flow -- an unbounded correction term was found to
    give the autoregressive rollout a Jacobian eigenvalue < -1 (a period-2
    limit cycle), see docs/debugging/findings-period2-oscillation.md.
    grid_sample uses padding_mode="border" (not "zeros") and the conv
    stack uses padding_mode="replicate" so border pixels don't see a hard
    synthetic-zero discontinuity, see
    docs/debugging/findings-gridding-artifact.md.

    correction is also re-centered per-channel per-step (its own spatial
    mean subtracted) so it can only redistribute mass, not inject a
    sustained per-channel bias -- an uncentered correction acted as an
    undamped integrator, accumulating a one-directional drift over many
    rollout steps. grid_sample uses mode="bicubic" instead of "bilinear":
    repeated bilinear resampling is a structural source of numerical
    diffusion under autoregressive self-feed (confirmed with a zero-model,
    ground-truth-flow probe -- peak intensity collapsed to 13% by step 10
    from bilinear resampling alone), bicubic preserves peak sharpness
    substantially better. See
    docs/debugging/findings-correction-drift-and-mass-dissolution.md.

    The PROB channel (0) is renormalized after each step so its frame-
    wide sum matches the pre-warp frame's sum. grid_sample does no
    Jacobian-determinant correction for locally convergent/divergent flow
    fields, so any region where predicted flow compresses (converges) is
    structurally oversampled every step regardless of padding_mode --
    this drove a persistent PROB->VX/VY drift even after the fixes above.
    VX/VY (channels 1/2) are left unrenormalized since they're physical
    velocity fields, not a conserved quantity -- pinning their frame-wide
    mean to the initial frame isn't physically justified the way it is
    for occupied "mass". See
    docs/debugging/findings-padding-mass-conservation.md.

    Before renormalization, PROB is soft-thresholded (`relu(prob -
    prob_threshold)`): repeated bicubic resampling diffuses PROB mass
    into low-magnitude noise across nearly the entire grid by ~step 8
    (confirmed against ground truth, which stays sparse), and exact
    mass renormalization then perpetually rescales that near-background
    diffusion noise back up to the true total mass every step, since
    the renorm sum has no way to distinguish real occupancy from
    diffused noise -- this produced a persistent, growing blocky/tiled
    "quilt" artifact. Thresholding before the sum (and before scaling)
    removes the near-zero diffusion tail so it can't be counted as
    mass. See docs/debugging/findings-quilting-artifact.md.
    """

    def __init__(self, in_channels=3, channels=64, depth=len(DILATIONS), max_flow=4.0, max_correction=0.2, prob_threshold=0.015):
        super().__init__()
        self.max_flow = max_flow
        self.max_correction = max_correction
        self.prob_threshold = prob_threshold
        self.stem = nn.Conv2d(in_channels, channels, 3, padding=1, padding_mode="replicate")
        dilations = [DILATIONS[i % len(DILATIONS)] for i in range(depth)]
        self.blocks = nn.ModuleList([ResidualConvBlock(channels, d) for d in dilations])
        self.flow_head = nn.Conv2d(channels, 2, 3, padding=1, padding_mode="replicate")
        self.correction_head = nn.Conv2d(channels, in_channels, 3, padding=1, padding_mode="replicate")
        nn.init.zeros_(self.flow_head.weight)
        nn.init.zeros_(self.flow_head.bias)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def forward(self, g_t):
        B, C, H, W = g_t.shape
        x = self.stem(g_t)
        for block in self.blocks:
            x = block(x)
        flow = torch.tanh(self.flow_head(x)) * self.max_flow
        correction = torch.tanh(self.correction_head(x)) * self.max_correction
        correction = correction - correction.mean(dim=(2, 3), keepdim=True)

        ys, xs = torch.meshgrid(
            torch.arange(H, device=g_t.device, dtype=g_t.dtype),
            torch.arange(W, device=g_t.device, dtype=g_t.dtype),
            indexing="ij",
        )
        sample_x = xs.unsqueeze(0) + flow[:, 0]
        sample_y = ys.unsqueeze(0) + flow[:, 1]
        norm_x = sample_x / max(W - 1, 1) * 2 - 1
        norm_y = sample_y / max(H - 1, 1) * 2 - 1
        sample_grid = torch.stack([norm_x, norm_y], dim=-1)

        warped = F.grid_sample(g_t, sample_grid, mode="bicubic", padding_mode="border", align_corners=True)
        out = warped + correction

        prob_thresholded = F.relu(out[:, 0:1] - self.prob_threshold)
        prob_in = g_t[:, 0:1].sum(dim=(2, 3), keepdim=True)
        prob_out = prob_thresholded.sum(dim=(2, 3), keepdim=True)
        scale = prob_in / (prob_out + 1e-6)
        out = torch.cat([prob_thresholded * scale, out[:, 1:]], dim=1)
        return out
