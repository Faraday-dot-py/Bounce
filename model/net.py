import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
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
    """

    def __init__(self, in_channels=3, channels=64, depth=len(DILATIONS), max_flow=4.0):
        super().__init__()
        self.max_flow = max_flow
        self.stem = nn.Conv2d(in_channels, channels, 3, padding=1)
        dilations = [DILATIONS[i % len(DILATIONS)] for i in range(depth)]
        self.blocks = nn.ModuleList([ResidualConvBlock(channels, d) for d in dilations])
        self.flow_head = nn.Conv2d(channels, 2, 3, padding=1)
        self.correction_head = nn.Conv2d(channels, in_channels, 3, padding=1)
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
        correction = self.correction_head(x)

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

        warped = F.grid_sample(g_t, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return warped + correction
