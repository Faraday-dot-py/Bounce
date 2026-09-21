import torch.nn as nn

from model.patchify import PatchEmbed, Unpatchify
from model.block import SwinBlock


class BounceNextFrameModel(nn.Module):
    def __init__(self, in_channels=3, patch_size=2, embed_dim=128, depth=6,
                 num_heads=4, window_size=8, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels, patch_size, embed_dim)
        self.blocks = nn.ModuleList([
            SwinBlock(embed_dim, num_heads, window_size, shift=(i % 2 == 1), mlp_ratio=mlp_ratio)
            for i in range(depth)
        ])
        self.unpatchify = Unpatchify(embed_dim, patch_size, in_channels)

    def forward(self, g_t):
        x = self.patch_embed(g_t)
        for block in self.blocks:
            x = block(x)
        delta = self.unpatchify(x)
        return g_t + delta
