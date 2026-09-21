import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x


class Unpatchify(nn.Module):
    def __init__(self, embed_dim, patch_size, out_channels):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

    def forward(self, x):
        B, H, W, _ = x.shape
        x = self.proj(x)
        x = x.view(B, H, W, self.patch_size, self.patch_size, self.out_channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, self.out_channels, H * self.patch_size, W * self.patch_size)
        return x
