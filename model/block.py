import torch
import torch.nn as nn

from model.windows import pad_to_multiple, window_partition, window_reverse, compute_shift_mask
from model.attention import WindowAttention


class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift, mlp_ratio=4.0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)

        x, (orig_H, orig_W) = pad_to_multiple(x, self.window_size)
        Hp, Wp = x.shape[1], x.shape[2]

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            mask = compute_shift_mask(Hp, Wp, self.window_size, self.shift_size, x.device)
        else:
            mask = None

        windows = window_partition(x, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, C)
        attn_out = self.attn(windows, mask=mask)
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_out, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x[:, :orig_H, :orig_W, :]
        x = shortcut + x

        x = x + self.mlp(self.norm2(x))
        return x
