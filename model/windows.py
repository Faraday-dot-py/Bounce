import torch
import torch.nn.functional as F


def pad_to_multiple(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    return x, (H, W)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    B = windows.shape[0] // (num_windows_h * num_windows_w)
    x = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


def compute_shift_mask(H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, H, W, 1), device=device)
    h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1
    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


def compute_validity_mask(Hp, Wp, orig_H, orig_W, window_size, device, roll_shift=0):
    valid = torch.zeros((1, Hp, Wp, 1), device=device)
    valid[:, :orig_H, :orig_W, :] = 1.0
    if roll_shift:
        valid = torch.roll(valid, shifts=(-roll_shift, -roll_shift), dims=(1, 2))
    mask_windows = window_partition(valid, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    pair_valid = mask_windows.unsqueeze(1) * mask_windows.unsqueeze(2)
    return torch.where(pair_valid > 0, torch.zeros_like(pair_valid), torch.full_like(pair_valid, -100.0))
