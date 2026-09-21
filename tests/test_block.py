import torch
from model.block import SwinBlock


def test_swin_block_preserves_shape_multiple_of_window():
    block = SwinBlock(dim=16, num_heads=4, window_size=8, shift=False)
    x = torch.randn(2, 16, 16, 16)
    out = block(x)
    assert out.shape == x.shape


def test_swin_block_preserves_shape_non_multiple_of_window():
    block = SwinBlock(dim=16, num_heads=4, window_size=8, shift=True)
    x = torch.randn(2, 25, 25, 16)  # not divisible by 8
    out = block(x)
    assert out.shape == x.shape


def test_swin_block_shift_and_noshift_both_run():
    x = torch.randn(1, 16, 16, 16)
    for shift in (False, True):
        block = SwinBlock(dim=16, num_heads=4, window_size=8, shift=shift)
        out = block(x)
        assert out.shape == x.shape
