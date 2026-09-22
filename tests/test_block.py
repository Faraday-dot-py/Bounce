from unittest.mock import patch

import torch

import model.block as block_module
from model.block import SwinBlock
from model.windows import pad_to_multiple as base_pad_to_multiple


def test_default_offset_matches_old_fixed_shift_behavior():
    # randomize_offset defaults to False and offset defaults to None -> 0,
    # so existing (pre-jitter) callers see identical, deterministic behavior.
    torch.manual_seed(4738)
    block = SwinBlock(dim=16, num_heads=4, window_size=8, shift=True)
    block.eval()
    x = torch.randn(1, 16, 16, 16)
    out1 = block(x)
    out2 = block(x)
    assert torch.equal(out1, out2)


def test_explicit_offset_changes_window_alignment():
    torch.manual_seed(4738)
    block = SwinBlock(dim=16, num_heads=4, window_size=8, shift=False)
    block.eval()
    x = torch.randn(1, 16, 16, 16)
    out_offset0 = block(x, offset=0)
    out_offset3 = block(x, offset=3)
    assert out_offset0.shape == out_offset3.shape == x.shape
    assert not torch.equal(out_offset0, out_offset3)


def test_randomize_offset_draws_from_torch_randint():
    block = SwinBlock(dim=16, num_heads=4, window_size=8, shift=False, randomize_offset=True)
    block.eval()
    x = torch.randn(1, 16, 16, 16)
    with patch.object(block_module.torch, "randint", return_value=torch.tensor([5])) as mock_randint:
        block(x)
    mock_randint.assert_called_once()
    args, _ = mock_randint.call_args
    assert args[0] == 0 and args[1] == 8  # low, high bounds are (0, window_size)


def test_randomize_offset_still_preserves_shape_and_runs():
    block = SwinBlock(dim=16, num_heads=4, window_size=8, shift=True, randomize_offset=True)
    x = torch.randn(2, 25, 25, 16)  # non-multiple-of-window-size, exercises padding too
    for _ in range(5):
        out = block(x)
        assert out.shape == x.shape


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


def test_swin_block_preserves_shape_non_multiple_of_window_no_shift():
    block = SwinBlock(dim=16, num_heads=4, window_size=8, shift=False)
    x = torch.randn(2, 25, 25, 16)
    out = block(x)
    assert out.shape == x.shape


def test_swin_block_padding_does_not_leak_into_valid_region():
    # Regression test: padding-validity mask must actually suppress attention
    # to padded tokens in shifted blocks. Combining the shift mask and the
    # validity mask with the wrong operator (torch.maximum instead of
    # torch.minimum) silently discards the validity mask wherever the shift
    # mask allows attention, letting padded content leak into valid outputs.
    # window_size=8, shift_size=4: input 19x19 pads to 24x24 (pad amount 5),
    # which straddles the shift-mask's own group boundary at [16, 20), so
    # some padded tokens fall in the same shift-window group as valid
    # tokens. This is the geometry that actually exercises the bug -- a
    # pad amount equal to shift_size (e.g. 20x20 -> 24x24) lines up exactly
    # with the shift-mask boundary and masks out the padding regardless of
    # the validity mask, hiding the bug.
    torch.manual_seed(4738)
    block = SwinBlock(dim=16, num_heads=4, window_size=8, shift=True)
    block.eval()
    x = torch.randn(1, 19, 19, 16)  # pads to 24x24 with window_size=8

    def make_pad_fn(fill_fn):
        def _pad(inp, window_size):
            padded, orig = base_pad_to_multiple(inp, window_size)
            H, W = orig
            Hp, Wp = padded.shape[1], padded.shape[2]
            if Hp != H or Wp != W:
                fill = fill_fn(padded.shape)
                region_mask = torch.ones_like(padded)
                region_mask[:, :H, :W, :] = 0
                padded = padded + fill * region_mask
            return padded, orig
        return _pad

    with patch.object(block_module, "pad_to_multiple", make_pad_fn(lambda shape: torch.zeros(shape))):
        out_zeros = block(x)

    torch.manual_seed(999)
    with patch.object(block_module, "pad_to_multiple", make_pad_fn(lambda shape: torch.randn(shape) * 10)):
        out_noise = block(x)

    assert torch.allclose(out_zeros[:, :19, :19, :], out_noise[:, :19, :19, :], atol=1e-5)
