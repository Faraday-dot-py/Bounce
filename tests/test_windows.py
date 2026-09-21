import torch
from model.windows import pad_to_multiple, window_partition, window_reverse, compute_shift_mask


def test_pad_to_multiple_pads_up_and_reports_original_size():
    x = torch.randn(1, 10, 10, 4)
    padded, (orig_h, orig_w) = pad_to_multiple(x, window_size=8)
    assert padded.shape == (1, 16, 16, 4)
    assert (orig_h, orig_w) == (10, 10)


def test_pad_to_multiple_no_op_when_already_multiple():
    x = torch.randn(1, 16, 16, 4)
    padded, (orig_h, orig_w) = pad_to_multiple(x, window_size=8)
    assert padded.shape == (1, 16, 16, 4)


def test_window_partition_reverse_round_trip():
    x = torch.randn(2, 16, 16, 4)
    windows = window_partition(x, window_size=8)
    assert windows.shape == (2 * 4, 8, 8, 4)  # 2 batch * (16/8)*(16/8)=4 windows
    reconstructed = window_reverse(windows, window_size=8, H=16, W=16)
    assert torch.allclose(reconstructed, x)


def test_shift_mask_shape_and_self_attend_is_always_allowed():
    mask = compute_shift_mask(H=16, W=16, window_size=8, shift_size=4, device="cpu")
    num_windows = (16 // 8) * (16 // 8)
    assert mask.shape == (num_windows, 64, 64)
    # a token always "attends" to itself (zero relative region id difference)
    diag = torch.diagonal(mask, dim1=-2, dim2=-1)
    assert torch.all(diag == 0.0)


def test_shift_mask_has_some_masked_entries():
    mask = compute_shift_mask(H=16, W=16, window_size=8, shift_size=4, device="cpu")
    assert (mask != 0.0).any()
