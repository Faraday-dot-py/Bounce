import torch
from model.attention import RelativePositionBias, WindowAttention


def test_relative_position_bias_shape():
    bias_module = RelativePositionBias(window_size=8, num_heads=4)
    bias = bias_module()
    assert bias.shape == (4, 64, 64)


def test_relative_position_bias_table_size_independent_of_grid():
    # table size depends only on window_size, never on any grid dimension
    bias_module = RelativePositionBias(window_size=8, num_heads=4)
    expected_table_size = (2 * 8 - 1) * (2 * 8 - 1)
    assert bias_module.bias_table.shape == (expected_table_size, 4)


def test_window_attention_output_shape_no_mask():
    attn = WindowAttention(dim=16, window_size=8, num_heads=4)
    x = torch.randn(3, 64, 16)  # 3 windows, 64 tokens each, dim 16
    out = attn(x)
    assert out.shape == (3, 64, 16)


def test_window_attention_output_shape_with_mask():
    attn = WindowAttention(dim=16, window_size=8, num_heads=4)
    num_windows = 4
    x = torch.randn(num_windows * 2, 64, 16)  # batch 2, 4 windows each
    mask = torch.zeros(num_windows, 64, 64)
    out = attn(x, mask=mask)
    assert out.shape == (num_windows * 2, 64, 16)
