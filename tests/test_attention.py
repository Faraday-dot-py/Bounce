import torch
from model.attention import RelativePositionBias


def test_relative_position_bias_shape():
    bias_module = RelativePositionBias(window_size=8, num_heads=4)
    bias = bias_module()
    assert bias.shape == (4, 64, 64)


def test_relative_position_bias_table_size_independent_of_grid():
    # table size depends only on window_size, never on any grid dimension
    bias_module = RelativePositionBias(window_size=8, num_heads=4)
    expected_table_size = (2 * 8 - 1) * (2 * 8 - 1)
    assert bias_module.bias_table.shape == (expected_table_size, 4)
