import torch

from model.token_gate import occluding_mask


def test_occluding_mask_flags_close_pair_not_far_token():
    radius = 0.75
    positions = torch.tensor([
        [10.0, 10.0],
        [10.6, 10.0],   # 0.6 < 2*radius=1.5 from token 0 -> occluding
        [18.0, 18.0],   # far from everything -> not occluding
    ])
    mask = occluding_mask(positions, radius)
    assert mask.tolist() == [True, True, False]


def test_occluding_mask_single_token_is_never_occluding():
    positions = torch.tensor([[5.0, 5.0]])
    mask = occluding_mask(positions, radius=0.75)
    assert mask.tolist() == [False]


def test_occluding_mask_zero_tokens():
    positions = torch.zeros((0, 2))
    mask = occluding_mask(positions, radius=0.75)
    assert mask.shape == (0,)


def test_occluding_mask_boundary_uses_strict_inequality_at_exactly_2r():
    radius = 0.75
    positions = torch.tensor([[10.0, 10.0], [11.5, 10.0]])  # exactly 2*radius apart
    mask = occluding_mask(positions, radius)
    assert mask.tolist() == [False, False]
