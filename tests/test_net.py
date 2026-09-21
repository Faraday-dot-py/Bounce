import torch
from model.net import BounceNextFrameModel


def test_model_output_shape_matches_input():
    model = BounceNextFrameModel(embed_dim=16, depth=2, num_heads=4, window_size=8)
    g_t = torch.randn(2, 3, 50, 50)
    out = model(g_t)
    assert out.shape == g_t.shape


def test_model_runs_at_a_different_resolution_without_retraining():
    model = BounceNextFrameModel(embed_dim=16, depth=2, num_heads=4, window_size=8)
    for n in (50, 300):
        g_t = torch.randn(1, 3, n, n)
        out = model(g_t)
        assert out.shape == (1, 3, n, n)


def test_model_output_is_residual_identity_plus_delta():
    model = BounceNextFrameModel(embed_dim=16, depth=2, num_heads=4, window_size=8)
    g_t = torch.randn(1, 3, 50, 50)
    out = model(g_t)
    assert not torch.equal(out, g_t)  # delta is non-trivial (random init weights)
