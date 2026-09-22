import torch
from model.net import BounceNextFrameModel


def test_model_output_shape_matches_input():
    model = BounceNextFrameModel(channels=16, depth=2)
    g_t = torch.randn(2, 3, 50, 50)
    out = model(g_t)
    assert out.shape == g_t.shape


def test_model_runs_at_a_different_resolution_without_retraining():
    model = BounceNextFrameModel(channels=16, depth=2)
    for n in (50, 300):
        g_t = torch.randn(1, 3, n, n)
        out = model(g_t)
        assert out.shape == (1, 3, n, n)


def test_zero_init_heads_give_exact_identity_at_init():
    # flow=0 (tanh(0)*max_flow) and correction=0 at init, so grid_sample
    # with an identity sampling grid must reproduce g_t exactly -- "copy
    # the last frame" is a strong single-step baseline, so the model
    # should start there rather than at an arbitrary random delta.
    model = BounceNextFrameModel(channels=16, depth=2)
    g_t = torch.randn(1, 3, 50, 50)
    out = model(g_t)
    assert torch.allclose(out, g_t, atol=1e-5)


def test_nonzero_flow_head_shifts_content():
    model = BounceNextFrameModel(channels=16, depth=2, max_flow=4.0)
    with torch.no_grad():
        model.flow_head.bias[0] = 2.0  # constant +2 px shift in x
    g_t = torch.zeros(1, 3, 20, 20)
    g_t[0, 0, 10, 10] = 1.0  # single bright pixel
    out = model(g_t)
    # the warp samples g_t at (x + flow, y), so a bright pixel at column 10
    # should now be read back from column 10 + shift -> content moves
    assert not torch.allclose(out, g_t, atol=1e-5)
    assert out[0, 0, 10, 10].item() < 0.5


def test_correction_head_adds_local_delta_on_top_of_identity_warp():
    # correction is tanh-bounded by max_correction (see
    # docs/debugging/findings-period2-oscillation.md), so a saturating bias
    # pushes correction toward max_correction, not toward the raw bias value.
    model = BounceNextFrameModel(channels=16, depth=2, max_correction=0.2)
    with torch.no_grad():
        model.correction_head.bias[0] = 10.0
    g_t = torch.zeros(1, 3, 20, 20)
    out = model(g_t)
    assert torch.allclose(out[:, 0], torch.full_like(out[:, 0], 0.2), atol=1e-3)
    assert torch.allclose(out[:, 1:], g_t[:, 1:], atol=1e-5)


def test_correction_head_is_bounded_by_max_correction():
    model = BounceNextFrameModel(channels=16, depth=2, max_correction=0.2)
    with torch.no_grad():
        model.correction_head.bias[:] = 1000.0
    g_t = torch.randn(1, 3, 20, 20)
    out = model(g_t)
    correction = out - g_t
    assert correction.abs().max().item() <= 0.2 + 1e-4


def test_gradients_flow_to_all_params():
    model = BounceNextFrameModel(channels=16, depth=2)
    g_t = torch.randn(1, 3, 20, 20, requires_grad=False)
    target = torch.randn(1, 3, 20, 20)
    out = model(g_t)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
