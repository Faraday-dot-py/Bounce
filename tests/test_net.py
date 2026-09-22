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
    # with an identity sampling grid must reproduce g_t exactly on VX/VY --
    # "copy the last frame" is a strong single-step baseline, so the model
    # should start there rather than at an arbitrary random delta. PROB
    # goes through the soft-threshold + renorm round trip (see
    # docs/debugging/findings-quilting-artifact.md) which is a deliberate
    # shrink-and-rescale, not an identity map, so it's checked separately:
    # background (exactly 0) stays exactly 0, and total mass is conserved.
    model = BounceNextFrameModel(channels=16, depth=2)
    g_t = torch.zeros(1, 3, 50, 50)
    g_t[0, 0, 10:15, 10:15] = 0.8
    g_t[:, 1:] = torch.randn(1, 2, 50, 50)
    out = model(g_t)
    assert torch.allclose(out[:, 1:], g_t[:, 1:], atol=1e-5)
    background = out[0, 0].clone()
    background[10:15, 10:15] = 0
    assert torch.allclose(background, torch.zeros_like(background), atol=1e-5)
    assert torch.allclose(out[:, 0].sum(), g_t[:, 0].sum(), atol=1e-3)


def test_prob_below_threshold_is_zeroed():
    # Diffusion from repeated resampling spreads PROB into near-zero noise
    # across nearly the whole grid; soft-thresholding before renorm should
    # zero it out rather than let exact mass renormalization amplify it
    # back up (see docs/debugging/findings-quilting-artifact.md).
    model = BounceNextFrameModel(channels=16, depth=2)
    g_t = torch.zeros(1, 3, 20, 20)
    g_t[0, 0, 5, 5] = 1.0
    g_t[0, 0] += 0.005  # below prob_threshold=0.015 everywhere
    out = model(g_t)
    below_thresh_mask = torch.ones(20, 20, dtype=torch.bool)
    below_thresh_mask[5, 5] = False
    assert torch.all(out[0, 0][below_thresh_mask] == 0.0)


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


def test_uniform_correction_bias_is_centered_away():
    # correction_head's raw output is uniform in space when its conv weight
    # is zero (only bias contributes), and correction is re-centered per
    # channel per step (its own spatial mean subtracted, see
    # docs/debugging/findings-correction-drift-and-mass-dissolution.md) so
    # it can only redistribute mass, not inject a sustained per-channel
    # bias. A purely uniform correction should therefore cancel to exactly
    # zero, leaving the identity warp untouched.
    model = BounceNextFrameModel(channels=16, depth=2, max_correction=0.2)
    with torch.no_grad():
        model.correction_head.bias[0] = 10.0
    g_t = torch.zeros(1, 3, 20, 20)
    out = model(g_t)
    assert torch.allclose(out, g_t, atol=1e-4)


def test_correction_is_centered_and_bounded_per_channel():
    # A spatially-varying raw correction (conv weight nonzero) should still
    # be able to redistribute mass locally, but the *spatial mean* per
    # channel must land at ~0 (centering), and no single cell can exceed
    # 2*max_correction (the full tanh saturation range after centering).
    model = BounceNextFrameModel(channels=16, depth=2, max_correction=0.2)
    g_t = torch.randn(1, 3, 20, 20)
    raw = torch.randn(1, 3, 20, 20) * 50  # saturates tanh, varies spatially
    model.correction_head.forward = lambda x: raw
    out = model(g_t)
    # flow_head is zero-init, so flow=0 and bicubic grid_sample at an
    # identity grid reproduces g_t exactly -> out - g_t is the correction.
    # Channel 0 (PROB) goes through the soft-threshold + renorm round trip
    # afterwards (see docs/debugging/findings-quilting-artifact.md) so it's
    # no longer bounded by max_correction alone -- only VX/VY (channels
    # 1-2) bypass that and keep the raw centered/bounded correction.
    correction = out - g_t
    assert torch.allclose(correction[:, 1:].mean(dim=(2, 3)), torch.zeros(1, 2), atol=1e-3)
    assert correction[:, 1:].abs().max().item() <= 2 * model.max_correction + 1e-3


def test_gradients_flow_to_all_params():
    model = BounceNextFrameModel(channels=16, depth=2)
    g_t = torch.randn(1, 3, 20, 20, requires_grad=False)
    target = torch.randn(1, 3, 20, 20)
    out = model(g_t)
    loss = ((out - target) ** 2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
