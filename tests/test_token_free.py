import torch

from model.token_free import TokenFreeDynamics, wall_features


def test_wall_features_zero_in_interior_and_saturate_at_wall():
    n, W = 20, 3.0
    pos = torch.tensor([[10.0, 10.0], [0.0, 10.0], [-1.0, 19.0], [19.0, 0.0]])
    f = wall_features(pos, n, W)
    assert f.shape == (4, 4)
    assert torch.all(f[0] == 0)
    assert f[1, 0] == 1.0 and f[1, 1] == 0.0
    assert f[2, 0] == 1.0 and f[2, 1] == 0.0 and f[2, 2] == 0.0 and f[2, 3] == 1.0
    assert f[3, 1] == 1.0 and f[3, 2] == 1.0
    assert torch.isfinite(f).all()


def test_zero_init_is_identity_delta():
    torch.manual_seed(4738)
    dyn = TokenFreeDynamics(n=20)
    pos = torch.rand(3, 2) * 10 + 5
    vel = torch.randn(3, 2)
    hidden = torch.zeros(3, dyn.hidden_dim)
    dp, dv, nh = dyn(pos, vel, hidden)
    assert torch.all(dp == 0) and torch.all(dv == 0)
    assert nh.shape == hidden.shape


def test_interior_translation_invariance():
    torch.manual_seed(4738)
    dyn = TokenFreeDynamics(n=40)
    torch.nn.init.normal_(dyn.delta_head.weight, std=0.1)
    pos = torch.tensor([[15.0, 15.0], [16.5, 15.5], [17.0, 19.0]])
    vel = torch.randn(3, 2)
    hidden = torch.randn(3, dyn.hidden_dim)
    a = dyn(pos, vel, hidden)
    b = dyn(pos + torch.tensor([3.0, -2.0]), vel, hidden)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-6)


def test_wall_proximity_changes_output():
    torch.manual_seed(4738)
    dyn = TokenFreeDynamics(n=20)
    torch.nn.init.normal_(dyn.delta_head.weight, std=0.1)
    vel = torch.zeros(1, 2)
    hidden = torch.zeros(1, dyn.hidden_dim)
    mid = dyn(torch.tensor([[10.0, 10.0]]), vel, hidden)[0]
    wall = dyn(torch.tensor([[0.5, 10.0]]), vel, hidden)[0]
    assert not torch.allclose(mid, wall)


def test_zero_and_one_token_keep_grad_fn():
    dyn = TokenFreeDynamics(n=20)
    for count in (0, 1):
        pos = torch.full((count, 2), 10.0)
        vel = torch.zeros(count, 2)
        hidden = torch.zeros(count, dyn.hidden_dim)
        dp, dv, nh = dyn(pos, vel, hidden)
        assert dp.shape == (count, 2)
        assert (dp.sum() + dv.sum() + nh.sum()).requires_grad
        assert torch.isfinite(nh).all()


def test_mirror_sym_is_y_reflection_equivariant():
    torch.manual_seed(4738)
    n = 20
    dyn = TokenFreeDynamics(n=n, mirror_sym=True)
    torch.nn.init.normal_(dyn.delta_head.weight, std=0.1)
    torch.nn.init.normal_(dyn.delta_head.bias, std=0.1)
    pos = torch.tensor([[5.0, 2.0], [6.0, 3.0], [12.0, 17.0]])
    vel = torch.randn(3, 2)
    hidden = torch.randn(3, dyn.hidden_dim)
    mirrored_pos = torch.stack([pos[:, 0], (n - 1) - pos[:, 1]], dim=1)
    mirrored_vel = vel * torch.tensor([1.0, -1.0])
    swapped = torch.cat([hidden[:, dyn.core_dim:], hidden[:, :dyn.core_dim]], dim=-1)
    dp, dv, nh = dyn(pos, vel, hidden)
    dp_m, dv_m, nh_m = dyn(mirrored_pos, mirrored_vel, swapped)
    flip = torch.tensor([1.0, -1.0])
    assert torch.allclose(dp_m, dp * flip, atol=1e-6)
    assert torch.allclose(dv_m, dv * flip, atol=1e-6)
    assert torch.allclose(nh_m, torch.cat([nh[:, dyn.core_dim:], nh[:, :dyn.core_dim]], dim=-1), atol=1e-6)


def test_mirror_sym_hidden_is_twice_core_and_default_unchanged():
    assert TokenFreeDynamics(n=20, hidden_dim=8, mirror_sym=True).hidden_dim == 16
    assert TokenFreeDynamics(n=20, hidden_dim=8).hidden_dim == 8
