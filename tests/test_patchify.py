import torch
from model.patchify import PatchEmbed, Unpatchify


def test_patch_embed_shape():
    embed = PatchEmbed(in_channels=3, patch_size=2, embed_dim=16)
    x = torch.randn(2, 3, 50, 50)
    out = embed(x)
    assert out.shape == (2, 25, 25, 16)


def test_unpatchify_shape():
    unpatch = Unpatchify(embed_dim=16, patch_size=2, out_channels=3)
    x = torch.randn(2, 25, 25, 16)
    out = unpatch(x)
    assert out.shape == (2, 3, 50, 50)


def test_patch_embed_resolution_independent():
    embed = PatchEmbed(in_channels=3, patch_size=2, embed_dim=16)
    for n in (50, 300):
        x = torch.randn(1, 3, n, n)
        out = embed(x)
        assert out.shape == (1, n // 2, n // 2, 16)


def test_icnr_init_sub_pixel_positions_start_equal():
    torch.manual_seed(4738)
    unpatch = Unpatchify(embed_dim=16, patch_size=2, out_channels=3)
    w = unpatch.proj.weight.data  # (12, 16)
    group_size = 4  # patch_size * patch_size
    out_channels = 3
    w_grouped = w.view(group_size, out_channels, -1)
    for c in range(out_channels):
        for g in range(1, group_size):
            assert torch.allclose(w_grouped[0, c], w_grouped[g, c])


def test_icnr_init_diverges_after_optimizer_step():
    torch.manual_seed(4738)
    unpatch = Unpatchify(embed_dim=16, patch_size=2, out_channels=3)
    opt = torch.optim.Adam(unpatch.parameters(), lr=0.1)
    x = torch.randn(2, 5, 5, 16)
    target = torch.randn(2, 3, 10, 10)
    out = unpatch(x)
    loss = ((out - target) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    w_grouped = unpatch.proj.weight.data.view(4, 3, -1)
    assert not torch.allclose(w_grouped[0, 0], w_grouped[1, 0])
