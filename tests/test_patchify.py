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
