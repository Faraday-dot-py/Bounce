import torch
from torch.utils.data import DataLoader

from model.dataset import BouncePairDataset
from model.net import BounceNextFrameModel
from model.losses import weighted_channel_mse


def test_training_loop_overfits_a_single_batch():
    torch.manual_seed(4738)
    dataset = BouncePairDataset(num_samples=4, n=20, ball_range=(3, 6), seed=4738)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    g_t, g_t1 = next(iter(loader))

    model = BounceNextFrameModel(embed_dim=16, depth=2, num_heads=4, window_size=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    weights = torch.tensor([1.0, 0.1, 0.1])

    first_loss = None
    last_loss = None
    for step in range(50):
        pred = model(g_t)
        loss = weighted_channel_mse(pred, g_t1, weights)
        if step == 0:
            first_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = loss.item()

    assert last_loss < first_loss * 0.5
