import argparse

import torch
from torch.utils.data import DataLoader

from model.dataset import BouncePairDataset
from model.net import BounceNextFrameModel
from model.losses import occupancy_weighted_mse


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BouncePairDataset(
        num_samples=args.num_samples,
        n=args.n,
        ball_range=(args.min_balls, args.max_balls),
        seed=args.seed,
        cache_path=args.cache_path,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = BounceNextFrameModel(
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
        window_size=args.window_size,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = torch.tensor(args.channel_weights, device=device)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for g_t, g_t1 in loader:
            g_t, g_t1 = g_t.to(device), g_t1.to(device)
            pred = model(g_t)
            loss = occupancy_weighted_mse(pred, g_t1, weights, bg_weight=args.bg_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"epoch {epoch} loss {avg:.6f}")
        torch.save(model.state_dict(), args.checkpoint)


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--min-balls", type=int, default=50)
    ap.add_argument("--max-balls", type=int, default=250)
    ap.add_argument("--num-samples", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--window-size", type=int, default=8)
    ap.add_argument("--channel-weights", type=float, nargs=3, default=[1.0, 0.1, 0.1])
    ap.add_argument("--bg-weight", type=float, default=0.05)
    ap.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    ap.add_argument("--cache-path", type=str, default=None)
    return ap


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
