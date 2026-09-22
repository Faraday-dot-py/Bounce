import argparse
import random

import torch
from torch.utils.data import DataLoader

from model.dataset import BounceSequenceDataset
from model.net import BounceNextFrameModel
from model.losses import occupancy_weighted_mse


def sampling_probability(epoch, ramp_epochs):
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, epoch / ramp_epochs)


def rollout_loss(model, sequence, horizon, sampling_p, weights, bg_weight):
    device = next(model.parameters()).device
    sequence = sequence.to(device)
    weights = weights.to(device)
    frame = sequence[:, 0]
    total = 0.0
    for k in range(1, horizon + 1):
        target = sequence[:, k]
        pred = model(frame)
        total = total + occupancy_weighted_mse(pred, target, frame, weights, bg_weight=bg_weight)
        self_feed = random.random() < sampling_p
        frame = pred.detach() if self_feed else target
    return total / horizon


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BounceSequenceDataset(
        num_samples=args.num_samples,
        n=args.n,
        ball_range=(args.min_balls, args.max_balls),
        seed=args.seed,
        horizon=args.horizon,
        cache_path=args.cache_path,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = BounceNextFrameModel(
        embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads,
        window_size=args.window_size,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = torch.tensor(args.channel_weights, device=device)
    ramp_epochs = args.sampling_ramp_epochs if args.sampling_ramp_epochs is not None else args.epochs

    for epoch in range(args.epochs):
        p = sampling_probability(epoch, ramp_epochs)
        total_loss = 0.0
        for sequence in loader:
            loss = rollout_loss(model, sequence, args.horizon, p, weights, args.bg_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"epoch {epoch} loss {avg:.6f} sampling_p {p:.3f}")
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
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--sampling-ramp-epochs", type=int, default=None)
    ap.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    ap.add_argument("--cache-path", type=str, default=None)
    return ap


if __name__ == "__main__":
    train(build_arg_parser().parse_args())
