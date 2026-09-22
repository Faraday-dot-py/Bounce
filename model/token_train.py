import argparse
import random

import torch
from torch.utils.data import DataLoader

from model.token_dataset import BounceTokenSequenceDataset
from model.token_model import TokenModel
from model.token_losses import token_grid_loss


def sampling_probability(epoch, ramp_epochs):
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, epoch / ramp_epochs))


def token_rollout_loss(model, grid_seq, horizon, sampling_p, weights):
    """Teacher-forced/self-feed rollout loss for one sequence sample.
    Mirrors model.train.rollout_loss's per-step self-feed coin flip
    (re-drawn every step, not once per rollout, per
    tests/test_train.py's regression test for the same reason). Token
    state (positions/velocities/hidden) always carries forward through
    the recurrence regardless of self-feed; only the *observed grid* fed
    into the observation branch is swapped for the model's own (detached)
    prediction on a self-fed step."""
    positions, velocities, hidden = model.init_tokens(grid_seq[0], grid_seq[1])
    observed_frame = grid_seq[1]
    total_loss = grid_seq.new_zeros(())
    num_steps = max(horizon - 1, 1)
    for step in range(num_steps):
        positions, velocities, hidden, pred_grid = model.step(
            positions, velocities, hidden, observed_frame
        )
        target_grid = grid_seq[step + 2]
        total_loss = total_loss + token_grid_loss(pred_grid, target_grid, weights)
        self_feed = random.random() < sampling_p
        observed_frame = pred_grid.detach() if self_feed else target_grid
    return total_loss / num_steps


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dataset = BounceTokenSequenceDataset(
        num_samples=args.num_samples, n=args.n, ball_range=(args.min_balls, args.max_balls),
        seed=args.seed, horizon=args.horizon,
    )
    loader = DataLoader(dataset, batch_size=None, shuffle=True)
    model = TokenModel(n=args.n, radius=0.75, dt=0.15, hidden_dim=args.hidden_dim,
                        neighbor_radius=args.neighbor_radius)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = torch.tensor([1.0, 0.1, 0.1])

    for epoch in range(args.epochs):
        sampling_p = sampling_probability(epoch, args.ramp_epochs)
        epoch_loss = 0.0
        for grid_seq, _ in loader:
            loss = token_rollout_loss(model, grid_seq, args.horizon, sampling_p, weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f"epoch {epoch} sampling_p={sampling_p:.2f} loss={epoch_loss / len(dataset):.4f}", flush=True)
        torch.save(model.state_dict(), args.checkpoint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min-balls", type=int, default=2)
    ap.add_argument("--max-balls", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--ramp-epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--neighbor-radius", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--checkpoint", type=str, default="checkpoint_token.pt")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
