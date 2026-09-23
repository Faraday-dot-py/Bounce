import argparse
import random

import torch
from torch.utils.data import DataLoader

from model.token_dataset import BounceTokenSequenceDataset, load_dataset_samples
from model.token_match import match_tokens_to_state
from model.token_model import TokenModel
from model.token_losses import boundary_loss, token_grid_loss, token_state_loss


def sampling_probability(epoch, ramp_epochs):
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, epoch / ramp_epochs))


def token_rollout_loss(model, grid_seq, state_seq, horizon, sampling_p, weights,
                        bg_weight=0.05, peak_weight=0.1, mass_weight=0.0, mass_tile=16,
                        boundary_weight=0.1, boundary_margin=0.0,
                        state_weight=1.0, grid_weight=0.1):
    """Teacher-forced/self-feed rollout loss for one sequence sample.
    Mirrors model.train.rollout_loss's per-step self-feed coin flip
    (re-drawn every step, not once per rollout, per
    tests/test_train.py's regression test for the same reason). Token
    state (positions/velocities/hidden) always carries forward through
    the recurrence regardless of self-feed; only the *observed grid* fed
    into the observation branch is swapped for the model's own (detached)
    prediction on a self-fed step.

    Primary signal is `token_state_loss` (model/token_losses.py): direct
    MSE against the dataset's exact per-ball state, matched once via
    match_tokens_to_state right after init (token identity is stable for
    the rest of the rollout -- see model/token_match.py). This has no
    give-up degenerate solution: there's no background cell to hide
    behind, so vanishing/stalling aren't cheap regardless of loss
    weighting, unlike the grid-space losses below.

    `token_grid_loss` (occupancy-weighted, not plain per-pixel MSE --
    job 2840 showed plain MSE converges to an all-background give-up
    solution) is kept as a smaller secondary term so the actual
    rasterized output stays directly optimized too, ramped in from 0
    over the same sampling_p schedule as self-feed (`grid_weight` is the
    weight sampling_p=1.0 reaches, not the weight used at sampling_p=0).

    `boundary_loss` operates on tracked positions directly, at every
    rollout step, not just the rasterized output -- added because job
    2847 (peak_weight=0.5 alone) still showed tokens drifting off-grid
    and dissolving by step ~12-20."""
    device = next(model.parameters()).device
    grid_seq = grid_seq.to(device)
    weights = weights.to(device)
    state_seq = [{k: v.to(device) for k, v in frame.items()} for frame in state_seq]
    positions, velocities, hidden = model.init_tokens(grid_seq[0], grid_seq[1])
    match_idx = match_tokens_to_state(positions, state_seq[1])
    observed_frame = grid_seq[1]
    total_loss = grid_seq.new_zeros(())
    num_steps = max(horizon - 1, 1)
    grid_weight_effective = grid_weight * sampling_p
    for step in range(num_steps):
        source_grid = observed_frame
        positions, velocities, hidden, pred_grid = model.step(
            positions, velocities, hidden, observed_frame
        )
        target_grid = grid_seq[step + 2]
        target_state = state_seq[step + 2]
        total_loss = total_loss + state_weight * token_state_loss(
            positions, velocities, target_state, match_idx
        )
        total_loss = total_loss + grid_weight_effective * token_grid_loss(
            pred_grid, target_grid, source_grid, weights,
            bg_weight=bg_weight, peak_weight=peak_weight,
            mass_weight=mass_weight, mass_tile=mass_tile,
        )
        total_loss = total_loss + boundary_weight * boundary_loss(
            positions, model.n, margin=boundary_margin
        )
        self_feed = random.random() < sampling_p
        observed_frame = pred_grid.detach() if self_feed else target_grid
    return total_loss / num_steps


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.dataset_cache:
        print(f"loading cached dataset from {args.dataset_cache}", flush=True)
        dataset = BounceTokenSequenceDataset.from_samples(load_dataset_samples(args.dataset_cache))
        print(f"loaded {len(dataset)} samples", flush=True)
    else:
        dataset = BounceTokenSequenceDataset(
            num_samples=args.num_samples, n=args.n, ball_range=(args.min_balls, args.max_balls),
            seed=args.seed, horizon=args.horizon,
        )
    loader = DataLoader(dataset, batch_size=None, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TokenModel(n=args.n, radius=0.75, dt=0.15, hidden_dim=args.hidden_dim,
                        neighbor_radius=args.neighbor_radius).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = torch.tensor([1.0, 0.1, 0.1], device=device)

    for epoch in range(args.epochs):
        sampling_p = sampling_probability(epoch, args.ramp_epochs)
        epoch_loss = 0.0
        running_loss = 0.0
        for batch_idx, (grid_seq, state_seq) in enumerate(loader):
            loss = token_rollout_loss(
                model, grid_seq, state_seq, args.horizon, sampling_p, weights,
                bg_weight=args.bg_weight, peak_weight=args.peak_weight,
                boundary_weight=args.boundary_weight, boundary_margin=args.boundary_margin,
                state_weight=args.state_weight, grid_weight=args.grid_weight,
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item()
            running_loss += loss.item()
            if args.log_every > 0 and (batch_idx + 1) % args.log_every == 0:
                print(f"epoch {epoch} batch {batch_idx + 1}/{len(dataset)} "
                      f"sampling_p={sampling_p:.2f} loss={running_loss / args.log_every:.4f}", flush=True)
                running_loss = 0.0
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
    # Must stay above TokenModel's occlusion-gate threshold
    # (2*(radius+detect_margin) = 3.5 at defaults), or every token with a
    # graph neighbor is also gated off from observation correction (see
    # docs/debugging/experiment-log.md, job 2840).
    ap.add_argument("--neighbor-radius", type=float, default=4.0)
    ap.add_argument("--bg-weight", type=float, default=0.05)
    ap.add_argument("--peak-weight", type=float, default=0.1)
    ap.add_argument("--boundary-weight", type=float, default=0.1)
    ap.add_argument("--boundary-margin", type=float, default=0.0)
    ap.add_argument("--state-weight", type=float, default=1.0)
    # Weight token_grid_loss ramps to (from 0) over the same sampling_p
    # schedule as self-feed -- see token_rollout_loss's docstring.
    ap.add_argument("--grid-weight", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--checkpoint", type=str, default="checkpoint_token.pt")
    ap.add_argument("--log-every", type=int, default=100,
                     help="print running-avg loss every N batches (0 disables)")
    ap.add_argument("--dataset-cache", type=str, default=None,
                     help="path to a dataset saved by scripts/generate_token_dataset.py; "
                          "if set, loads instead of generating on the fly")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
