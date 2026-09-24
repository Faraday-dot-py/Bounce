import argparse
import random

import torch
from torch.utils.data import DataLoader

from model.token_dataset import BounceTokenSequenceDataset, load_dataset_samples
from model.token_match import match_tokens_to_state
from model.token_model import TokenModel
from model.token_losses import boundary_loss, token_grid_loss, token_state_loss, window_collapse_loss


def sampling_probability(epoch, ramp_epochs):
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, max(0.0, epoch / ramp_epochs))


def horizon_for_epoch(epoch, ramp_epochs, start, full):
    if ramp_epochs <= 0:
        return full
    frac = min(1.0, epoch / ramp_epochs)
    return int(round(start + frac * (full - start)))


def curriculum_advance(chunk_losses, tol, max_chunks):
    """Whether the stepwise curriculum should move to the next unroll
    length: the last chunk improved on the one before it by less than
    `tol` (relative), or the stage has used `max_chunks` chunks."""
    if len(chunk_losses) >= max_chunks:
        return True
    if len(chunk_losses) < 2:
        return False
    prev, last = chunk_losses[-2], chunk_losses[-1]
    return (prev - last) < tol * prev


def curriculum_steps(stage_steps, replay_p, rng):
    """Unroll length for one batch: the current stage's, or with
    probability `replay_p` a uniformly drawn shorter/equal one so earlier
    stages keep being exercised."""
    if stage_steps > 1 and rng.random() < replay_p:
        return rng.randint(1, stage_steps)
    return stage_steps


def token_rollout_loss(model, grid_seq, state_seq, horizon, sampling_p, weights,
                        bg_weight=0.05, peak_weight=0.1, mass_weight=0.0, mass_tile=16,
                        boundary_weight=0.1, boundary_margin=0.0,
                        state_weight=1.0, grid_weight=0.1, state_vel_weight=0.1,
                        collapse_weight=0.0, collapse_floor=0.3, collapse_margin=1.0,
                        free=False, max_steps=None, speed_weight=0.0):
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
    and dissolving by step ~12-20.

    `window_collapse_loss` (0 weight by default, off) is the v13 A/B
    against v9-v12's synthesis in docs/debugging/experiment-log.md: none
    of the existing terms above ever look at whether a token's own
    rasterized PROB mass near its tracked position is actually
    detectable, so give-up dropout costs nothing beyond whatever
    state-space error it happens to cause. Reads `pred_grid` (this
    step's own prediction), not `observed_frame`/`source_grid`, so
    gradient flows through the same forward pass rather than a possibly
    self-fed-and-detached observation."""
    device = next(model.parameters()).device
    grid_seq = grid_seq.to(device)
    weights = weights.to(device)
    state_seq = [{k: v.to(device) for k, v in frame.items()} for frame in state_seq]
    positions, velocities, hidden = model.init_tokens(grid_seq[0], grid_seq[1])
    match_idx = match_tokens_to_state(positions, state_seq[1])
    observed_frame = grid_seq[1]
    prev_obs_pos = positions
    total_loss = grid_seq.new_zeros(())
    num_steps = max(horizon - 1, 1)
    if max_steps is not None:
        num_steps = min(num_steps, max_steps)
    grid_weight_effective = grid_weight * (1.0 if free else sampling_p)
    for step in range(num_steps):
        if free:
            source_grid = grid_seq[step + 1]
            positions, velocities, hidden, pred_grid = model.step_free(positions, velocities, hidden)
        else:
            source_grid = observed_frame
            positions, velocities, hidden, pred_grid, prev_obs_pos = model.step(
                positions, velocities, hidden, observed_frame, prev_obs_pos
            )
        target_grid = grid_seq[step + 2]
        target_state = state_seq[step + 2]
        total_loss = total_loss + state_weight * token_state_loss(
            positions, velocities, target_state, match_idx, vel_weight=state_vel_weight,
            speed_weight=speed_weight,
        )
        total_loss = total_loss + grid_weight_effective * token_grid_loss(
            pred_grid, target_grid, source_grid, weights,
            bg_weight=bg_weight, peak_weight=peak_weight,
            mass_weight=mass_weight, mass_tile=mass_tile,
        )
        total_loss = total_loss + boundary_weight * boundary_loss(
            positions, model.n, margin=boundary_margin
        )
        if collapse_weight > 0.0:
            total_loss = total_loss + collapse_weight * window_collapse_loss(
                pred_grid[0], positions, model.radius, margin=collapse_margin, floor=collapse_floor,
                all_positions=positions, self_idx_offset=0,
            )
        if not free:
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
            seed=args.seed, horizon=args.horizon, gravity=args.gravity,
        )
    loader = DataLoader(dataset, batch_size=None, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TokenModel(n=args.n, radius=0.75, dt=0.15, hidden_dim=args.hidden_dim,
                        neighbor_radius=args.neighbor_radius,
                        velocity_weight=args.velocity_weight,
                        territory_masking=args.territory_masking,
                        track_query=args.track_query,
                        free_rollout=args.free_rollout,
                        mirror_sym=args.mirror_sym,
                        velocity_readout=args.velocity_readout,
                        wall_lookahead=args.wall_lookahead, wall_head=args.wall_head,
                        pair_impulse=args.pair_impulse,
                        position_refine=args.position_refine).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    weights = torch.tensor([1.0, 0.1, 0.1], device=device)

    if args.stepwise_curriculum:
        train_stepwise(args, model, opt, loader, weights)
        return

    for epoch in range(args.epochs):
        sampling_p = sampling_probability(epoch, args.ramp_epochs)
        epoch_horizon = horizon_for_epoch(epoch, args.horizon_ramp, args.horizon_start, args.horizon)
        epoch_loss = 0.0
        running_loss = 0.0
        for batch_idx, (grid_seq, state_seq) in enumerate(loader):
            loss = token_rollout_loss(
                model, grid_seq, state_seq, args.horizon, sampling_p, weights,
                bg_weight=args.bg_weight, peak_weight=args.peak_weight,
                boundary_weight=args.boundary_weight, boundary_margin=args.boundary_margin,
                state_weight=args.state_weight, grid_weight=args.grid_weight,
                state_vel_weight=args.state_vel_weight,
                collapse_weight=args.collapse_weight, collapse_floor=args.collapse_floor,
                free=args.free_rollout,
                max_steps=epoch_horizon - 1 if args.free_rollout else None,
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
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch},
                   args.checkpoint + ".full")


def train_stepwise(args, model, opt, loader, weights):
    """Free-rollout curriculum: train 1 step out, advance to 2 once the
    loss plateaus (see curriculum_advance), and so on up to
    --curriculum-max-steps. A chunk is --stage-batches batches; only
    batches run at the stage's own length count toward the plateau test,
    replayed shorter batches (--replay-p) just keep training."""
    rng = random.Random(args.seed)
    batches = iter(loader)
    total_batches = 0
    for stage_steps in range(1, args.curriculum_max_steps + 1):
        chunk_losses = []
        while True:
            chunk_sum, chunk_n = 0.0, 0
            for _ in range(args.stage_batches):
                try:
                    grid_seq, state_seq = next(batches)
                except StopIteration:
                    batches = iter(loader)
                    grid_seq, state_seq = next(batches)
                steps = curriculum_steps(stage_steps, args.replay_p, rng)
                loss = token_rollout_loss(
                    model, grid_seq, state_seq, args.horizon, 1.0, weights,
                    bg_weight=args.bg_weight, peak_weight=args.peak_weight,
                    boundary_weight=args.boundary_weight, boundary_margin=args.boundary_margin,
                    state_weight=args.state_weight, grid_weight=args.grid_weight,
                    state_vel_weight=args.state_vel_weight,
                    free=True, max_steps=steps, speed_weight=args.speed_weight,
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                total_batches += 1
                if steps == stage_steps:
                    chunk_sum += loss.item()
                    chunk_n += 1
            chunk_losses.append(chunk_sum / max(chunk_n, 1))
            print(f"stage {stage_steps} chunk {len(chunk_losses)} batches={total_batches} "
                  f"loss={chunk_losses[-1]:.4f}", flush=True)
            torch.save(model.state_dict(), args.checkpoint)
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "stage": stage_steps},
                       args.checkpoint + ".full")
            if curriculum_advance(chunk_losses, args.stage_tol, args.stage_max_chunks):
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=2000)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min-balls", type=int, default=2)
    ap.add_argument("--max-balls", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--gravity", type=float, default=9.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--ramp-epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=32)
    # Must stay above TokenModel's occlusion-gate threshold
    # (2*(radius+detect_margin) = 3.5 at defaults), or every token with a
    # graph neighbor is also gated off from observation correction (see
    # docs/debugging/experiment-log.md, job 2840).
    ap.add_argument("--neighbor-radius", type=float, default=4.0)
    # Explicit velocity re-anchoring from consecutive observation reads
    # (see model/token_model.py TokenModel.__init__'s docstring); 0.0
    # keeps existing behavior unchanged.
    ap.add_argument("--velocity-weight", type=float, default=0.0)
    # Opt-in per-token observation-window exclusivity (see
    # model/token_model.py TokenModel.__init__'s docstring). Off by
    # default so existing recipes/checkpoints are byte-for-byte
    # unaffected -- see docs/debugging/experiment-log.md's v14 final
    # review for why the default must stay off.
    ap.add_argument("--territory-masking", action="store_true")
    # Opt-in architecture swap (see
    # docs/superpowers/specs/2026-09-23-token-track-query-design.md):
    # replaces TokenModel's hard occlusion gate + centroid_near blend
    # with learned self-/cross-attention. Off by default so existing
    # recipes/checkpoints are byte-for-byte unaffected.
    ap.add_argument("--track-query", action="store_true")
    ap.add_argument("--free-rollout", action="store_true",
                     help="observation-free rollout (TokenModel.step_free); see "
                          "docs/superpowers/specs/2026-09-23-token-free-rollout-design.md")
    ap.add_argument("--horizon-ramp", type=int, default=0,
                     help="epochs over which the free-rollout unroll grows from --horizon-start "
                          "to --horizon (0 = always full)")
    ap.add_argument("--horizon-start", type=int, default=4)
    ap.add_argument("--stepwise-curriculum", action="store_true",
                     help="free-rollout curriculum 1..--curriculum-max-steps steps out, advancing "
                          "on loss plateau; overrides --epochs/--horizon-ramp")
    ap.add_argument("--curriculum-max-steps", type=int, default=20)
    ap.add_argument("--stage-batches", type=int, default=500)
    ap.add_argument("--stage-tol", type=float, default=0.03)
    ap.add_argument("--stage-max-chunks", type=int, default=4)
    ap.add_argument("--replay-p", type=float, default=0.25)
    ap.add_argument("--wall-lookahead", action="store_true",
                     help="add wall penetration + one-step-lookahead penetration features (TokenFreeDynamics)")
    ap.add_argument("--position-refine", action="store_true",
                     help="sub-cell init position refinement (needs --velocity-readout)")
    ap.add_argument("--pair-impulse", action="store_true",
                     help="explicit antisymmetric sum-aggregated pair impulse (TokenFreeDynamics)")
    ap.add_argument("--wall-head", action="store_true",
                     help="separate two-layer wall-impulse head (TokenFreeDynamics)")
    ap.add_argument("--velocity-readout", action="store_true",
                     help="init velocity from the frame VX/VY channels (TokenModel velocity_readout)")
    ap.add_argument("--mirror-sym", action="store_true",
                     help="y-reflection-symmetrized free-rollout dynamics (TokenFreeDynamics mirror_sym)")
    ap.add_argument("--bg-weight", type=float, default=0.05)
    ap.add_argument("--peak-weight", type=float, default=0.1)
    ap.add_argument("--boundary-weight", type=float, default=0.1)
    ap.add_argument("--boundary-margin", type=float, default=0.0)
    ap.add_argument("--state-weight", type=float, default=1.0)
    # token_state_loss's own vel_weight (model/token_losses.py) -- how much
    # the direct state-space loss penalizes velocity error relative to
    # position. Raised from the project default (0.1) is a direct,
    # measurement-motivated test (not blind tuning): 2026-09-23's
    # scripts/measure_error_compounding.py showed self-fed VELOCITY error
    # growing ~5x faster than position error and driving it, but the loss
    # that's supposed to teach the network to track velocity barely
    # penalizes getting it wrong.
    ap.add_argument("--state-vel-weight", type=float, default=0.1)
    ap.add_argument("--speed-weight", type=float, default=0.0,
                     help="weight of the |v| magnitude term in token_state_loss (stepwise curriculum only)")
    # Weight token_grid_loss ramps to (from 0) over the same sampling_p
    # schedule as self-feed -- see token_rollout_loss's docstring.
    ap.add_argument("--grid-weight", type=float, default=0.1)
    # window_collapse_loss weight -- 0.0 (off) keeps existing behavior
    # unchanged. See token_rollout_loss's docstring and
    # docs/debugging/experiment-log.md's v9-v12 synthesis for why this
    # exists: v13's A/B test of the "give-up is free" hypothesis.
    ap.add_argument("--collapse-weight", type=float, default=0.0)
    ap.add_argument("--collapse-floor", type=float, default=0.3)
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
