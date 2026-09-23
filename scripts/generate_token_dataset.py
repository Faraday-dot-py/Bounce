"""Generate and cache a BounceTokenSequenceDataset to disk, so a training
job can load it instead of paying the physics-sim generation cost on
every run.

Usage:
    python -m scripts.generate_token_dataset \
        --num-samples 10000 --horizon 12 --seed 4738 \
        --out checkpoints/token_dataset_10000_h12_seed4738.pt
"""
import argparse

from model.token_dataset import generate_dataset_samples, save_dataset_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-samples", type=int, default=10000)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--min-balls", type=int, default=2)
    ap.add_argument("--max-balls", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--seed", type=int, default=4738)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    samples = generate_dataset_samples(
        args.num_samples, args.n, (args.min_balls, args.max_balls), args.seed,
        horizon=args.horizon, log_every=args.log_every,
    )
    save_dataset_samples(samples, args.out)
    print(f"saved {len(samples)} samples to {args.out}", flush=True)


if __name__ == "__main__":
    main()
