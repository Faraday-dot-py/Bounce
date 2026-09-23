#!/bin/bash
#SBATCH --job-name=bounce-token-model-v12
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=bounce-token-model-v12-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"

echo "[$(date -Iseconds)] starting pip install"
pip install -r requirements.txt
echo "[$(date -Iseconds)] pip install done"

mkdir -p checkpoints

echo "[$(date -Iseconds)] running tests before real training run"
python -m pytest tests/test_token_*.py -q
echo "[$(date -Iseconds)] tests passed"

# Single-variable A/B vs v9 (docs/debugging/experiment-log.md,
# 2026-09-23 entries): identical recipe (state-loss-only ablation, 3
# epochs, 10k cache, velocity_weight=0.0 -- the trained-in explicit
# velocity correction was already tested in v10 and did NOT help).
# ONLY change here: token_state_loss's vel_weight 0.1 -> 1.0
# (--state-vel-weight), i.e. weight position and velocity error equally
# in the direct state-space loss instead of down-weighting velocity 10x.
# Motivated directly by measurement, not blind tuning:
# scripts/measure_error_compounding.py showed self-fed velocity error
# growing ~5x faster than position error under v9, but the loss term
# that's supposed to teach accurate velocity tracking barely penalizes
# getting it wrong at the default weight.
echo "[$(date -Iseconds)] starting training"
python -m model.token_train \
  --dataset-cache checkpoints/token_dataset_10000_h12_seed4738.pt \
  --n 20 --horizon 12 --epochs 3 \
  --hidden-dim 32 --neighbor-radius 4.0 \
  --bg-weight 0.05 --peak-weight 0.5 --boundary-weight 0.0 \
  --state-weight 1.0 --grid-weight 0.0 \
  --state-vel-weight 1.0 \
  --lr 1e-3 --ramp-epochs 2 \
  --seed 4738 --log-every 200 \
  --checkpoint checkpoints/token_model_h12_v12.pt
echo "[$(date -Iseconds)] training done"
