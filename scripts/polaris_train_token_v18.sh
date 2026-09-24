#!/bin/bash
#SBATCH --job-name=bounce-token-model-v18
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=bounce-token-model-v18-%j.log

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

# v18: observation-free rollout (TokenModel free_rollout) -- tokens evolve in
# state space after init_tokens, the rendered frame is never read back
# (docs/superpowers/specs/2026-09-23-token-free-rollout-design.md). Unroll
# ramps 4 -> 24 steps over 15 epochs. Primary criteria: mean position error
# and identity swaps vs v9 (scripts/eval_free_rollout.py, 48 seeds).
echo "[$(date -Iseconds)] starting training"
python -m model.token_train \
  --dataset-cache checkpoints/token_dataset_10000_h24_seed4738.pt \
  --n 20 --horizon 24 --epochs 50 \
  --free-rollout --horizon-start 4 --horizon-ramp 15 \
  --hidden-dim 32 --neighbor-radius 4.0 \
  --bg-weight 0.05 --peak-weight 0.5 --boundary-weight 0.1 \
  --state-weight 1.0 --grid-weight 0.1 \
  --lr 1e-3 --ramp-epochs 1 \
  --seed 4738 --log-every 200 \
  --checkpoint checkpoints/token_model_h24_v18.pt
echo "[$(date -Iseconds)] training done"
