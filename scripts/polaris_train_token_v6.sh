#!/bin/bash
#SBATCH --job-name=bounce-token-model-v6
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=bounce-token-model-v6-%j.log

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

echo "[$(date -Iseconds)] starting training"
python -m model.token_train \
  --dataset-cache checkpoints/token_dataset_10000_h12_seed4738.pt \
  --n 20 --horizon 12 --epochs 3 \
  --hidden-dim 32 --neighbor-radius 4.0 \
  --bg-weight 0.05 --peak-weight 0.5 \
  --lr 1e-3 --ramp-epochs 2 \
  --seed 4738 --log-every 200 \
  --checkpoint checkpoints/token_model_h12_v6.pt
echo "[$(date -Iseconds)] training done"
