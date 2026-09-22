#!/bin/bash
#SBATCH --job-name=bounce-stage2-h12
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:30:00
#SBATCH --output=bounce-stage2-h12-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"

echo "[$(date -Iseconds)] starting pip install"
pip install -r requirements.txt
echo "[$(date -Iseconds)] pip install done"

mkdir -p checkpoints

echo "[$(date -Iseconds)] starting training"
python -m model.train \
  --n 50 --min-balls 50 --max-balls 250 \
  --num-samples 1500 --batch-size 16 --epochs 80 \
  --embed-dim 128 --depth 6 --num-heads 4 --window-size 8 \
  --horizon 12 \
  --seed 4738 \
  --checkpoint checkpoints/stage2_h12.pt \
  --cache-path checkpoints/dataset_cache_seq_h12.npz
echo "[$(date -Iseconds)] training done"
