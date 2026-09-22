#!/bin/bash
#SBATCH --job-name=bounce-stage2-flownet-v4
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=bounce-stage2-flownet-v4-%j.log

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
  --channels 64 --depth 7 --max-flow 4.0 \
  --horizon 12 \
  --seed 4738 \
  --checkpoint checkpoints/stage2_flownet_h12_v4.pt \
  --cache-path checkpoints/dataset_cache_seq_h12.npz
echo "[$(date -Iseconds)] training done"
