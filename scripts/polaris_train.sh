#!/bin/bash
#SBATCH --job-name=bounce-stage1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=bounce-stage1-%j.log

set -euo pipefail
cd "$HOME/bounce"

pip install -r requirements.txt

mkdir -p checkpoints

python -m model.train \
  --n 50 --min-balls 50 --max-balls 250 \
  --num-samples 3000 --batch-size 16 --epochs 20 \
  --embed-dim 128 --depth 6 --num-heads 4 --window-size 8 \
  --seed 4738 \
  --checkpoint checkpoints/stage1.pt
