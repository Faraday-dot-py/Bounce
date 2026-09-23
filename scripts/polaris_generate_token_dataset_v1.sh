#!/bin/bash
#SBATCH --job-name=bounce-token-dataset-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=bounce-token-dataset-v1-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"

echo "[$(date -Iseconds)] starting pip install"
pip install -r requirements.txt
echo "[$(date -Iseconds)] pip install done"

mkdir -p checkpoints

echo "[$(date -Iseconds)] generating dataset"
PYTHONPATH=. python scripts/generate_token_dataset.py \
  --num-samples 10000 --n 20 --min-balls 2 --max-balls 6 \
  --horizon 12 --seed 4738 --log-every 200 \
  --out checkpoints/token_dataset_10000_h12_seed4738.pt
echo "[$(date -Iseconds)] dataset generation done"
