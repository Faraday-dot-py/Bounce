#!/bin/bash
#SBATCH --job-name=bounce-token-model-v15
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=bounce-token-model-v15-%j.log

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

# v15: identical recipe to v9/v14 (state-loss-only ablation) plus
# --territory-masking, now that centroid_near falls back to an unmasked
# search when a token's own territory is completely empty (see
# docs/debugging/experiment-log.md's v14 final-review entry, commits
# e72f790/f7076b2). v14 was trained under the fallback-less version of
# masking and is not representative of the fixed design -- this is the
# real single-variable test of territory masking against the v9 baseline.
echo "[$(date -Iseconds)] starting training"
python -m model.token_train \
  --dataset-cache checkpoints/token_dataset_10000_h12_seed4738.pt \
  --n 20 --horizon 12 --epochs 3 \
  --hidden-dim 32 --neighbor-radius 4.0 \
  --bg-weight 0.05 --peak-weight 0.5 --boundary-weight 0.0 \
  --state-weight 1.0 --grid-weight 0.0 \
  --territory-masking \
  --lr 1e-3 --ramp-epochs 2 \
  --seed 4738 --log-every 200 \
  --checkpoint checkpoints/token_model_h12_v15.pt
echo "[$(date -Iseconds)] training done"
