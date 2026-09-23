#!/bin/bash
#SBATCH --job-name=bounce-token-model-v16
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=bounce-token-model-v16-%j.log

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

# Curiosity variant, not part of the territory-masking line: v9's recipe
# (unmasked, best checkpoint so far) with gravity=0 and a fixed 15 balls
# instead of the usual 2-6, same n=20 grid. No --dataset-cache since the
# cached dataset was generated at gravity=9 with 2-6 balls -- this needs
# a fresh generation to match. Time limit raised to 3h vs the usual 1h:
# job 2862 (killed) showed 15-ball batches run much slower than the
# usual 2-6 ball recipe (heavier radius-graph attention/rasterization).
echo "[$(date -Iseconds)] starting training"
python -m model.token_train \
  --n 20 --horizon 12 --epochs 3 \
  --num-samples 10000 --min-balls 15 --max-balls 15 --gravity 0.0 \
  --hidden-dim 32 --neighbor-radius 4.0 \
  --bg-weight 0.05 --peak-weight 0.5 --boundary-weight 0.0 \
  --state-weight 1.0 --grid-weight 0.0 \
  --lr 1e-3 --ramp-epochs 2 \
  --seed 4738 --log-every 200 \
  --checkpoint checkpoints/token_model_h12_v16.pt
echo "[$(date -Iseconds)] training done"
