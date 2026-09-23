#!/bin/bash
#SBATCH --job-name=bounce-token-model-v13
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=bounce-token-model-v13-%j.log

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

# v13: identical recipe to v9 (state-loss-only ablation, same cache/seed)
# plus window_collapse_loss (model/token_losses.py), the untested
# direction flagged in docs/debugging/experiment-log.md's v9-v12
# synthesis after three single-variable reweights of the EXISTING loss
# terms (v10 velocity re-anchoring, v11 5x training, v12 vel_weight
# 0.1->1.0) all came back flat-to-worse on position error and
# categorically worse on dropout count. Rather than reweighting
# state_vel_weight/grid/boundary again, this adds a term none of v9-v12
# had: an explicit penalty when a token's own rasterized PROB mass near
# its tracked position collapses toward zero, closing the "give-up is
# free" gap those terms never addressed. collapse-weight=0.5 and
# collapse-floor=0.3 are the loss's documented calibration (see
# model/token_losses.py's window_collapse_loss docstring) against a
# single well-centered ball's own window mass (~0.22-0.63 depending on
# sub-pixel offset), not blind tuning.
echo "[$(date -Iseconds)] starting training"
python -m model.token_train \
  --dataset-cache checkpoints/token_dataset_10000_h12_seed4738.pt \
  --n 20 --horizon 12 --epochs 3 \
  --hidden-dim 32 --neighbor-radius 4.0 \
  --bg-weight 0.05 --peak-weight 0.5 --boundary-weight 0.0 \
  --state-weight 1.0 --grid-weight 0.0 \
  --collapse-weight 0.5 --collapse-floor 0.3 \
  --lr 1e-3 --ramp-epochs 2 \
  --seed 4738 --log-every 200 \
  --checkpoint checkpoints/token_model_h12_v13.pt
echo "[$(date -Iseconds)] training done"
