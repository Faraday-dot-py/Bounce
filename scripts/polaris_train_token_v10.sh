#!/bin/bash
#SBATCH --job-name=bounce-token-model-v10
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=bounce-token-model-v10-%j.log

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

# Single-variable A/B vs v9 (docs/debugging/experiment-log.md, job 2852):
# identical recipe (state-loss-only ablation, 3 epochs, 10k cache), ONLY
# change is velocity_weight=0.5, the new explicit two-frame velocity
# re-anchoring in model/token_model.py's TokenModel.step. Direct
# instrumentation (scripts/measure_error_compounding.py) showed v9's
# self-fed velocity error grows ~5x over 9 steps with nothing correcting
# it (only implicit correction through delta_vel), and splicing this
# correction onto the frozen v9 checkpoint at inference time made things
# WORSE (scripts/probe_velocity_correction.py) -- the hypothesis under
# test here is that the dynamics network needs to see this correction
# during training to learn to use it, not that the correction itself is
# wrong.
echo "[$(date -Iseconds)] starting training"
python -m model.token_train \
  --dataset-cache checkpoints/token_dataset_10000_h12_seed4738.pt \
  --n 20 --horizon 12 --epochs 3 \
  --hidden-dim 32 --neighbor-radius 4.0 \
  --bg-weight 0.05 --peak-weight 0.5 --boundary-weight 0.0 \
  --state-weight 1.0 --grid-weight 0.0 \
  --velocity-weight 0.5 \
  --lr 1e-3 --ramp-epochs 2 \
  --seed 4738 --log-every 200 \
  --checkpoint checkpoints/token_model_h12_v10.pt
echo "[$(date -Iseconds)] training done"
