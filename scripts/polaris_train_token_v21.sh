#!/bin/bash
#SBATCH --job-name=bounce-token-model-v21
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=bounce-token-model-v21-%j.log

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

# v21: v21 recipe plus --velocity-readout (init velocity from frame VX/VY channels, model/token_detect.py read_token_velocities). v18 architecture, stepwise curriculum -- 1 step out, advancing on loss
# plateau up to 20 steps (model/token_train.py train_stepwise), 25% replay of
# shorter horizons. Batch cost ~6 ms per rollout step, worst case ~75 min.
# Compare against v18 with scripts/eval_free_rollout.py at steps 5/10/20 and
# scripts/probe_y_mirror.py / probe_vx_drift.py.
echo "[$(date -Iseconds)] starting training"
python -m model.token_train \
  --dataset-cache checkpoints/token_dataset_10000_h24_seed4738.pt \
  --n 20 --horizon 24 \
  --free-rollout --stepwise-curriculum --curriculum-max-steps 20 \
  --stage-batches 1000 --stage-tol 0.03 --stage-max-chunks 4 --replay-p 0.25 \
  --hidden-dim 32 --neighbor-radius 4.0 \
  --bg-weight 0.05 --peak-weight 0.5 --boundary-weight 0.1 \
  --state-weight 1.0 --grid-weight 0.1 \
  --velocity-readout --lr 1e-3 \
  --seed 4738 --log-every 0 \
  --checkpoint checkpoints/token_model_h24_v21.pt
echo "[$(date -Iseconds)] training done"
