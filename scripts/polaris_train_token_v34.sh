#!/bin/bash
#SBATCH --job-name=bounce-token-model-v34
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=bounce-token-model-v34-%j.log

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

# v34: control for v33: continue v31 for 3000 batches at 40-step unrolls with --ball-split and --speed-weight 0.1.
# plateau up to 20 steps (model/token_train.py train_stepwise), 25% replay of
# shorter horizons. Batch cost ~6 ms per rollout step, worst case ~75 min.
# Compare against v18 with scripts/eval_free_rollout.py at steps 5/10/20 and
# scripts/probe_y_mirror.py / probe_vx_drift.py.
echo "[$(date -Iseconds)] starting training"
python -m model.token_train \
  --dataset-cache checkpoints/token_dataset_6000_h44_seed4738.pt \
  --n 20 --horizon 44 \
  --free-rollout --stepwise-curriculum --curriculum-min-steps 40 --curriculum-max-steps 40 --init-checkpoint checkpoints/token_model_h44_v31.pt \
  --stage-batches 3000 --stage-tol 0.03 --stage-max-chunks 1 --replay-p 0.5 \
  --hidden-dim 32 --neighbor-radius 4.0 \
  --bg-weight 0.05 --peak-weight 0.5 --boundary-weight 0.1 \
  --state-weight 1.0 --grid-weight 0.1 \
  --velocity-readout --position-refine --ball-split --wall-lookahead --wall-head --pair-impulse --speed-weight 0.1 --lr 3e-4 \
  --seed 4738 --log-every 0 \
  --checkpoint checkpoints/token_model_h44_v34.pt
echo "[$(date -Iseconds)] training done"
