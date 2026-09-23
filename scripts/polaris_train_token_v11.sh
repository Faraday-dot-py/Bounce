#!/bin/bash
#SBATCH --job-name=bounce-token-model-v11
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=bounce-token-model-v11-%j.log

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

# Single-variable A/B vs v9: identical recipe (state-loss-only ablation,
# 10k cache, velocity_weight=0.0 -- same as v9), ONLY change is
# epochs 3->15 and ramp_epochs 2->10 (same ratio as the project default
# 25/50). Tests the untested assumption behind v6-v9's "large dataset,
# few epochs" recipe (docs/debugging/experiment-log.md: "applied the
# user's LLM-pretraining-style intuition ... since no grokking signal
# had been observed to justify heavy repetition") -- that assumption was
# never empirically validated for this recurrent multi-step model, and
# the "confirmed architectural" conclusion drawn from job 2852's 48-seed
# diagnostic was drawn from a checkpoint trained for only 3 epochs
# (30k sample-steps, fewer total gradient updates than the project's own
# default recipe of 2000 samples x 50 epochs = 100k). If more epochs on
# the same data measurably reduces self-fed rollout error
# (scripts/measure_error_compounding.py), the residual compounding is at
# least partly a training-budget confound, not purely architectural.
echo "[$(date -Iseconds)] starting training"
python -m model.token_train \
  --dataset-cache checkpoints/token_dataset_10000_h12_seed4738.pt \
  --n 20 --horizon 12 --epochs 15 \
  --hidden-dim 32 --neighbor-radius 4.0 \
  --bg-weight 0.05 --peak-weight 0.5 --boundary-weight 0.0 \
  --state-weight 1.0 --grid-weight 0.0 \
  --lr 1e-3 --ramp-epochs 10 \
  --seed 4738 --log-every 500 \
  --checkpoint checkpoints/token_model_h12_v11.pt
echo "[$(date -Iseconds)] training done"
