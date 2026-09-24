#!/bin/bash
#SBATCH --job-name=bounce-gravity-v1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=bounce-gravity-v1-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
pip install -q -r requirements.txt
mkdir -p checkpoints results

echo "[$(date -Iseconds)] starting gravity test"
python -m scripts.train_gravity_dynamics --seed 4738 --checkpoint checkpoints/gravity_dynamics_v1.pt --out results/gravity_test_v1.json
echo "[$(date -Iseconds)] done"
