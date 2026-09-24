#!/bin/bash
#SBATCH --job-name=bounce-bench-scaling
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=bounce-bench-scaling-%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
cd "$HOME/bounce"
export PYTHONPATH=.
pip install -q -r requirements.txt

nproc
nvidia-smi -L
mkdir -p results

echo "[$(date -Iseconds)] cuda"
python scripts/bench_scaling.py --device cuda --out results/bench_scaling_cuda.json
echo "[$(date -Iseconds)] done"
