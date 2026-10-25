#!/usr/bin/env bash
#SBATCH --job-name=dynsphere-heldout
#SBATCH --array=0-15%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/plgrid-heldout-%A-%a.out
#SBATCH --error=logs/plgrid-heldout-%A-%a.err
set -euo pipefail
: "${DYNSPHERE_REPO:?Set DYNSPHERE_REPO to the remote checkout}"
cd "$DYNSPHERE_REPO"
export DYNSPHERE_DEVICE=cuda OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1
read -r -a dynsphere_methods <<< "${DYNSPHERE_DEEP_METHODS:-mtad_gat catch tranad anomaly_transformer}"
exec .venv/bin/python -m research.submission_campaign.deep \
  --methods "${dynsphere_methods[@]}" \
  --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count 16
