#!/usr/bin/env bash
#SBATCH --job-name=dynsphere-heldout-cpu
#SBATCH --array=0-15%4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/plgrid-heldout-cpu-%A-%a.out
#SBATCH --error=logs/plgrid-heldout-cpu-%A-%a.err
set -euo pipefail
: "${DYNSPHERE_REPO:?Set DYNSPHERE_REPO to the remote checkout}"
cd "$DYNSPHERE_REPO"
export DYNSPHERE_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1
exec .venv/bin/python -m research.submission_campaign.deep \
  --methods dagmm omni_anomaly_tsb_ad \
  --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count 16
