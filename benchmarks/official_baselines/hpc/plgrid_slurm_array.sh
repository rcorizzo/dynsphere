#!/bin/bash -l
#SBATCH --job-name=dynsphere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/plgrid-%A-%a.out
#SBATCH --error=logs/plgrid-%A-%a.err

set -euo pipefail

: "${DYNSPHERE_REPO:?Set DYNSPHERE_REPO to the repository path on PLGrid}"
: "${SLURM_ARRAY_TASK_ID:?Submit this script as a SLURM job array}"

PROFILE="${DYNSPHERE_PROFILE:-hpc-seed37}"
if [[ "$PROFILE" == "paper-multiseed" ]] && [[ -n "$(git status --porcelain)" ]]; then
  echo "The paper-multiseed campaign requires a clean committed repository" >&2
  exit 2
fi
SHARD_COUNT="${DYNSPHERE_SHARD_COUNT:-16}"
SHARD_INDEX="$SLURM_ARRAY_TASK_ID"
if (( SHARD_INDEX < 0 || SHARD_INDEX >= SHARD_COUNT )); then
  echo "SLURM_ARRAY_TASK_ID=${SHARD_INDEX} is outside the configured 0-$((SHARD_COUNT - 1)) array" >&2
  exit 2
fi

if [[ -n "${DYNSPHERE_PYTHON_MODULE:-}" ]]; then
  module load "$DYNSPHERE_PYTHON_MODULE"
fi

PYTHON_EXECUTABLE="${DYNSPHERE_PYTHON:-$DYNSPHERE_REPO/.venv/bin/python}"
TSB_DATA_ROOT="${DYNSPHERE_TSB_DATA_ROOT:-$DYNSPHERE_REPO/benchmarks/tsb_ad_m/data/TSB-AD-M}"
RESULT_ROOT="${DYNSPHERE_RESULT_ROOT:-$DYNSPHERE_REPO/results/official_baselines_v1}"
CAMPAIGN_MANIFEST="${DYNSPHERE_CAMPAIGN_MANIFEST:-$RESULT_ROOT/campaigns/$PROFILE.json}"
METHOD_ARGS=()
if [[ -n "${DYNSPHERE_METHODS:-}" ]]; then
  read -r -a METHODS <<< "$DYNSPHERE_METHODS"
  METHOD_ARGS=(--methods "${METHODS[@]}")
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
if [[ -n "${SLURM_TMPDIR:-}" ]]; then
  export TMPDIR="$SLURM_TMPDIR"
fi

cd "$DYNSPHERE_REPO"
exec "$PYTHON_EXECUTABLE" -m benchmarks.official_baselines.run_campaign \
  --profile "$PROFILE" \
  "${METHOD_ARGS[@]}" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --tsb-data-root "$TSB_DATA_ROOT" \
  --result-root "$RESULT_ROOT" \
  --campaign-manifest "$CAMPAIGN_MANIFEST" \
  --resume
