#!/usr/bin/env bash
#BSUB -J "dynsphere[1-16]"
#BSUB -q normal
#BSUB -n 4
#BSUB -R "rusage[mem=16000]"
#BSUB -W 48:00
#BSUB -o logs/au-%J-%I.out
#BSUB -e logs/au-%J-%I.err

set -euo pipefail

: "${DYNSPHERE_REPO:?Set DYNSPHERE_REPO to the repository path on Lovelace}"
: "${LSB_JOBINDEX:?Submit this script as an LSF job array}"

PROFILE="${DYNSPHERE_PROFILE:-hpc-seed37}"
SHARD_COUNT="${DYNSPHERE_SHARD_COUNT:-16}"
SHARD_INDEX=$((LSB_JOBINDEX - 1))
if (( LSB_JOBINDEX < 1 || SHARD_INDEX >= SHARD_COUNT )); then
  echo "LSB_JOBINDEX=${LSB_JOBINDEX} is outside the configured 1-${SHARD_COUNT} array" >&2
  exit 2
fi

PYTHON_EXECUTABLE="${DYNSPHERE_PYTHON:-$DYNSPHERE_REPO/.venv/bin/python}"
TSB_DATA_ROOT="${DYNSPHERE_TSB_DATA_ROOT:-$DYNSPHERE_REPO/benchmarks/tsb_ad_m/data/TSB-AD-M}"
RESULT_ROOT="${DYNSPHERE_RESULT_ROOT:-$DYNSPHERE_REPO/results/official_baselines_v1}"
CAMPAIGN_MANIFEST="${DYNSPHERE_CAMPAIGN_MANIFEST:-$RESULT_ROOT/campaigns/$PROFILE.json}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${LSB_DJOB_NUMPROC:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

cd "$DYNSPHERE_REPO"
exec "$PYTHON_EXECUTABLE" -m benchmarks.official_baselines.run_campaign \
  --profile "$PROFILE" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --tsb-data-root "$TSB_DATA_ROOT" \
  --result-root "$RESULT_ROOT" \
  --campaign-manifest "$CAMPAIGN_MANIFEST" \
  --resume
