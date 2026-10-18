#!/usr/bin/env bash

set -euo pipefail

: "${DYNSPHERE_REPO:?Set DYNSPHERE_REPO to the repository path on Lovelace}"
: "${DYNSPHERE_METHODS:?Set DYNSPHERE_METHODS to a space-separated method list}"
: "${LSB_JOBINDEX:?Submit this script through an LSF job array}"

if ! command -v git >/dev/null 2>&1; then
  if ! command -v module >/dev/null 2>&1; then
    echo "Git is unavailable and the environment-modules command is not initialized" >&2
    exit 2
  fi
  module load "${DYNSPHERE_GIT_MODULE:-git/2.40.0}"
fi
if ! command -v git >/dev/null 2>&1; then
  echo "Git remains unavailable after loading ${DYNSPHERE_GIT_MODULE:-git/2.40.0}" >&2
  exit 2
fi

PROFILE="${DYNSPHERE_PROFILE:-hpc-seed37}"
if [[ "$PROFILE" == "paper-multiseed" ]] && [[ -n "$(git status --porcelain)" ]]; then
  echo "The paper-multiseed campaign requires a clean committed repository" >&2
  exit 2
fi
SHARD_COUNT="${DYNSPHERE_SHARD_COUNT:-16}"
SHARD_INDEX=$((LSB_JOBINDEX - 1))
if (( LSB_JOBINDEX < 1 || SHARD_INDEX >= SHARD_COUNT )); then
  echo "LSB_JOBINDEX=${LSB_JOBINDEX} is outside the configured 1-${SHARD_COUNT} array" >&2
  exit 2
fi

PYTHON_EXECUTABLE="${DYNSPHERE_PYTHON:-$DYNSPHERE_REPO/.venv/bin/python}"
TSB_DATA_ROOT="${DYNSPHERE_TSB_DATA_ROOT:-$DYNSPHERE_REPO/benchmarks/tsb_ad_m/data/TSB-AD-M}"
RESULT_ROOT="${DYNSPHERE_RESULT_ROOT:-$DYNSPHERE_REPO/results/official_baselines_v1}"
CAMPAIGN_MANIFEST="${DYNSPHERE_CAMPAIGN_MANIFEST:?Set a component-specific campaign manifest}"
read -r -a METHODS <<< "$DYNSPHERE_METHODS"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${LSB_DJOB_NUMPROC:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

cd "$DYNSPHERE_REPO"
exec "$PYTHON_EXECUTABLE" -m benchmarks.official_baselines.run_campaign \
  --profile "$PROFILE" \
  --methods "${METHODS[@]}" \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --tsb-data-root "$TSB_DATA_ROOT" \
  --result-root "$RESULT_ROOT" \
  --campaign-manifest "$CAMPAIGN_MANIFEST" \
  --resume
