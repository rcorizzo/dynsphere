#!/usr/bin/env bash
set -euo pipefail

: "${DYNSPHERE_REPO:?Set DYNSPHERE_REPO to the code repository}"
cd "$DYNSPHERE_REPO"
SHARD_COUNT="${DYNSPHERE_SHARD_COUNT:-16}"
if [[ -n "${LSB_JOBINDEX:-}" ]]; then
  SHARD_INDEX=$((LSB_JOBINDEX - 1))
  THREAD_COUNT="${LSB_DJOB_NUMPROC:-4}"
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SHARD_INDEX="$SLURM_ARRAY_TASK_ID"
  THREAD_COUNT="${SLURM_CPUS_PER_TASK:-4}"
else
  SHARD_INDEX=0
  SHARD_COUNT=1
  THREAD_COUNT=1
fi
if ! command -v git >/dev/null 2>&1; then
  if command -v module >/dev/null 2>&1; then
    module load "${DYNSPHERE_GIT_MODULE:-git/2.40.0}"
  fi
fi
command -v git >/dev/null 2>&1 || { echo 'Git is required for experiment provenance' >&2; exit 2; }
export OMP_NUM_THREADS="$THREAD_COUNT"
export OPENBLAS_NUM_THREADS="$THREAD_COUNT"
export MKL_NUM_THREADS="$THREAD_COUNT"
export VECLIB_MAXIMUM_THREADS="$THREAD_COUNT"
export LOKY_MAX_CPU_COUNT="$THREAD_COUNT"
export DYNSPHERE_DEVICE=cpu
export PYTHONUNBUFFERED=1
read -r -a METHODS <<< "${DYNSPHERE_METHODS:-dynsphere dynsphere_euclidean dynsphere_one_regime dynsphere_euclidean_one_regime pca_reconstruction linear_ridge_ar1}"
exec "${DYNSPHERE_PYTHON:-$DYNSPHERE_REPO/.venv/bin/python}" \
  -m benchmarks.official_baselines.extension_campaign \
  --families Exathlon GHL GutenTAG \
  --native-manifest "${DYNSPHERE_NATIVE_MANIFEST:-$DYNSPHERE_REPO/data/gutentag_extension_v1/native_manifest.json}" \
  --tsb-data-root "${DYNSPHERE_TSB_DATA_ROOT:-$DYNSPHERE_REPO/benchmarks/tsb_ad_m/data/TSB-AD-M}" \
  --result-root "${DYNSPHERE_EXTENSION_RESULT_ROOT:-$DYNSPHERE_REPO/results/extension_v1}" \
  --limit-per-family "${DYNSPHERE_EXTENSION_LIMIT:-0}" \
  --seeds 37 38 39 --methods "${METHODS[@]}" \
  --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" "$@"
