#!/usr/bin/env bash
set -euo pipefail
: "${DYNSPHERE_REPO:?Set DYNSPHERE_REPO to the code checkout}"
cd "$DYNSPHERE_REPO"
RESULT_ROOT="${DYNSPHERE_EXTENSION_RESULT_ROOT:-$DYNSPHERE_REPO/results/extension_v1}"
CAMPAIGN="${DYNSPHERE_EXTENSION_CAMPAIGN:-seven-methods-20261022}"
SHARD_COUNT="${DYNSPHERE_SHARD_COUNT:-16}"
if [[ -n "${LSB_JOBINDEX:-}" ]]; then
  SHARD_INDEX=$((LSB_JOBINDEX - 1))
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SHARD_INDEX="$SLURM_ARRAY_TASK_ID"
else
  SHARD_INDEX=0
  SHARD_COUNT=1
fi
# VUS is evaluated one input at a time; bound numerical-library threads.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 PYTHONUNBUFFERED=1
read -r -a FAMILIES <<< "${DYNSPHERE_METRIC_FAMILIES:-Exathlon GHL}"
exec "${DYNSPHERE_PYTHON:-$DYNSPHERE_REPO/.venv/bin/python}" \
  -m benchmarks.official_baselines.cached_extension_metrics \
  --input-csv "$RESULT_ROOT/summaries/$CAMPAIGN/combined/per_run.csv" \
  --runs-root "$RESULT_ROOT/runs" \
  --output-dir "$RESULT_ROOT/vus/$CAMPAIGN/real" \
  --families "${FAMILIES[@]}" \
  --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" "$@"
