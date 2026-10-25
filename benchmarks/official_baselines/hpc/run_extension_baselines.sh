#!/usr/bin/env bash
set -euo pipefail
: "${DYNSPHERE_REPO:?Set DYNSPHERE_REPO to the code checkout}"
cd "$DYNSPHERE_REPO"
if ! command -v git >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
  module load "${DYNSPHERE_GIT_MODULE:-git/2.40.0}"
fi
command -v git >/dev/null 2>&1 || { echo 'Git is required for provenance' >&2; exit 2; }
GROUP="${DYNSPHERE_EXTENSION_GROUP:-gpu}"
case "$GROUP" in
  gpu) METHODS=(catch mtad_gat anomaly_transformer tranad); export DYNSPHERE_DEVICE=cuda ;;
  cpu) METHODS=(dagmm omni_anomaly_tsb_ad); export DYNSPHERE_DEVICE=cpu ;;
  local) METHODS=(kmeans_distance predictive_euclidean_single); export DYNSPHERE_DEVICE=cpu ;;
  predictor) METHODS=(predictive_euclidean_single); export DYNSPHERE_DEVICE=cpu ;;
  *) echo "Unknown experiment group: $GROUP" >&2; exit 2 ;;
esac
SHARD_COUNT="${DYNSPHERE_SHARD_COUNT:-16}"
if [[ -n "${LSB_JOBINDEX:-}" ]]; then
  SHARD_INDEX=$((LSB_JOBINDEX - 1))
elif [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  SHARD_INDEX="$SLURM_ARRAY_TASK_ID"
else
  SHARD_INDEX=0
  SHARD_COUNT=1
fi
# Bound numerical libraries independently of scheduler slot count.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 PYTHONUNBUFFERED=1
exec "${DYNSPHERE_PYTHON:-$DYNSPHERE_REPO/.venv/bin/python}" \
  -m benchmarks.official_baselines.extension_campaign \
  --families Exathlon GHL GutenTAG \
  --native-manifest "${DYNSPHERE_NATIVE_MANIFEST:-$DYNSPHERE_REPO/data/gutentag_extension_v1/native_manifest.json}" \
  --tsb-data-root "${DYNSPHERE_TSB_DATA_ROOT:-$DYNSPHERE_REPO/benchmarks/tsb_ad_m/data/TSB-AD-M}" \
  --result-root "${DYNSPHERE_BASELINE_RESULT_ROOT:-$DYNSPHERE_REPO/results/extension_baselines_v2}" \
  --baseline-profile audited-v2 --limit-per-family "${DYNSPHERE_EXTENSION_LIMIT:-0}" \
  --methods "${METHODS[@]}" --seeds 37 38 39 \
  --shard-index "$SHARD_INDEX" --shard-count "$SHARD_COUNT" "$@"
