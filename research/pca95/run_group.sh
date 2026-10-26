#!/usr/bin/env bash
set -euo pipefail
: "${DYNSPHERE_PCA95_REPO:?Set DYNSPHERE_PCA95_REPO to the verified checkout}"
cd "$DYNSPHERE_PCA95_REPO"
group="${1:?Choose local, external_cpu, or gpu}"
shift
output="${DYNSPHERE_PCA95_OUTPUT:-$DYNSPHERE_PCA95_REPO/results/pca95_sensitivity_20261026/v1}"
proof="${DYNSPHERE_PCA95_TIMING_PROOF:-$DYNSPHERE_PCA95_REPO/results/quality_cost_20261026/five-family-cpu-v1/completion.json}"
python_executable="${DYNSPHERE_PYTHON:-$DYNSPHERE_PCA95_REPO/.venv/bin/python}"
if [[ ! -f "$output/manifest.json" ]]; then
  echo "Prepare the PCA-95 planning manifest with --dry-run before running this group" >&2
  exit 2
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 PYTHONUNBUFFERED=1
device=cpu
if [[ "$group" == gpu ]]; then device=cuda; fi
shard_index=0
shard_count=1
if [[ -n "${LSB_JOBINDEX:-}" ]]; then
  shard_index="$((LSB_JOBINDEX-1))"
  shard_count="${DYNSPHERE_PCA95_SHARDS:-16}"
fi
exec "$python_executable" -m research.pca95.run --output "$output" --group "$group" \
  --device "$device" --timing-proof "$proof" \
  --shard-index "$shard_index" --shard-count "$shard_count" "$@"
