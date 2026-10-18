#!/usr/bin/env bash

set -euo pipefail

: "${DYNSPHERE_REPO:?Set DYNSPHERE_REPO to the repository path on Lovelace}"
PYTHON_EXECUTABLE="${DYNSPHERE_PYTHON:-$DYNSPHERE_REPO/.venv/bin/python}"
TSB_DATA_ROOT="${DYNSPHERE_TSB_DATA_ROOT:-$DYNSPHERE_REPO/benchmarks/tsb_ad_m/data/TSB-AD-M}"
RESULT_ROOT="${DYNSPHERE_RESULT_ROOT:-$DYNSPHERE_REPO/results/official_baselines_v1}"
PROBE_ROOT="${DYNSPHERE_PROBE_ROOT:-$RESULT_ROOT/au-gpu-probe}"

export DYNSPHERE_DEVICE=cuda
export PYTHONUNBUFFERED=1

cd "$DYNSPHERE_REPO"
nvidia-smi
for method in catch mtad_gat anomaly_transformer tranad; do
  environment="benchmarks/official_baselines/.envs/$method/bin/python"
  "$environment" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
done

exec "$PYTHON_EXECUTABLE" -m benchmarks.official_baselines.run_campaign \
  --profile hpc-seed37 \
  --methods catch mtad_gat anomaly_transformer tranad \
  --datasets DynSphere-Contextual \
  --shard-index 0 \
  --shard-count 1 \
  --tsb-data-root "$TSB_DATA_ROOT" \
  --result-root "$PROBE_ROOT" \
  --campaign-manifest "$PROBE_ROOT/campaigns/au-gpu-probe.json" \
  --resume
