#!/bin/bash -l

set -euo pipefail

: "${DYNSPHERE_REPO:?Set DYNSPHERE_REPO to the repository path on the cluster}"

if [[ -n "${DYNSPHERE_PYTHON_MODULE:-}" ]]; then
  module load "$DYNSPHERE_PYTHON_MODULE"
fi

BOOTSTRAP_PYTHON="${DYNSPHERE_BOOTSTRAP_PYTHON:-python3}"
cd "$DYNSPHERE_REPO"

"$BOOTSTRAP_PYTHON" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt pytest uv vus
export PATH="$DYNSPHERE_REPO/.venv/bin:$PATH"

.venv/bin/python - <<'PY'
import json
from dataclasses import asdict
from pathlib import Path

from benchmarks.official_baselines.setup_external import prepare_source

package = Path("benchmarks/official_baselines")
methods = (
    "dagmm",
    "catch",
    "mtad_gat",
    "omni_anomaly",
    "omni_anomaly_tsb_ad",
    "anomaly_transformer",
    "tranad",
)
records = []
for method_id in methods:
    record = prepare_source(method_id, package)
    row = asdict(record)
    row["source_dir"] = str(record.source_dir)
    row["environment_dir"] = str(record.environment_dir)
    row["packages"] = list(record.packages)
    records.append(row)
    print(method_id, record.status, record.reason)

output = Path("results/official_baselines_v1/setup/provenance_report.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
PY

.venv/bin/python -m pytest \
  tests/test_official_baseline_contract.py \
  tests/test_official_baseline_external.py \
  tests/test_official_baseline_adapters.py \
  tests/test_official_baseline_campaign.py \
  tests/test_official_baseline_runner.py -q
