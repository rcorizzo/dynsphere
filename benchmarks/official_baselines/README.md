# Baseline setup and evaluation

This package provides pinned implementations, dataset adapters, process
isolation, and coverage-aware aggregation. Run commands from the repository
root after completing the [main installation](../../README.md) and
[dataset preparation](../../docs/DATASETS.md).

## External implementations

Each deep method uses an isolated environment. The source revision and
runtime requirements are declared in [registry.json](registry.json).
Prepare the preferred implementations:

```bash
.venv/bin/python -m pip install uv
export PATH="$PWD/.venv/bin:$PATH"

.venv/bin/python - <<'PY'
from pathlib import Path
from benchmarks.official_baselines.setup_external import prepare_source

package = Path("benchmarks/official_baselines")
for method in ("dagmm", "catch", "mtad_gat", "omni_anomaly", "omni_anomaly_tsb_ad",
               "anomaly_transformer", "tranad"):
    record = prepare_source(method, package)
    print(method, record.status, record.reason)
PY
```

Use the Python version specified for each method. Source setup creates
`.external/` checkouts and method environments under `.envs/`.
An unavailable interpreter or dependency set is reported as incompatible;
it does not select a fallback implementation.

Check previously prepared sources:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from benchmarks.official_baselines.setup_external import verify_source

package = Path("benchmarks/official_baselines")
for method in ("dagmm", "catch", "mtad_gat", "omni_anomaly", "omni_anomaly_tsb_ad",
               "anomaly_transformer", "tranad"):
    record = verify_source(method, package)
    print(method, record.status, record.revision, record.reason)
PY
```

## Validation and execution

```bash
.venv/bin/python -m pytest \
  tests/test_official_baseline_contract.py \
  tests/test_official_baseline_external.py \
  tests/test_official_baseline_adapters.py \
  tests/test_official_baseline_runner.py \
  tests/test_official_baseline_aggregation.py -q

.venv/bin/python -m benchmarks.official_baselines.run_campaign \
  --profile macbook-safe --dry-run
```

A dry run records the planned matrix without model fitting. Inspect the
source, input, runtime, and configuration evidence before starting a large
campaign. Run the chosen profile without `--dry-run` only after its
prerequisites are available.

Profiles in this package include historical protocols. Use the
[held-out guide](../../research/submission_campaign/HPC_RUN_GUIDE.md) for
the separate held-out comparison and the
[CPU baseline guide](../../research/submission_recovery/README.md) for the
corrected DAGMM/OmniAnomaly interface. The
[extension guide](EXTENSION_EVALUATION.md) covers Exathlon/GHL/GutenTAG.
[Cluster execution](HPC_EXECUTION.md) describes scheduler configuration.

## Output and aggregation

Adapters receive observed training/test arrays and metadata. Evaluation
labels are isolated from detector fitting and scoring. The output records
scores, source/input hashes, configurations, runtime information, and
terminal status. Preserve unsuccessful outcomes.

```bash
.venv/bin/python -m benchmarks.official_baselines.aggregate_campaign \
  --campaign-manifest results/official_baselines_v1/campaigns/macbook-safe.json \
  --result-root results/official_baselines_v1
```

Use the manifest corresponding to the executed profile. Different data
splits, boundary exclusions, fitting budgets, or method identities define
different comparisons and must remain separate. Official VUS evaluation
uses `vus==0.0.6`; fallback metrics are not interchangeable.
