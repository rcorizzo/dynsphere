# Official Baselines and Expanded Evaluation

This directory defines the source-traceable evaluation introduced after the
original ICDM submission. Outputs written by earlier runners remain historical
audit artifacts. Only runs below `results/official_baselines_v1/` use the new
private-label contract, training-only scaling, immutable source registry, and
family-balanced aggregation protocol.

## 1. Environment and Data

Create the repository environment and install the official VUS implementation:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt pytest
uv pip install --python .venv/bin/python vus
```

The contextual data are already generated under `data/contextual/`. Prepare the
real-world and TSB-AD-M data with the download instructions in the repository
README. The default TSB root is
`benchmarks/tsb_ad_m/data/TSB-AD-M/`, and the evaluation manifest is
`benchmarks/official_baselines/manifests/TSB-AD-M-Eva.csv`.

## 2. Registry and Manifest Validation

Resolve every local revision, validate all pinned external revisions, and check
the 180-row manifest before creating an experiment:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from benchmarks.official_baselines.datasets import read_eval_manifest
from benchmarks.official_baselines.provenance import load_registry

root = Path("benchmarks/official_baselines")
registry = load_registry(root / "registry.json")
manifest = read_eval_manifest(root / "manifests" / "TSB-AD-M-Eva.csv")
print(f"implementations={len(registry)} manifest_rows={len(manifest)}")
for method_id, spec in sorted(registry.items()):
    print(method_id, spec.revision, spec.implementation_category)
PY
```

The ten primary displayed methods are PCA reconstruction, KMeans distance,
DAGMM, CATCH, MTAD-GAT, OmniAnomaly, Anomaly Transformer, TranAD, the one-step
Euclidean predictor, and one-regime DynSphere. Fallback implementations have
distinct method identifiers and are never selected implicitly.

## 3. External Source Setup

Prepare the six preferred external implementations at their pinned commits:

```bash
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
```

Each source checkout is detached at a 40-character commit and remains
unchanged. Each implementation receives an isolated environment. An unavailable
Python version or incompatible dependency set produces a structured
`incompatible` record. It does not activate a fallback.

Audit previously prepared checkouts with:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from benchmarks.official_baselines.setup_external import verify_source

package = Path("benchmarks/official_baselines")
for method_id in ("dagmm", "catch", "mtad_gat", "omni_anomaly", "anomaly_transformer", "tranad"):
    record = verify_source(method_id, package)
    print(method_id, record.status, record.revision, record.python_version, record.reason)
PY
```

## 4. Contract and Adapter Checks

Run the label-isolation, fake-process, real-adapter interface, runner, and
aggregation tests:

```bash
.venv/bin/pytest \
  tests/test_official_baseline_contract.py \
  tests/test_official_baseline_external.py \
  tests/test_official_baseline_adapters.py \
  tests/test_official_baseline_runner.py \
  tests/test_official_baseline_aggregation.py -q
```

Every adapter process receives a directory containing exactly `train.npy`,
`test.npy`, and public `metadata.json`. Labels remain under the private case
directory. A run is complete only when it returns one finite score for every
test row.

## 5. Matrix Validation

Validate the complete matrix without fitting a model:

```bash
.venv/bin/python -m benchmarks.official_baselines.run_expanded_evaluation \
  --dry-run \
  --result-root results/official_baselines_v1
```

With the default manifest, the dry run recognizes 180 TSB-AD-M trajectories.
MSL, SMAP, and SMD cases selected through both the current-dataset stage and the
complete TSB stage are deduplicated for seed 37. The current datasets use seeds
37, 38, and 39. Remaining TSB trajectories use seed 37.

## 6. One-Trajectory Smoke Stage

The following command selects the first MSL trajectory for each requested
method and executes one job per method:

```bash
.venv/bin/python - <<'PY'
from dataclasses import replace
from pathlib import Path
from benchmarks.official_baselines.run_expanded_evaluation import (
    RunConfig,
    build_matrix,
    execute_job,
)

config = RunConfig(
    datasets=("MSL",),
    seeds=(37,),
    result_root=Path("results/official_baselines_v1"),
    timeout_seconds=3600,
    resume=True,
)
selected = {}
for job in build_matrix(config):
    selected.setdefault(job.implementation.method_id, job)
for method_id, job in sorted(selected.items()):
    result = execute_job(job)
    print(method_id, result["status"], result.get("error", ""))
PY
```

Inspect each terminal record, `execution/stdout.log`, `execution/stderr.log`,
and score vector before expanding execution. Preferred implementations that
cannot run on the host must have a reproducible `incompatible` record.

## 7. Current Datasets

Run the contextual benchmark, MSL, SMAP, SMD, BATADAL, HAI-20.07, and
PowerSystem with seeds 37, 38, and 39:

```bash
.venv/bin/python -m benchmarks.official_baselines.run_expanded_evaluation \
  --datasets DynSphere-Contextual MSL SMAP SMD BATADAL HAI-20.07 PowerSystem \
  --seeds 37 38 39 \
  --resume \
  --timeout-seconds 3600 \
  --result-root results/official_baselines_v1
```

## 8. Complete 180-Trajectory Evaluation

Run every trajectory in the pinned TSB-AD-M evaluation manifest with seed 37:

```bash
.venv/bin/python -m benchmarks.official_baselines.run_expanded_evaluation \
  --datasets TSB-AD-M \
  --seeds 37 \
  --resume \
  --timeout-seconds 3600 \
  --result-root results/official_baselines_v1
```

Resume uses the case identity, source revision, hyperparameters, seed, input
digest, and scaling rule. A pending, failed, timed-out, corrupt, or mismatched
record executes again. Existing complete identities are validated before they
are skipped.

## 9. Aggregation

Generate per-run, per-trajectory, per-family, family-macro, paired, completion,
and readiness artifacts:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from benchmarks.official_baselines.aggregate import aggregate_runs
from benchmarks.official_baselines.provenance import resolve_local_revision

artifacts = aggregate_runs(
    Path("results/official_baselines_v1"),
    Path("results/official_baselines_v1/aggregate"),
    expected_adapter_revision=resolve_local_revision(Path.cwd()),
)
print(artifacts)
PY
```

Aggregation first averages seeds within a trajectory, then trajectories within
a source family, and finally gives each source family equal weight. AUPRC and
VUS-PR are the primary metrics. A manuscript-ready LaTeX table is created only
when every observed method has a valid run, every terminal record is complete,
and every VUS value comes from the official backend. The adapter-revision
filter excludes preserved smoke runs and records produced by earlier adapter
code from the current summary.

## 10. Output Audit

The versioned result root separates artifacts as follows:

1. `setup/` stores source and environment provenance.
2. `runs/` stores one identity directory per case, method, and seed.
3. Each run stores public input under `case/model_input/`, labels under
   `case/private/`, process artifacts under `execution/`, and its terminal
   record at `run.json`.
4. `aggregate/` stores metric and completion summaries.

Run the complete repository verification before interpreting results:

```bash
.venv/bin/pytest -q
```

Do not copy values from the historical local-proxy result files into the new
comparison. Any paper-table replacement requires the completion and readiness
reports from this protocol.

## 11. Portable MacBook and Cluster Runners

Named campaign profiles, deterministic job partitioning, and scheduler array
wrappers are documented in [HPC_EXECUTION.md](HPC_EXECUTION.md). The supported
targets are:

1. sequential MacBook execution through `--profile macbook-safe`;
2. American University Lovelace through the LSF array wrapper;
3. PLGrid through the SLURM array wrapper.

Inspect any profile without fitting models:

```bash
.venv/bin/python -m benchmarks.official_baselines.run_campaign \
  --profile macbook-safe --dry-run
```

Campaign-scoped aggregation excludes historical, smoke, and unrelated run
records:

```bash
.venv/bin/python -m benchmarks.official_baselines.aggregate_campaign \
  --campaign-manifest results/official_baselines_v1/campaigns/macbook-safe.json \
  --result-root results/official_baselines_v1
```
