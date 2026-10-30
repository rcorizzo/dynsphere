# DynSphere

DynSphere detects anomalies in multivariate time series using state and motion
consistency within normal operating concepts. It provides spherical and
Euclidean variants with one or four concepts. The Euclidean one-concept
configuration is DynSphere (E1).

## Installation

Use Python 3.11 for the controller environment. External baselines require
their own Python versions and dependency sets.

```bash
git clone https://github.com/rcorizzo/dynsphere.git
cd dynsphere
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt pytest
```

For official temporal metrics, install the optional VUS implementation in a
compatible environment:

```bash
.venv/bin/python -m pip install vus==0.0.6
```

## Data

Generated contextual and GutenTAG examples are included in [data/](data/).
Obtain third-party datasets from their original sources:

```bash
.venv/bin/python benchmarks/real_world_datasets/download_real_world_datasets.py --dataset batadal
.venv/bin/python benchmarks/real_world_datasets/download_real_world_datasets.py --dataset hai_20_07
.venv/bin/python benchmarks/tsb_ad_m/download_tsb_ad_m.py
```

See [dataset preparation](docs/DATASETS.md) for expected paths, feature and
label columns, source-native Exathlon preparation, and data terms. Downloaded
data retain their upstream licenses and are not redistributed here.

## Detector API

```python
from pendulum_detector import MultiConceptEuclideanDetector

detector = MultiConceptEuclideanDetector(n_concepts=1, random_state=37)
detector.fit(train_values, train_entity_ids)
scores = detector.decision_function(test_values, test_entity_ids)
components = detector.decision_components(test_values, test_entity_ids)
```

Inputs are finite numeric arrays with the same ordered features. Fit on
normal training observations. Entity identifiers separate independent
sequences. Exclude labels, timestamps, and simulator diagnostic columns from
model features. Higher scores indicate stronger anomaly evidence. Centered
derivatives and score propagation make the current implementation suitable
for retrospective scoring.

## Tests

The following tests use synthetic fixtures and require no dataset downloads:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_pendulum_components.py \
  tests/test_official_baseline_local_controls.py \
  tests/test_official_baseline_contract.py \
  tests/test_exathlon_native.py \
  tests/test_exathlon_native_metrics.py \
  tests/test_replication_preparation.py -q
```

The full suite is run with `.venv/bin/python -m pytest tests -q`.
Some integration tests additionally require downloaded datasets and pinned
external source checkouts. The timing-extension integration test also
requires its original manifest and input bundle.

## Experiments

Run commands from the repository root. The guides specify input preparation,
environment requirements, execution, and aggregation.

| Experiment | Instructions |
| --- | --- |
| Baseline setup and benchmark evaluation | [Baseline guide](benchmarks/official_baselines/README.md) |
| Held-out multiseed comparison | [Held-out guide](research/submission_campaign/HPC_RUN_GUIDE.md) |
| Corrected DAGMM and OmniAnomaly evaluation | [CPU baseline guide](research/submission_recovery/README.md) |
| Source-native Exathlon | [Exathlon guide](research/exathlon_native/README.md) |
| Exathlon/GHL/GutenTAG extension | [Extension guide](benchmarks/official_baselines/EXTENSION_EVALUATION.md) |
| CPU fitting, scoring, memory, and storage | [Timing guide](research/quality_cost/README.md), [four-method extension](research/quality_cost_extension/README.md) |
| PCA-95 sensitivity | [PCA guide](research/pca95/README.md) |
| Controlled-edit explanations | [Explanation guide](research/explanation_study/README.md) |
| Independent Linux explanations and embedding PCA | [Replication guide](research/explanation_replication/README.md) |

External methods use pinned upstream repositories and separate environments.
See the baseline guide before running deep models. Scheduler templates must
be configured for the user's cluster and allocation.

Frozen study controllers require their specified manifests, prepared arrays,
runtime inventories, and, where applicable, earlier score/checkpoint bundles.
These generated artifacts are not bundled with the code. A fresh clone can
run the synthetic tests and prepare new benchmark evaluations; exact
regeneration of the paper's tables additionally requires the corresponding
saved artifacts. The guides identify these prerequisites. Changed code or
inputs require a new campaign identity.

Results are written to ignored output directories. Preserve failed and
timed-out outcomes and compare methods on identical case/seed support.
Use a new output directory for a new experiment.
