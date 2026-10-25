# DynSphere: State and Motion Consistency for Multivariate Telemetry Anomaly Detection

DynSphere evaluates state and motion consistency within normal operating
concepts learned from multivariate telemetry. Each concept has ridge models
that predict latent and standardized-feature acceleration from state and
velocity. Robust training statistics scale the residuals and the remaining
state, speed, and transition terms. The implementation supports spherical
and Euclidean geometry with one or multiple concepts.

The repository includes the current detector, evaluation protocols, tests,
synthetic examples, and earlier research prototypes. The manuscript is
maintained in a separate Overleaf repository. The project is under active
research development; individual studies have distinct inputs, splits,
score support, and aggregation rules.

## Current implementation and evaluation

| Configuration | Python class | Concepts |
| --- | --- | ---: |
| DynSphere-S4 | `MultiConceptPendulumDetector` | 4 |
| DynSphere-S1 | `MultiConceptPendulumDetector` | 1 |
| DynSphere-E4 | `MultiConceptEuclideanDetector` | 4 |
| DynSphere-E1 | `MultiConceptEuclideanDetector` | 1 |

Both classes are in [`pendulum_detector.py`](pendulum_detector.py). Their
`decision_components()` method exposes the weighted score terms and the
score before and after temporal propagation. Numerical derivatives and
centered propagation use future observations, so the current configurations
provide retrospective scoring. The separate neural predictors and
window-conditioned prototypes retain their experimental status.

The primary held-out study uses standardized partial AUROC through a 5%
false-positive rate, with equal source-family weighting after seed and
trajectory averaging. AP/AUPRC, AUROC, VUS-PR, and VUS-ROC provide additional
summaries. Normal-calibrated thresholds are evaluated separately. Official
VUS results require `vus==0.0.6`; buffered fallback values are diagnostic.

| Study | Protocol and code |
| --- | --- |
| Five-family held-out telemetry | BATADAL, HAI-20.07, MSL, SMAP, and SMD; disjoint fitting, normal calibration, and test observations. See [`research/evaluation_protocols/`](research/evaluation_protocols/) and the [matched deep-comparison guide](research/submission_campaign/HPC_RUN_GUIDE.md). The 13-method matched comparison contains 59 trajectories after three DAGMM failures on one SMAP case. |
| Corrected CPU baselines and aggregation | [Isolated recovery and matched aggregation](research/submission_recovery/README.md), preserving original attempts and documenting upstream normalization limitations. |
| Exathlon/GHL/GutenTAG extension | [Extension preparation and evaluation](benchmarks/official_baselines/EXTENSION_EVALUATION.md). TSB-AD versions of Exathlon/GHL and independently generated GutenTAG examples have separately stated support. |
| Source-native Exathlon | [Frozen execution-separated study](research/exathlon_native/README.md): four applications, five test executions, three seeds, ten configurations, 108 fits, and 150 score evaluations. Pointwise and segment-weighted VUS summaries use different, explicitly reported cohorts. |
| Research prototypes | [Window-conditioned pilot](research/window_dynsphere/README.md) and [conditional-dynamics implementation](manifold_dynamics_detector.py), evaluated separately from the current DynSphere configurations. |

These studies are reported separately. Existing manifests, cached scores,
failed attempts, and historical outputs should be preserved. Frozen runners
verify source and input hashes; a changed checkout can require a new
campaign identity. Several dated guides describe their status at creation
and can contain references to jobs that have since completed.

## Installation and tests

For a fresh checkout, Python 3.11 is recommended for the controller environment.
External baselines use their own pinned environments and Python versions;
consult the [baseline setup guide](benchmarks/official_baselines/README.md)
and [cluster execution guide](benchmarks/official_baselines/HPC_EXECUTION.md).
The core test suite has also been checked in the existing Python 3.9 environment.

```bash
git clone https://github.com/rcorizzo/dynsphere.git
cd dynsphere
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt pytest
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m pytest \
  tests/test_pendulum_components.py \
  tests/test_official_baseline_local_controls.py \
  tests/test_official_baseline_contract.py \
  tests/test_exathlon_native.py \
  tests/test_exathlon_native_metrics.py -q
```

These self-contained checks run without downloading real datasets or upstream
baseline repositories. The complete suite additionally includes integration
checks requiring the 180-file TSB-AD-M selection and pinned CATCH/MTAD-GAT
source checkouts. After acquiring those assets using the guides below, run
`.venv/bin/python -m pytest tests -q`. The configured local environment passes
288 tests, with one dedicated-environment check skipped. A missing external
asset can cause an integration test to fail in a fresh clone.

Install the optional official temporal metrics in a compatible environment:

```bash
.venv/bin/python -m pip install vus==0.0.6
```

Full baseline training is separate from the tests. The cluster guides include
AU LSF and PLGrid Slurm runners. Submitting a runner starts computational
jobs; no job is submitted by installation or by reading this README.

## Algorithm-ready datasets

The checkout includes project-generated contextual and distribution-preserving
train/test splits and the 18 generated GutenTAG cases. Generation metadata,
feature names, labels, and file hashes are retained where provided. The
largest included dataset file is approximately 21 MiB. Earlier ensemble and
heterogeneous stress datasets remain reproducible through their generator
and are excluded from this publication snapshot.

For third-party data, the following commands populate the paths expected by
the loaders. They download source observations; each runner applies its own
declared split and fitting-only preprocessing.

```bash
.venv/bin/python benchmarks/real_world_datasets/download_real_world_datasets.py --dataset batadal
.venv/bin/python benchmarks/real_world_datasets/download_real_world_datasets.py --dataset hai_20_07
.venv/bin/python benchmarks/tsb_ad_m/download_tsb_ad_m.py
```

The TSB-AD-M download supplies the selected MSL, SMAP, SMD, Exathlon, and GHL
files. Source-native Exathlon uses a separate pinned release and preparation.
PowerSystem requires manual archive extraction. See [dataset acquisition,
expected layouts, attribution, and preparation](docs/DATASETS.md) for all
sources, including generation and download-retry instructions.

For an existing algorithm-ready train/test pair, the API is:

```python
from pendulum_detector import MultiConceptEuclideanDetector

detector = MultiConceptEuclideanDetector(n_concepts=1, random_state=37)
detector.fit(train_values, train_entity_ids)
scores = detector.decision_function(test_values, test_entity_ids)
components = detector.decision_components(test_values, test_entity_ids)
```

`train_values` and `test_values` are finite arrays with the same ordered
feature columns. Fit on normal observations only. Entity identifiers must
separate executions and uninterrupted segments. Labels, timestamps, and
synthetic debug columns are excluded from model features. Higher scores
indicate greater anomaly evidence. The detector fits its own feature scaler.

Large result directories, fitted models, logs, environments, and downloaded
external code are excluded from Git. Exact historical table regeneration
requires the corresponding frozen manifests and score artifacts, which are
maintained separately; the Git checkout alone is not a complete archive of
all manuscript experiments. Some historical analysis scripts expect those
artifacts at the documented paths.
