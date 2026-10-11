# Official Baselines and Expanded Evaluation Design

## Purpose

This revision will replace the evidential role of the compact local deep-model
proxies with source-traceable implementations, add the requested minimum
baseline set, and extend evaluation beyond the six real-world datasets
currently emphasized in the manuscript. Existing results and manuscript tables
will remain intact until the replacement results have been generated, checked,
and explicitly approved for use.

The work has three goals:

1. make every named external method traceable to a pinned implementation;
2. compare DynSphere with stronger density, clustering, frequency, graph, and
   temporal baselines under a common leakage-free protocol; and
3. determine whether the paper's methodology is understandable and consistent
   with the executed code after the new evidence is available.

## Scope

The minimum baseline set is:

1. PCA reconstruction;
2. KMeans distance;
3. DAGMM;
4. CATCH;
5. MTAD-GAT, selected as the graph-attention baseline because its available
   PyTorch implementation is more compatible with the current hardware than
   the legacy GDN environment;
6. source-traceable OmniAnomaly, Anomaly Transformer, and TranAD;
7. the matched Euclidean conditional-dynamics control; and
8. the one-regime DynSphere control.

The expanded dataset set is:

1. DynSphere-Contextual;
2. BATADAL;
3. HAI-20.07;
4. PowerSystem; and
5. all trajectories in the official `TSB-AD-M-Eva.csv` manifest.

The historical proxy results remain available for comparison and audit. No
historical JSON, CSV, table, or discussion will be deleted or overwritten by
this work.

## Implementation Provenance Policy

Each external method will have a registry entry containing:

- the displayed method name;
- the implementation category, such as `author-released` or
  `benchmark-maintained`;
- the repository URL;
- a pinned Git commit identifier;
- the supported Python environment;
- the adapter command;
- default hyperparameters; and
- an optional explicitly named fallback implementation.

The preferred sources are:

| Method | Preferred implementation |
| --- | --- |
| PCA reconstruction | local scikit-learn adapter |
| KMeans distance | local scikit-learn adapter |
| DAGMM | TAB benchmark implementation |
| CATCH | author-released CATCH repository |
| MTAD-GAT | ML4ITS PyTorch implementation |
| OmniAnomaly | author-released OmniAnomaly repository |
| Anomaly Transformer | author-released Anomaly Transformer repository |
| TranAD | author-released TranAD repository |
| Euclidean control | existing local matched control |
| One-regime DynSphere | existing local DynSphere ablation |

Compatibility fallbacks may use TSB-AD or TAB implementations. A fallback
requires an explicit command-line selection and receives a distinct provenance
record. The runner will never substitute one implementation silently.

External repositories will be checked out beneath an ignored workspace
directory. Their source will remain unmodified. Integration-specific code will
live in this repository.

## Architecture

### Provenance registry

A machine-readable registry will define implementation sources, pinned
revisions, environments, and adapter entry points. A validation command will
check required fields and reject mutable or missing revision identifiers.

### Canonical experiment contract

The main runner will export each experiment as a canonical bundle containing:

- `train.npy`: normal training observations;
- `test.npy`: observations to score;
- `labels.npy`: evaluation labels, unavailable to model fitting;
- `metadata.json`: dataset, trajectory, feature count, seed, scaling, declared
  prefix length, and row identifiers.

An external adapter will consume the training and test arrays and write:

- `scores.npy`: one score for every test row;
- `run.json`: implementation identity, hyperparameters, timing, environment,
  warnings, and status.

Warm-up positions without a complete temporal window will receive an explicit
alignment policy recorded in `run.json`. Adapters must return a finite score
vector whose length exactly matches the test labels.

### Local controls

PCA reconstruction will fit PCA on normal training observations and score the
squared reconstruction residual. Its component rule will be fixed before
evaluation and recorded in every run.

KMeans will fit on normal training observations and score Euclidean distance to
the closest centroid. Its cluster-count rule will also be fixed before
evaluation.

The matched Euclidean control and one-regime DynSphere control will use the
existing detector implementations through the same canonical experiment
contract.

### External execution

Each implementation family will execute in an isolated environment through a
subprocess. This isolates conflicting Python, PyTorch, TensorFlow, and graph
library requirements. Setup scripts will create or verify the environments,
check out pinned revisions, and emit a lock and provenance report.

The experiment runner will support:

- method and dataset filters;
- manifest-based TSB-AD-M selection;
- per-run timeouts;
- resumable execution;
- deterministic seed propagation where supported;
- score caching;
- failed-run recording; and
- dry-run validation of the complete experiment matrix.

## Evaluation Protocol

### Training and preprocessing

For TSB-AD-M, the declared filename prefix supplies training data and the full
trajectory supplies test observations. Prefix integrity will be checked before
fitting. The known contaminated SMD trajectory will retain its documented
maximal-clean-prefix treatment and will be marked in metadata.

BATADAL, HAI-20.07, and PowerSystem will use their existing normal training
portions and labeled test portions. DynSphere-Contextual will use the released
normal training and labeled test files.

Scaling parameters will be fitted exclusively on training observations and
then applied to test observations. Labels will be loaded by the orchestration
process after scoring and will never be placed in an external model's input
bundle.

### Repetitions

DynSphere-Contextual and the current six real-world datasets will use seeds 37,
38, and 39 for stochastic methods. The initial complete TSB-AD-M evaluation
split will use seed 37, with file-level paired uncertainty. Additional complete
repetitions will be considered after the first run establishes computational
cost and completion rates.

### Metrics

Primary metrics are:

1. AUPRC; and
2. VUS-PR.

Secondary metrics include AUROC, VUS-ROC, standardized partial AUROC at 5%
false-positive rate where appropriate, fixed-budget event metrics, fitting
time, scoring time, throughput, and peak memory when measurable.

Aggregation will be reported at three levels:

1. individual trajectory;
2. source-family mean with paired uncertainty; and
3. unweighted source-family mean across the evaluation collection.

No method, hyperparameter, score direction, or implementation source will be
selected using evaluation labels.

## Output and Preservation

New artifacts will be written beneath a new versioned result root. Every run
will have a unique identity derived from dataset, trajectory, method,
implementation revision, hyperparameters, and seed. Existing result locations
will remain untouched.

The aggregation step will generate:

- per-run results;
- per-trajectory comparisons;
- source-family summaries;
- paired confidence intervals and effect sizes;
- completion and incompatibility reports;
- runtime and memory summaries; and
- manuscript-ready tables in separate generated files.

The paper will continue to show its existing tables until the new tables have
been reviewed. Replacement will be a separate explicit task.

## Validation and Testing

Tests will establish that:

1. registry entries require immutable revisions and complete provenance;
2. training prefixes exclude evaluation-only rows;
3. scaling uses training statistics exclusively;
4. label files are absent from external model input bundles;
5. score vectors align exactly with test rows;
6. nonfinite scores and partial outputs produce failed records;
7. explicit fallback selection produces a distinct implementation identity;
8. resume logic skips only complete matching runs;
9. PCA and KMeans scores match hand-computed fixtures;
10. TSB-AD-M manifest selection matches the official evaluation list; and
11. aggregation treats trajectories and source families according to the
    declared weighting rules.

A fake external adapter will exercise subprocess, timeout, failure, alignment,
and resume behavior without downloading third-party models. Each real external
adapter will then receive a one-trajectory integration smoke test before any
large run begins.

## Execution Sequence

1. Add the provenance registry, canonical bundle format, and validation tests.
2. Add PCA, KMeans, Euclidean-control, and one-regime-control adapters.
3. Add isolated external execution and the fake-adapter test fixture.
4. Integrate DAGMM and CATCH.
5. Integrate MTAD-GAT.
6. Integrate the three existing deep baselines from source-traceable
   implementations.
7. Add the official TSB-AD-M evaluation manifest and expanded experiment
   orchestration.
8. Run one-trajectory smoke tests for every method.
9. Run the current datasets and inspect score alignment and completion.
10. Run the complete TSB-AD-M evaluation split with resume enabled.
11. Aggregate results and compare them with the preserved historical tables.
12. Audit the manuscript methodology for conceptual clarity, procedural
    completeness, and consistency with the executed experiment contract.

## Methodology-Clarity Assessment

The final assessment will evaluate whether a reader can reconstruct the method
from the paper alone. It will examine:

1. the anomaly definition and intended deployment setting;
2. the role and empirical status of spherical-pendulum intuition;
3. representation, normalization, derivative, concept, and transition
   construction;
4. the mathematical definition and calibration of each score channel;
5. the relationship between the individual channels and the composite score;
6. training, validation, inference, and computational complexity;
7. causal and offline assumptions;
8. the relationship between the contextual figure and the formal anomaly
   definition;
9. dataset splitting, preprocessing, and metric computation; and
10. alignment among equations, pseudocode, implementation, and result tables.

The assessment will list concrete ambiguities with manuscript locations and
proposed corrections. Paper edits will follow only after the new experimental
evidence and the clarity assessment have been reviewed.

## Acceptance Criteria

The implementation phase is successful when:

1. every requested method has a validated adapter and provenance record;
2. all local tests pass;
3. every adapter completes a real one-trajectory smoke test or has a documented
   reproducible incompatibility;
4. the official TSB-AD-M evaluation manifest can be executed resumably;
5. new artifacts remain separate from historical results;
6. aggregate outputs can be regenerated from per-run records; and
7. the manuscript clarity assessment cites exact sections, equations, figures,
   and protocol statements.
