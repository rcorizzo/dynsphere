# Extension pilot: October 22, 2026

The subsequent full 1,188-configuration local-method campaign is now complete.
See [the full-workload report](/Users/robertocorizzo/git/pendulum/code/benchmarks/official_baselines/EXTENSION_RESULTS_20261022.md).
The results below retain the original pilot scope.

## Scope and completion

All 132 planned pilot executions completed with finite, aligned score vectors
and pointwise metrics. The pilot uses detector seed 37. Exathlon and GHL each
contribute the first two files in the fixed TSB-AD-M manifest. GutenTAG uses
all six predefined mechanisms at generation seeds 101, 102, and 103, giving
18 separately generated train/test pairs. The total is 22 cases × 6 methods.

Exathlon/GHL train on clean declared prefixes and evaluate held-out suffixes.
GutenTAG uses separate normal training files. All methods exclude the first
test row from metrics. Training and evaluation sequences are never joined.
The study uses the unchanged DynSphere scoring weights and concept counts.
The AR(1) control has ridge penalty 1 and no temporal propagation.

## Preliminary average precision

These values are trajectory means within source, including generation
replicates for GutenTAG. They have no inferential intervals and are outside
the manuscript tables.

| Method | Exathlon (2 traces) | GHL (2 traces) | GutenTAG (18 generated cases) |
|---|---:|---:|---:|
| DynSphere-S | 0.8992 | 0.0078 | 0.2657 |
| DynSphere-E | 0.7286 | 0.0066 | 0.3706 |
| DynSphere-S1 | 0.9019 | 0.0091 | 0.1399 |
| DynSphere-E1 | 0.9939 | 0.0066 | 0.3230 |
| PCA reconstruction | 0.9853 | 0.0065 | 0.2352 |
| Linear ridge AR(1) | 0.9477 | 0.0116 | 0.3153 |
| Mean anomaly prevalence | 0.1368 | 0.0123 | 0.0500 |

The prevalence row provides context for precision-recall interpretation. It
describes the expected precision level of a random ranking asymptotically;
finite-sample expected average precision can differ.

## Interpretation

1. DynSphere-E1 performs well on the two Exathlon traces. PCA also has high
   average precision, indicating that these examples provide limited evidence
   for an advantage specific to motion scoring. The remaining 23 selected
   traces must remain in the expanded comparison.
2. The first two GHL traces are challenging for all six methods. DynSphere's
   low average precision and AUROC require reporting and investigation. The
   collection remains in scope, with the original settings preserved.
3. Four-concept DynSphere-E has the highest GutenTAG average precision in
   this six-method pilot. The one-concept version performs less favorably in
   this controlled study, illustrating a source-dependent concept-count effect.
4. The mechanism results are heterogeneous. Variance anomalies strongly favor
   the Euclidean DynSphere configurations (approximately 0.982 average
   precision). AR(1) leads the frequency-anomaly comparison (0.191). PCA
   perfectly detects this mode-correlation configuration, so that mechanism
   does not establish observed-space marginal plausibility. Pattern-shift
   average precision remains modest (best value approximately 0.094).
5. No deep baseline is included in this pilot. The results support pipeline
   feasibility and a broader comparison. They do not establish state-of-the-art
   performance, robustness across anomaly severity, or unseen real deployments.

## Reproducibility

The official `timeeval-gutenTAG==1.5.0` generator runs in an isolated
environment. Generation records include the full configuration, dependencies,
generation seeds, and train/test hashes. All 18 test files have distinct
content hashes. The initial import emitted font-cache warnings; generation
completed successfully without source changes to GutenTAG.

Frozen pilot manifests and metrics:

- [Exathlon/GHL manifest](/Users/robertocorizzo/git/pendulum/code/results/extension_v1/campaigns/5b4a6ef50915.json)
- [Exathlon/GHL metrics](/Users/robertocorizzo/git/pendulum/code/results/extension_v1/summaries/5b4a6ef50915/point_metrics-shard-000.csv)
- [GutenTAG manifest](/Users/robertocorizzo/git/pendulum/code/results/extension_v1/campaigns/8e0f2d6c05a3.json)
- [GutenTAG metrics](/Users/robertocorizzo/git/pendulum/code/results/extension_v1/summaries/8e0f2d6c05a3/point_metrics-shard-000.csv)

Each run also retains score vectors and a provenance record. Source-content
fingerprints supplement the current Git revision because the implementation
has uncommitted changes. The result directories are ignored by Git and must
be transferred separately when sharing the experiment artifact.

Independent code review identified four provenance/interpretation checks.
Regression tests now cover native generator/hash validation, canonical method
ordering for scheduler shards, cache invalidation after label corrections,
and preservation of the pilot-only flag when official VUS metrics succeed.
The pilot was repeated after these corrections; the reported metrics were
unchanged. Earlier pilot records remain preserved.

## Next stage

The AU LSF and PLGrid SLURM CPU runners select 25 Exathlon traces, 23 GHL
traces, and 18 GutenTAG cases. With six methods and detector seeds 37, 38,
and 39, the declared workload contains 1,188 executions. Their manifests can
be checked without executing models. No HPC jobs have been submitted.

Complete the local-method matrix, then add audited external baselines and
official VUS metrics. Source-native Exathlon/GHL releases remain a separate
validation stage. An existing-baseline score can be reused only after
checking preprocessing, split, boundary context, and row alignment.

The earlier 2,457-execution campaign and all manuscript tables are preserved.
UCR is excluded from this extension.
