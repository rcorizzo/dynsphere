# Exathlon, GHL, and GutenTAG extension

## Scope and protocol

This extension preserves the earlier campaigns. UCR is excluded. The first
stage uses the 25 Exathlon and 23 GHL files in the fixed TSB-AD-M evaluation
manifest. Additional local files outside that manifest are excluded by the
selection rule. These are TSB-AD versions; source-native Exathlon/GHL
validation remains a separate subsequent stage.

Training uses the filename-declared prefix. Only the held-out suffix is
passed to scoring and evaluation. Contaminated prefixes and nonfinite
observations are rejected without removing rows or joining unrelated
segments. Scaling is fitted on training data. No sequence concatenation,
decimation, per-source tuning, or test-label-based configuration selection is
performed. The first test row is excluded from every method's metrics because
the AR(1) control lacks a test predecessor. DynSphere retains centered
derivatives and propagation, so this is an offline evaluation.

The six default methods are DynSphere-S, DynSphere-E, DynSphere-S1,
DynSphere-E1, PCA reconstruction, and linear ridge AR(1). AR(1) fits the next
multivariate observation from its predecessor with ridge penalty 1. Its
unpropagated mean-squared residual is a conventional temporal control.
External baselines can be selected explicitly through `--methods` after the
baseline-fidelity audit. Their historical scores require compatibility
checks before reuse, including preprocessing, boundary context, and alignment.

The initial pilot uses pointwise AP, AUROC, and partial AUROC. It records
prevalence and cached scores. `--with-vus` additionally requires the official
VUS backend and can be considerably slower. The pilot's readiness flag stays
false, including when VUS is enabled; independent scientific review and the
expanded baseline comparison remain necessary before manuscript use.

## Local pilot

Run from the code repository:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
  .venv/bin/python -m benchmarks.official_baselines.extension_campaign
```

This selects the first two manifest files per family before examining model
performance: four cases, six methods, and seed 37, for 24 executions.
Use `--dry-run` to write the frozen selection without fitting models.

Outputs are isolated under `results/extension_v1/`. Each run retains inputs,
private labels, score vectors, input digests, and implementation metadata.
The extension fingerprints actual source contents, including uncommitted
changes. Summary directories are separated by campaign identity. Repeating
an identical invocation resumes validated scores. Changes to inputs or source
produce distinct identities. A zero-execution scheduler shard is permitted.

## Official GutenTAG generation

The generator has its own environment and does not change the detector environment:

```bash
.venv/bin/python -m venv benchmarks/official_baselines/.envs/gutentag
benchmarks/official_baselines/.envs/gutentag/bin/python -m pip install timeeval-gutenTAG==1.5.0
benchmarks/official_baselines/.envs/gutentag/bin/python \
  -m benchmarks.official_baselines.prepare_gutentag_extension \
  --output-dir data/gutentag_extension_v1
```

The output directory must be new. Generation creates separate normal training
and anomalous test files. Six mechanisms (frequency, pattern shift, mode
correlation, amplitude, mean, and variance) use three observed channels and
generation seeds 101, 102, and 103. The fixed initial configuration uses 8,000
rows and one injected event per test case. It covers one severity/base design
per mechanism and provides no evidence of broad severity robustness. All
generated cases, including unfavorable results, remain in the analysis.

The manifest records generation seeds, file hashes, configuration, and package
version. The exact generator environment is saved with the data. Detector
seeds and generation seeds are distinct. Replicate seeds share a mechanism
and should not be interpreted as independent real deployments.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1 \
  .venv/bin/python -m benchmarks.official_baselines.extension_campaign \
  --families GutenTAG \
  --native-manifest data/gutentag_extension_v1/native_manifest.json
```

This runs all 18 generated cases with six methods at detector seed 37 (108
executions). The per-family case limit applies only to TSB-AD files; it never
silently restricts a supplied native manifest.

## AU and PLGrid runners

Transfer the code, selected TSB files, and generated GutenTAG folder to the
HPC. Include the Git metadata or a committed source snapshot for provenance.
The generator environment is unnecessary on the HPC when generated data are
transferred. Use an existing compatible detector environment. Create `logs/`
before submission and export `DYNSPHERE_REPO` to the code checkout.

AU (LSF):

```bash
bsub < benchmarks/official_baselines/hpc/au_lsf_extension_cpu_array.bsub
```

PLGrid (SLURM), replacing the allocation and partition values with valid
cluster-specific values:

```bash
sbatch --account=YOUR_ALLOCATION --partition=YOUR_CPU_PARTITION \
  benchmarks/official_baselines/hpc/plgrid_extension_cpu_array.sbatch
```

Both runners use CPU execution, 16 shards, and at most four concurrent array
elements. They limit BLAS/OpenMP threads to the requested CPUs. They select
all 48 Exathlon/GHL cases and 18 GutenTAG cases at detector seeds 37, 38, 39:
66 cases × 6 methods × 3 seeds = 1,188 planned executions. No jobs are
submitted automatically. This is the local-method campaign; deep baselines
are added after their fidelity checks.

The runner supports `DYNSPHERE_PYTHON`, `DYNSPHERE_EXTENSION_RESULT_ROOT`,
`DYNSPHERE_NATIVE_MANIFEST`, `DYNSPHERE_TSB_DATA_ROOT`, and `DYNSPHERE_METHODS`.
For a scheduler-free manifest check, execute the shared runner with
`--dry-run`; without scheduler variables it uses one shard. `--with-vus` can
also be passed to the shared runner. Its default metrics remain pointwise.

Each shard writes separate metric, family-summary, and completion files.
Shard-local family summaries are diagnostic. For the final comparison,
combine per-run metrics from all shards, verify coverage against the frozen
manifest, average detector seeds within case, then cases within source.
Report the three sources separately, with generation mechanism breakdowns for
GutenTAG. Scientific conclusions must use the full selected workload.

## Source-native follow-up

The loader also accepts TimeEval-style paired CSV files through a JSON
manifest containing `cases`, with fields `family`, `case_name`, `train`, and
`test`. Paths are relative to the manifest. Use separate, verified normal
training files and preserve original application/trace identities. The loader
excludes timestamps from model features and requires identical feature order.
Obtaining and auditing these source-native releases is subsequent work;
the current pilot must not be described as evaluating their complete releases.

Sources: [Exathlon](https://github.com/exathlonbenchmark/exathlon),
[TimeEval datasets](https://timeeval.github.io/evaluation-paper/notebooks/Datasets.html),
[official GutenTAG](https://github.com/TimeEval/GutenTAG).
