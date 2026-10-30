# Separate PCA-95 accuracy sensitivity study

This study preserves the current DynSphere detector, all original baseline
results and the CPU quality–cost campaign. It evaluates additional dimensions
and optional PCA input preprocessing. Its results are exploratory and are
stored separately under `results/pca95_sensitivity_20261026/`.

## Variants and scope

The dimension rule is the smallest count reaching 95% cumulative training
variance, with a five-component minimum and a cap at the input dimension.
PCA uses fitting observations only, after a fitting-only StandardScaler, with
no whitening. Calibration/test data use the same fitted transformation.

| Group | Methods | Executions |
| --- | --- | ---: |
| `local` | DynSphere-E1/E4/S1/S4, KMeans, ridge AR(1) | 1,080 |
| `external_cpu` | DAGMM (TAB), OmniAnomaly (TSB-AD) | 360 |
| `gpu` | CATCH, MTAD-GAT (ML4ITS), Anomaly Transformer, TranAD | 720 |
| Total | 12 additional configurations | 2,160 |

Each method uses 60 eligible cases across BATADAL, HAI-20.07, MSL, SMAP and
SMD, with seeds 37–39. Primary summaries retain the original 59-case matched
cohort. The extra eligible SMAP case remains available separately. The
contaminated SMD prefix remains excluded.

For DynSphere, only the internal PCA dimension changes. The original-feature
acceleration branch still receives the original standardized features. For
the competitors, PCA is an added input preprocessing step; their existing
architectures and training budgets remain fixed. These are labeled
`<method>+PCA95-min5` and are compared to their original versions. The
original PCA reconstruction baseline already selects 95% variance and is
not duplicated. Its existing dimension rule has no five-component minimum.

Upstream implementation limitations remain, including OmniAnomaly's separate
partition normalization. DAGMM uses a method-specific dependency overlay
under `dependencies/dagmm-overlay`; its runtime must be prepared separately
using the existing recovery instructions. No baseline repair occurs here.

## Planning and execution protection

The metadata-only planning step uses the verified PCA diagnostic and existing
primary comparison. These files are required at their original result paths:

- `results/pca_diagnostics_20261026/fitting-only-v1/per_case.csv`
- `results/protocol_validation_20261024/manifests/502a1002c7788742def0a4bc13aeac1885b7376aec38ad5f3ddab3b07b2a0423.json`
- `results/submission_gpu_l40_20261025/vus_20261025/per_run_matched.csv`
- `results/submission_recovery_20261024/deployment-evidence/runtime-evidence.json`

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m research.pca95.run \
  --group all --dry-run
```

The existing timing campaign must finish all 135 outcomes before any new
input preparation or model training. The runner requires its terminal
`completion.json`, including the frozen timing identity. For external hosts,
transfer this proof after completion. There is no command-line bypass.

After that proof exists, run the local group sequentially:

```bash
export DYNSPHERE_PCA95_REPO="$PWD"
bash research/pca95/run_group.sh local
```

The first execution prepares all inputs and binds their metadata/hashes to
an immutable execution manifest. Every worker verifies that manifest, local
and upstream source hashes, selected dimensions and input hashes. Completed
worker artifacts can be recovered after interruption; old outcomes and
failed attempts are preserved. Stop previous workers before resuming. Each
worker has a 12-hour safety limit and retains the full original epoch budget.

## AU HPC groups

Use the existing verified checkout with the same datasets, upstream sources,
method environments and DAGMM compatibility overlay. Transfer the new module,
planning manifest, referenced small files and timing completion proof. Run a
dry-run to verify the local source hashes before submitting. If hashes differ,
preserve the existing folder and create a separately identified campaign;
do not combine different manifests into one result directory.

Prepare the inputs once after timing completion, then transfer the complete
prepared `inputs/`, `execution-manifest.json`, and `manifest.json` to each
execution host. Keep those arrays byte-identical. Independent PCA preparation
on different numerical stacks can produce different hashes and incompatible
execution identities. Group-specific results can be combined only when their
execution manifest matches. CPU deep workers additionally verify the original
recovery runtime versions and DAGMM overlay hash before fitting.

The preparation-only command is:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m research.pca95.run --prepare-only
```

It is subject to the same timing-completion guard and starts no model training.

```bash
module load git/2.40.0
export DYNSPHERE_PCA95_REPO="$PWD"
mkdir -p logs
bash research/pca95/run_group.sh external_cpu --dry-run
bash research/pca95/run_group.sh gpu --dry-run
bsub < research/pca95/au_cpu_array.bsub
bsub < research/pca95/au_l40_array.bsub
```

`au_local_array.bsub` is an alternative to laptop execution for the local
group. Submit only one intended execution route; per-run locks protect against
accidental duplicate workers. The templates do not alter existing jobs.

## Analysis

```bash
.venv/bin/python -m research.pca95.aggregate
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m research.pca95.aggregate --vus
```

The second command uses official `vus==0.0.6` on cached scores, separately
from model execution. It retains the primary study's label-derived VUS window
rule and the common 100/100 test boundary exclusion. Buffered fallback
values are rejected. Identical score/label inputs share a validated cache.

Pointwise metrics are recomputed from stored scores. Each new method is
paired with its original version on identical cases and seeds. Five-family
means are suppressed until all 59 primary cases and all three seeds are
available for that variant. Every failure, timeout and missing execution is
retained in the completion report. Source-specific switching and selecting
the best configuration from test results are outside this protocol.
