# Extension campaign: manual HPC execution

## 1. Scope

The completed local comparison contains 1,188 configurations (six methods,
66 cases, three detector seeds). KMeans adds 198 inexpensive configurations.
The remaining planned detector fits are:

| Group | Methods | Planned fits |
|---|---|---:|
| GPU | CATCH, MTAD-GAT, Anomaly Transformer, TranAD | 792 |
| External CPU | DAGMM, TSB-AD OmniAnomaly | 396 |
| Neural predictor | Single-regime Euclidean predictor | 198 |
| Total | Seven methods | 1,386 |

The laptop predictor pilot was stopped after timing indicated an hours-long
full workload. Its completed records are preserved. Counts above describe
the full groups; an identical source/configuration snapshot can reuse exact
completed records. The extended comparison has 14 methods and 2,772 planned
fits in total. Component-study metric configurations are counted separately.

The real-data VUS stage covers 1,008 existing seven-method records. It fits
no models. The corresponding GutenTAG stage covers 378 records and was run
locally. Metric-only KMeans artifacts have been staged alongside the first
six methods, without replacing their original records.

No HPC jobs have been submitted by this work. The commands below are manual.

## 2. Transfer from the MacBook

These transfers preserve both repositories' `.git` directories. They do not
fetch, merge, push, or replace Git history. The source transfer uses a remote
backup directory to preserve overwritten files. First inspect the dry-run:

```bash
rsync -avn --backup --backup-dir=../pendulum-source-backup-20261022 \
  --exclude='.git/' --exclude='.venv/' --exclude='.envs/' \
  --exclude='.external/' --exclude='__pycache__/' \
  --exclude='results/' --exclude='logs/' --exclude='data/' \
  /Users/robertocorizzo/git/pendulum/code/ \
  rcorizzo@prodhpcloginnode01.american.edu:/home/rcorizzo/pendulum/
```

Run the same command with `-av` after checking the file list. Do not add
`--delete`. Transfer generated data independently:

```bash
rsync -av /Users/robertocorizzo/git/pendulum/code/data/gutentag_extension_v1/ \
  rcorizzo@prodhpcloginnode01.american.edu:/home/rcorizzo/pendulum/data/gutentag_extension_v1/
```

The HPC already had the TSB-AD-M collection for the previous campaign.
The dry-run in Section 3 checks presence and uniqueness of the 25 Exathlon
and 23 GHL files selected by the fixed evaluation manifest.

For VUS, transfer the frozen summary and only the required cached artifacts.
This avoids copying the much larger duplicated model-input arrays:

```bash
rsync -av /Users/robertocorizzo/git/pendulum/code/results/extension_v1/summaries/seven-methods-20261022/combined/ \
  rcorizzo@prodhpcloginnode01.american.edu:/home/rcorizzo/pendulum/results/extension_v1/summaries/seven-methods-20261022/combined/
rsync -av --include='*/' --include='run.json' --include='scores.npy' \
  --include='labels.npy' --exclude='*' \
  /Users/robertocorizzo/git/pendulum/code/results/extension_v1/runs/ \
  rcorizzo@prodhpcloginnode01.american.edu:/home/rcorizzo/pendulum/results/extension_v1/runs/
```

## 3. AU preflight and submission

On the AU login node:

```bash
cd /home/rcorizzo/pendulum
module load git/2.40.0
export DYNSPHERE_REPO=/home/rcorizzo/pendulum
mkdir -p logs
.venv/bin/python -c 'from benchmarks.official_baselines.extension_campaign import source_digest; print(source_digest())'

DYNSPHERE_EXTENSION_GROUP=gpu bash benchmarks/official_baselines/hpc/run_extension_baselines.sh --dry-run
DYNSPHERE_EXTENSION_GROUP=cpu bash benchmarks/official_baselines/hpc/run_extension_baselines.sh --dry-run
DYNSPHERE_EXTENSION_GROUP=predictor bash benchmarks/official_baselines/hpc/run_extension_baselines.sh --dry-run
bash benchmarks/official_baselines/hpc/run_extension_metrics.sh --dry-run
```

Expect planned counts 792, 396, 198, and 1,008 metric records, respectively.
The transferred source digest should be
`22357e3c923ef3219421f06a4eb49fd40e0d211347f4c2573994e5f083f4f28c`.
If it differs, inspect missing, changed, or additional Python sources before
submission. Preserve any remote-only work while resolving the difference.
Dry-runs validate selection and prepare manifests; they do not test GPU
availability or baseline dependencies. Reuse the working isolated baseline
environments from the previous AU campaign. If these environments changed,
repeat the existing GPU/dependency probe before scheduling the full arrays.

Submit each group once:

```bash
bsub < benchmarks/official_baselines/hpc/au_lsf_extension_h100_array.bsub
bsub < benchmarks/official_baselines/hpc/au_lsf_extension_deep_cpu_array.bsub
bsub < benchmarks/official_baselines/hpc/au_lsf_extension_predictor_array.bsub
bsub < benchmarks/official_baselines/hpc/au_lsf_extension_metrics_array.bsub
```

The first array uses H100. The other arrays use the long CPU queue.
Each array has 16 shards and allows at most four simultaneous elements.
Numerical-library threads are capped, including for VUS. A different CPU
queue can be selected with the scheduler's command-line queue override.

Once complete, transfer `results/extension_baselines_v2/` and
`results/extension_v1/vus/seven-methods-20261022/real/` back for coverage checks and
paired comparison. Preserve non-complete records and their execution logs.
The old general aggregator does not implement this extension's evaluation
support and must not be applied unchanged.

## 4. PLGrid

Use the same source, selected TSB data, generated data, and compatible
isolated environments. Set `DYNSPHERE_REPO` to the PLGrid checkout. Replace
allocation and partition placeholders with approved cluster values:

```bash
DYNSPHERE_EXTENSION_GROUP=gpu sbatch --account=YOUR_ALLOCATION \
  --partition=YOUR_GPU_PARTITION --gres=gpu:1 \
  benchmarks/official_baselines/hpc/plgrid_extension_baselines_array.sbatch
DYNSPHERE_EXTENSION_GROUP=cpu sbatch --account=YOUR_ALLOCATION \
  --partition=YOUR_CPU_PARTITION \
  benchmarks/official_baselines/hpc/plgrid_extension_baselines_array.sbatch
DYNSPHERE_EXTENSION_GROUP=predictor sbatch --account=YOUR_ALLOCATION \
  --partition=YOUR_CPU_PARTITION \
  benchmarks/official_baselines/hpc/plgrid_extension_baselines_array.sbatch
sbatch --account=YOUR_ALLOCATION --partition=YOUR_CPU_PARTITION \
  benchmarks/official_baselines/hpc/plgrid_extension_metrics_array.sbatch
```

Use one cluster per group to avoid duplicate fits. GPU resource syntax and
time limits can vary by PLGrid partition; validate the allocation before
submission. No allocation or partition name is invented in these runners.

## 5. Audited configurations and interpretation

TranAD `audited-v2` fits the released featurewise min-max rule to training
data only (epsilon 1e-4), includes each scored row as its reconstruction
target, uses sequential batches, and restores the released StepLR schedule.
It retains five training epochs. Test observations are not clipped to the
training range. The historical configuration remains available.

Anomaly Transformer `released-budget-v2` uses width 512, eight heads, ten
epochs, and the released learning-rate schedule. Batch size remains 128 for
memory control. The final fixed-budget checkpoint is used, with no test-set
checkpoint selection. Scores use the last position of overlapping windows.
This is an explicitly disclosed adapter configuration; exact equivalence
to the authors' complete training and non-overlapping evaluation pipeline
is not claimed. Record loss histories and inspect nonfinite or divergent
training before interpreting results.

The shared first test row is excluded. Canonical adapter padding is retained
for longer model-specific warmups. A common 100-row-exclusion sensitivity
analysis is required for the final deep-baseline table because MTAD-GAT has
a 100-row lookback. Scores are retained so this check requires no refitting.

Metric output preserves fitting `elapsed_seconds` and separately records
`metric_elapsed_seconds`. Reused VUS results share a computation, so metric
time must be summed over distinct metric keys. Input seals detect later
score/label mutation. The original table lacked score hashes; initial
historical integrity can be checked only against available identities and
pointwise metrics. Readiness stays false until scientific review.
