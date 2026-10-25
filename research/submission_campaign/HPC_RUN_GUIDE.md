# Matched held-out deep comparisons

These runners preserve the historical checkout and its Git history. No remote
job has been submitted by the assistant. The local campaign and manuscript
remain separate from the pending complete deep comparison.

## 1. Frozen scope

The source manifest is
`results/protocol_validation_20261024/manifests/502a1002c7788742def0a4bc13aeac1885b7376aec38ad5f3ddab3b07b2a0423.json`.
It selects 14 MSL, 25 SMAP, 20 SMD cases, BATADAL, and HAI for deep evaluation.
One SMD source is rejected because its training prefix contains anomalies.
The full six-method, three-seed matrix has 1,098 planned records: 1,080
eligible runs and 18 incompatible records. PowerSystem remains the separately reported local
diagnostic because source ordering and scenario independence are unverified.

The four GPU methods are MTAD-GAT, CATCH, TranAD, and Anomaly Transformer.
DAGMM and TSB-AD OmniAnomaly use the CPU runner. The current worker uses the
existing pinned implementations. Fresh-process import priming restores the
requested TranAD seed, and CATCH is explicitly seeded after upstream imports.
CATCH uses the reserved normal calibration portion for its validation set;
its upstream 80/20 split then exactly recovers the same fitting observations
as the local controls. No test labels are passed to a worker.

The deep comparison excludes the first and last 100 test rows from every
method, including the cached local controls. This avoids padded windows and
tails. Its numerical results must not be mixed directly with the earlier
local table, whose right trim is ten rows. Deep calibration-threshold and new
VUS results are separate pending work.

## 2. Transfer from the Mac

The generated `results/submission_push_20261024/submission-code-20261024.tar.gz`
contains code and the frozen source manifest. It excludes environments,
external repositories, raw data, prior result directories, and Git metadata.
Transfer it to AU:

```bash
scp /Users/robertocorizzo/git/pendulum/code/results/submission_push_20261024/submission-code-20261024.tar.gz \
  rcorizzo@prodhpcloginnode01.american.edu:/home/rcorizzo/
```

## 3. Create a separate AU working directory

Run on the AU login node. The directory is intentionally separate from
`/home/rcorizzo/pendulum`. Commands do not delete or replace its contents.
If the new directory or links already exist, inspect them before repeating.

```bash
module load git/2.40.0
mkdir /home/rcorizzo/pendulum-submission-20261024
tar -xzf /home/rcorizzo/submission-code-20261024.tar.gz \
  -C /home/rcorizzo/pendulum-submission-20261024
cd /home/rcorizzo/pendulum-submission-20261024
ln -s /home/rcorizzo/pendulum/.venv .venv
ln -s /home/rcorizzo/pendulum/benchmarks/official_baselines/.envs benchmarks/official_baselines/.envs
ln -s /home/rcorizzo/pendulum/benchmarks/official_baselines/.external benchmarks/official_baselines/.external
ln -s /home/rcorizzo/pendulum/benchmarks/real_world_datasets/data benchmarks/real_world_datasets/data
ln -s /home/rcorizzo/pendulum/benchmarks/tsb_ad_m/data benchmarks/tsb_ad_m/data
mkdir -p logs
export DYNSPHERE_REPO=/home/rcorizzo/pendulum-submission-20261024
.venv/bin/python -m research.submission_campaign.deep --dry-run \
  --methods mtad_gat catch tranad anomaly_transformer dagmm omni_anomaly_tsb_ad
```

Verify that the new worker manifest matches the local dry-run identifier in
`transfer-manifest.json`. Preflight enforces the actual `.external` Git
revisions and fingerprints tracked file contents and additional untracked
source files. Workers check this evidence again before importing the model;
changed source contents create a separate campaign identity. Source-file
hashes are checked when each case loads. A missing or
different data copy must be corrected before execution; do not replace the
frozen source with a different release.

Submit manually:

```bash
bsub < research/submission_campaign/au_h100.bsub
bsub < research/submission_campaign/au_cpu.bsub
```

To prioritize the two strongest relevant deep competitors first, replace the
GPU submission above with the following command (366 planned records,
including six incompatible records). This uses the same frozen identity.

```bash
DYNSPHERE_DEEP_METHODS="mtad_gat catch" bsub < research/submission_campaign/au_h100.bsub
```

After that array finishes, run the remaining two GPU methods without an
overlapping array:

```bash
DYNSPHERE_DEEP_METHODS="tranad anomaly_transformer" bsub < research/submission_campaign/au_h100.bsub
```

Each array has sixteen shards and a concurrency cap of four. Adjust the cap
to your allocated GPU share before submission if necessary. Do not run an
overlapping array against the same output directory while an earlier array
is active. The existing compute-node CUDA environment must pass the earlier
GPU compatibility probe.

## 4. PLGrid

Use the same code archive in a separate working directory. Reuse or create
Linux-compatible environments with the existing setup instructions; Mac
environments are not transferable. Link the corresponding data and pinned
source directories, set `DYNSPHERE_REPO`, and load the cluster's documented
Python/CUDA modules. Supply your allocation, partition, and QoS explicitly:

```bash
sbatch --account=YOUR_ALLOCATION --partition=YOUR_GPU_PARTITION \
  --qos=YOUR_QOS research/submission_campaign/plgrid_gpu.sbatch
```

The generic Slurm scripts require valid allocation and partition names. Use
the same account/partition/QoS options with
`research/submission_campaign/plgrid_cpu.sbatch` for DAGMM and OmniAnomaly.
The GPU script accepts the same `DYNSPHERE_DEEP_METHODS` priority setting.

## 5. Retrieve and aggregate

On the Mac, retrieve only the new output root:

```bash
rsync -av --progress \
  rcorizzo@prodhpcloginnode01.american.edu:/home/rcorizzo/pendulum-submission-20261024/results/submission_push_20261024/deep/ \
  /Users/robertocorizzo/git/pendulum/code/results/submission_push_20261024/deep/
rsync -av --progress \
  rcorizzo@prodhpcloginnode01.american.edu:/home/rcorizzo/pendulum-submission-20261024/results/submission_push_20261024/deep-manifests/ \
  /Users/robertocorizzo/git/pendulum/code/results/submission_push_20261024/deep-manifests/
```

Use the actual frozen deep-manifest directory in the aggregation command:

```bash
.venv/bin/python -m research.submission_campaign.aggregate \
  --deep-root results/submission_push_20261024/deep/DEEP_MANIFEST_PREFIX
```

Aggregation verifies the frozen deep manifest, configuration and upstream
content evidence, and independently reconstructs feature-array hashes from
each frozen source before accepting worker scores. It verifies score
artifacts, recomputes metrics on shared rows, and
uses common successful case-and-seed support before averaging. Keep failure
and incompatibility records. No table is automatically promoted to the paper.

## 6. Submission artifact checks

The transfer archive is a runner package, not the final public artifact.
Before submission, confirm the public artifact URL, author affiliations,
license and redistribution conditions, dataset acquisition steps, dependency
versions, raw-input hashes, worker configurations, and table-generation
commands. Include complete available result summaries and failures, and label
historical protocols and exploratory probes. Published raw data must follow
the upstream redistribution conditions. No online publication is performed here.
