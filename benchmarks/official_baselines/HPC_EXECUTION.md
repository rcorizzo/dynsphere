# Portable Official-Baseline Execution

This guide runs the same resumable experiment campaign on a MacBook,
American University Lovelace, and PLGrid. The Python campaign runner defines
the experiment matrix and shard membership. The scheduler scripts only request
resources and supply the array index.

## 1. Execution Stages

| Profile | Methods | Data | Seeds | Intended system |
|---|---|---|---|---|
| `macbook-safe` | DAGMM, TranAD, TSB-AD OmniAnomaly fallback | MSL, SMAP, SMD | 37 | MacBook |
| `hpc-seed37` | Six external implementations, including the explicit TSB-AD OmniAnomaly fallback | Current real data and complete TSB-AD-M manifest | 37 | AU or PLGrid |
| `hpc-followup` | Same external implementations | Current real data | 38, 39 | AU or PLGrid after method selection |
| `paper-multiseed` | 13 final comparison methods | Seven paper dataset groups | 37, 38, 39 | AU or PLGrid |

The `paper-multiseed` profile contains 63 cases for each method and seed. Its
13 methods and three seeds produce 2,457 logical runs. Execution is divided
into 756 H100 runs, 378 external-CPU runs, and 1,323 local-CPU runs. The three
component manifests are merged before aggregation.

The preferred OmniAnomaly implementation remains explicitly separate from
`omni_anomaly_tsb_ad`. An incompatible preferred implementation produces a
structured terminal record. No fallback is selected automatically.

## 2. Shared Prerequisites

The repository and all datasets must be visible from every compute node. Store
the result root on shared project storage. Node-local scratch is suitable for
temporary files, but it must not hold the only copy of result records.

Transfer the repository and data with a resumable command such as:

```bash
rsync -avP --exclude '.venv' --exclude 'benchmarks/official_baselines/.envs' \
  --exclude 'benchmarks/official_baselines/.external' \
  /local/path/pendulum/ USER@LOGIN_HOST:/shared/path/pendulum/
```

On the cluster, define the common paths:

```bash
export DYNSPHERE_REPO=/shared/path/pendulum
export DYNSPHERE_TSB_DATA_ROOT=/shared/path/pendulum/benchmarks/tsb_ad_m/data/TSB-AD-M
export DYNSPHERE_RESULT_ROOT=/shared/path/pendulum/results/official_baselines_v1
mkdir -p "$DYNSPHERE_REPO/logs" "$DYNSPHERE_RESULT_ROOT"
```

The environment setup downloads pinned source repositories and Python
packages. Run it through the scheduler on a compute node. If compute nodes
cannot reach package repositories, ask the local support team for the approved
package-cache or proxy procedure.

## 3. MacBook-Safe Stage

Prepare the environments as described in the main baseline README, then inspect
the 177-job matrix:

```bash
.venv/bin/python -m benchmarks.official_baselines.run_campaign \
  --profile macbook-safe \
  --dry-run \
  --result-root results/official_baselines_v1
```

Run the stage sequentially:

```bash
.venv/bin/python -m benchmarks.official_baselines.run_campaign \
  --profile macbook-safe \
  --result-root results/official_baselines_v1 \
  --resume
```

The `--resume` flag validates completed score vectors and skips only exact,
complete run identities.

## 4. American University Lovelace, LSF

The [current AU procedures](https://www.american.edu/cas/hpc/procedures.cfm)
require batch scheduling and prohibit computation on login nodes. The
[system specification](https://www.american.edu/cas/hpc/software.cfm) lists
dual H100 GPUs and L40 GPU nodes. The files below follow the LSF conventions
used by the Harmonic Loss project.

### 4.1 Materialize the repository

Run these commands on the MacBook. They create `~/pendulum` on Lovelace and
copy the source code and datasets. Generated environments, historical results,
and Git internals are excluded.

```bash
cd /Users/robertocorizzo/git/pendulum
ssh rcorizzo@prodhpcloginnode01.american.edu 'mkdir -p "$HOME/pendulum/logs"'
rsync -avP \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'results' \
  --exclude 'logs' \
  --exclude 'benchmarks/official_baselines/.envs' \
  --exclude 'benchmarks/official_baselines/.external' \
  code/ rcorizzo@prodhpcloginnode01.american.edu:pendulum/
```

Log in and define the paths inherited by every submitted job:

```bash
ssh rcorizzo@prodhpcloginnode01.american.edu
cd "$HOME/pendulum"
export DYNSPHERE_REPO="$HOME/pendulum"
export DYNSPHERE_TSB_DATA_ROOT="$DYNSPHERE_REPO/benchmarks/tsb_ad_m/data/TSB-AD-M"
export DYNSPHERE_RESULT_ROOT="$DYNSPHERE_REPO/results/official_baselines_v1"
export DYNSPHERE_PROFILE=hpc-seed37
export DYNSPHERE_SHARD_COUNT=16
export DYNSPHERE_GIT_MODULE=git/2.40.0
mkdir -p logs "$DYNSPHERE_RESULT_ROOT"
```

These exports apply to the current login session. Define them again after a
new login before using `bsub`.

### 4.2 Build the environments

Submit setup to a CPU node:

```bash
bsub < benchmarks/official_baselines/hpc/au_lsf_setup.bsub
bjobs
```

After the setup job finishes, inspect both logs and the pinned-environment
report:

```bash
tail -n 100 logs/au-setup-*.out
tail -n 100 logs/au-setup-*.err
.venv/bin/python -m json.tool \
  results/official_baselines_v1/setup/provenance_report.json | less
```

Every method intended for execution must have status `ready`. An incompatible
preferred OmniAnomaly environment is expected; the CPU campaign explicitly
uses the registered TSB-AD fallback.

### 4.3 Validate GPU execution

Submit one probe. H100 is recommended because its exact scheduler model token
is already established:

```bash
bsub < benchmarks/official_baselines/hpc/au_lsf_h100_probe.bsub
```

The alternative L40 probe uses the normal queue and requests one available
GPU, consistent with AU's published queue description:

```bash
bsub < benchmarks/official_baselines/hpc/au_lsf_l40_probe.bsub
```

The probe checks `nvidia-smi`, verifies CUDA in every isolated PyTorch
environment, and evaluates one contextual case with CATCH, MTAD-GAT, Anomaly
Transformer, and TranAD. Inspect its log and recorded devices:

```bash
bjobs
tail -n 200 logs/au-*-probe-*.out
tail -n 200 logs/au-*-probe-*.err
grep -R '"device": "cuda"' \
  results/official_baselines_v1/au-gpu-probe
```

Continue only when all four probe runs complete and record CUDA.

### 4.4 Submit the seed-37 campaign

The CPU and GPU arrays share a result root and use disjoint method lists.
Their manifests are separate, which prevents concurrent manifest writes from
conflicting.

Submit the CPU methods:

```bash
bsub < benchmarks/official_baselines/hpc/au_lsf_cpu_array.bsub
```

Submit exactly one GPU array. H100 is the recommended default:

```bash
bsub < benchmarks/official_baselines/hpc/au_lsf_h100_array.bsub
```

Use the following command only when the L40 probe succeeded and H100 is not
being used for the same campaign:

```bash
bsub < benchmarks/official_baselines/hpc/au_lsf_l40_array.bsub
```

The CPU array contains 368 jobs before sharding. The GPU array contains 736
jobs before sharding. Each uses 16 deterministic array elements. Completed
run identities are validated and skipped during resumption.

### 4.5 Monitor and resume

```bash
bjobs
bjobs -l JOB_ID
tail -f logs/au-h100-JOB_ID-ARRAY_INDEX.out
```

Cancel only the intended job or array:

```bash
bkill JOB_ID
```

Submitting the same file again resumes the campaign. To repeat selected H100
elements while preserving the 16-way partition:

```bash
export DYNSPHERE_SHARD_COUNT=16
bsub -J 'dynsphere-h100[2,7,11]' \
  < benchmarks/official_baselines/hpc/au_lsf_h100_array.bsub
```

Use the corresponding CPU or L40 file when resuming those resources.

#### CATCH frequency-overlap correction

The pinned author implementation computes an overlap denominator from nonzero
loss values. Exact-zero reconstruction losses can therefore create a zero
denominator and nonfinite frequency scores. The repository adapter uses patch
coverage counts and records `frequency_overlap_correction=coverage-count-v1`
in each corrected CATCH run.

Use the dedicated array to evaluate the corrected configuration across every
seed-37 CATCH case:

```bash
bsub < benchmarks/official_baselines/hpc/au_lsf_h100_catch_corrected_array.bsub
```

This array writes
`campaigns/hpc-seed37-au-catch-overlap-v1.json`. Its distinct configuration
changes every CATCH run identity, so completed CATCH runs are evaluated again.
Results from the other GPU methods remain unchanged.

### 4.6 Aggregate the complete campaign

After both arrays finish, submit the aggregation job:

```bash
bsub < benchmarks/official_baselines/hpc/au_lsf_aggregate.bsub
```

This job creates a combined six-method manifest and writes the readiness report
and manuscript tables under `results/official_baselines_v1/aggregate/`.

### 4.7 Retrieve results

Run these commands on the MacBook:

```bash
cd /Users/robertocorizzo/git/pendulum/code
mkdir -p logs/hpc/au
rsync -avP \
  rcorizzo@prodhpcloginnode01.american.edu:pendulum/results/official_baselines_v1/ \
  results/official_baselines_v1/
rsync -avP \
  rcorizzo@prodhpcloginnode01.american.edu:pendulum/logs/ \
  logs/hpc/au/
```

### 4.8 Submit the final three-seed paper campaign

Use one committed repository revision for all three components. Confirm the
revision and external-source provenance before submission:

```bash
cd "$DYNSPHERE_REPO"
git rev-parse HEAD
.venv/bin/python -m json.tool \
  "$DYNSPHERE_RESULT_ROOT/setup/provenance_report.json" | less
```

Validate the full matrix in a temporary directory:

```bash
DRY_DIRECTORY=$(mktemp -d)
.venv/bin/python -m benchmarks.official_baselines.run_campaign \
  --profile paper-multiseed \
  --campaign-manifest "$DRY_DIRECTORY/paper-multiseed.json" \
  --result-root "$DRY_DIRECTORY/results" \
  --dry-run
rm "$DRY_DIRECTORY/paper-multiseed.json"
rmdir "$DRY_DIRECTORY"
```

The summary must report 2,457 jobs, 189 jobs for each method, and 819 jobs for
each seed. Submit the three disjoint arrays after the H100 probe records CUDA:

```bash
bsub < benchmarks/official_baselines/hpc/au_lsf_multiseed_h100_array.bsub
bsub < benchmarks/official_baselines/hpc/au_lsf_multiseed_cpu_array.bsub
bsub < benchmarks/official_baselines/hpc/au_lsf_multiseed_local_array.bsub
```

Monitor the arrays and inspect their outcome summaries:

```bash
bjobs
bjobs -a | grep 'dynsphere-paper'
grep -H -A5 '"outcomes"' logs/au-paper-*.out
find logs -name 'au-paper-*.err' -type f -size +0 -print
```

Each rerun uses the existing component manifest. A changed method set or
repository revision is rejected at the same manifest path. Resubmit selected
array elements with the original shard count, for example:

```bash
bsub -J 'dynsphere-paper-h100[3,10]' \
  < benchmarks/official_baselines/hpc/au_lsf_multiseed_h100_array.bsub
bsub -J 'dynsphere-paper-local[4,19]' \
  < benchmarks/official_baselines/hpc/au_lsf_multiseed_local_array.bsub
```

After all three components report complete terminal records, submit the merge
and aggregation job:

```bash
bsub < benchmarks/official_baselines/hpc/au_lsf_multiseed_aggregate.bsub
```

The merged manifest is
`campaigns/paper-multiseed-au-complete.json`. Aggregation writes the standard
summaries, `geometry_factorial.csv`, and the strict readiness report.

## 5. PLGrid, SLURM

PLGrid grants use an account name with a resource suffix. Examples include
`GRANT-cpu` on Ares or Helios, `GRANT-gpu` on Ares,
`GRANT-gpu-a100` on Athena, and `GRANT-gpu-gh200` on Helios. Confirm the exact
account in the PLGrid Portal.

The initial campaign is CPU-oriented. Suitable starting partitions include
`plgrid-testing` for a short validation and `plgrid` for the full array on
Ares or Helios. A pilot grant has a 72-hour job limit.

### 5.1 Environment setup

Load the Python module name shown by the selected cluster's application
catalogue. Submit setup through SLURM:

```bash
cd "$DYNSPHERE_REPO"
mkdir -p logs
export DYNSPHERE_PYTHON_MODULE=YOUR_CURRENT_PYTHON_MODULE
sbatch -A GRANT-cpu -p plgrid \
  --job-name=dynsphere-setup --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=32G --time=08:00:00 \
  --output=logs/plgrid-setup-%j.out --error=logs/plgrid-setup-%j.err \
  --wrap="bash '$DYNSPHERE_REPO/benchmarks/official_baselines/hpc/setup_environment.sh'"
```

Use `plgrid-testing` with a one-hour limit for a small installation test when
the dependency installation is known to fit that limit.

### 5.2 Seed-37 array

```bash
export DYNSPHERE_PROFILE=hpc-seed37
export DYNSPHERE_SHARD_COUNT=16
sbatch -A GRANT-cpu -p plgrid \
  benchmarks/official_baselines/hpc/plgrid_slurm_array.sh
```

For 32 shards:

```bash
export DYNSPHERE_SHARD_COUNT=32
sbatch -A GRANT-cpu -p plgrid --array=0-31 \
  benchmarks/official_baselines/hpc/plgrid_slurm_array.sh
```

Monitor and inspect resource use with:

```bash
squeue --user "$USER"
scontrol show job JOB_ID
sacct -j JOB_ID --format=JobID,Elapsed,MaxRSS,State
```

Cancel only the intended job or array with `scancel JOB_ID`.

### 5.3 Submit the final three-seed paper campaign

Define the shared paths and account-specific scheduler values:

```bash
export DYNSPHERE_REPO=/shared/path/pendulum
export DYNSPHERE_RESULT_ROOT="$DYNSPHERE_REPO/results/official_baselines_v1"
export DYNSPHERE_TSB_DATA_ROOT="$DYNSPHERE_REPO/benchmarks/tsb_ad_m/data/TSB-AD-M"
export DYNSPHERE_PROFILE=paper-multiseed
export PLGRID_CPU_ACCOUNT=GRANT-cpu
export PLGRID_GPU_ACCOUNT=GRANT-gpu
export PLGRID_CPU_PARTITION=plgrid
export PLGRID_GPU_PARTITION=plgrid
mkdir -p "$DYNSPHERE_REPO/logs" "$DYNSPHERE_RESULT_ROOT/campaigns"
```

Validate the full matrix with the dry-run command in Section 4.8. Submit the
three components with distinct method lists and manifests:

```bash
DYNSPHERE_DEVICE=cuda \
DYNSPHERE_SHARD_COUNT=16 \
DYNSPHERE_METHODS="catch mtad_gat anomaly_transformer tranad" \
DYNSPHERE_CAMPAIGN_MANIFEST="$DYNSPHERE_RESULT_ROOT/campaigns/paper-multiseed-plgrid-h100.json" \
sbatch -A "$PLGRID_GPU_ACCOUNT" -p "$PLGRID_GPU_PARTITION" --gres=gpu:1 \
  --array=0-15 benchmarks/official_baselines/hpc/plgrid_slurm_array.sh

DYNSPHERE_DEVICE=cpu \
DYNSPHERE_SHARD_COUNT=16 \
DYNSPHERE_METHODS="dagmm omni_anomaly_tsb_ad" \
DYNSPHERE_CAMPAIGN_MANIFEST="$DYNSPHERE_RESULT_ROOT/campaigns/paper-multiseed-plgrid-external-cpu.json" \
sbatch -A "$PLGRID_CPU_ACCOUNT" -p "$PLGRID_CPU_PARTITION" \
  --array=0-15 benchmarks/official_baselines/hpc/plgrid_slurm_array.sh

DYNSPHERE_DEVICE=cpu \
DYNSPHERE_SHARD_COUNT=32 \
DYNSPHERE_METHODS="pca_reconstruction kmeans_distance predictive_euclidean_single dynsphere dynsphere_euclidean dynsphere_one_regime dynsphere_euclidean_one_regime" \
DYNSPHERE_CAMPAIGN_MANIFEST="$DYNSPHERE_RESULT_ROOT/campaigns/paper-multiseed-plgrid-local.json" \
sbatch -A "$PLGRID_CPU_ACCOUNT" -p "$PLGRID_CPU_PARTITION" \
  --array=0-31 benchmarks/official_baselines/hpc/plgrid_slurm_array.sh
```

Monitor with `squeue` and `sacct`. A selective rerun uses the same environment
values and an explicit array list, such as `--array=2,9,14`. Once every
component is complete, merge and aggregate:

```bash
.venv/bin/python -m benchmarks.official_baselines.merge_campaign_manifests \
  --output "$DYNSPHERE_RESULT_ROOT/campaigns/paper-multiseed-plgrid-complete.json" \
  "$DYNSPHERE_RESULT_ROOT/campaigns/paper-multiseed-plgrid-external-cpu.json" \
  "$DYNSPHERE_RESULT_ROOT/campaigns/paper-multiseed-plgrid-h100.json" \
  "$DYNSPHERE_RESULT_ROOT/campaigns/paper-multiseed-plgrid-local.json"

.venv/bin/python -m benchmarks.official_baselines.aggregate_campaign \
  --campaign-manifest "$DYNSPHERE_RESULT_ROOT/campaigns/paper-multiseed-plgrid-complete.json" \
  --result-root "$DYNSPHERE_RESULT_ROOT"
```

If AU and PLGrid execute different components, synchronize the component
result trees and manifests into one result root with `rsync -avP`. Compare the
manifest revisions before aggregation, then run the same merge command from
the synchronized repository.

## 6. Pilot Validation

Before the full campaign, execute one shard with a dry run:

```bash
.venv/bin/python -m benchmarks.official_baselines.run_campaign \
  --profile hpc-seed37 \
  --shard-index 0 \
  --shard-count 16 \
  --tsb-data-root "$DYNSPHERE_TSB_DATA_ROOT" \
  --result-root "$DYNSPHERE_RESULT_ROOT" \
  --dry-run
```

Then submit one array element through the scheduler. On PLGrid, use
`--array=0-0`. On AU, keep `DYNSPHERE_SHARD_COUNT=16` and submit element 1 of
the selected array file, for example:

```bash
bsub -J 'dynsphere-h100[1]' \
  < benchmarks/official_baselines/hpc/au_lsf_h100_array.bsub
```

The dedicated AU GPU probes in Section 4.3 provide the required device check
before this campaign pilot. Inspect the scheduler log, each method's stdout and
stderr, terminal `run.json`, and finite score vector.

## 7. Resumption and Failed Tasks

Rerunning the same array command is safe. Exact complete runs are skipped.
Timed-out, failed, incompatible, corrupt, or incomplete records are attempted
again. To rerun selected PLGrid elements, use an array list such as
`--array=2,5,11`. For AU, use an LSF array range or list supported by the local
LSF version.

Every array element writes the same deterministic campaign manifest. Reusing
that path with a different matrix is rejected. Use a new profile name, result
root, or explicit `DYNSPHERE_CAMPAIGN_MANIFEST` path for a changed campaign.

## 8. Campaign-Scoped Aggregation

After all array elements reach terminal states, aggregate only the jobs listed
in the campaign manifest:

```bash
.venv/bin/python -m benchmarks.official_baselines.aggregate_campaign \
  --campaign-manifest "$DYNSPHERE_RESULT_ROOT/campaigns/hpc-seed37.json" \
  --result-root "$DYNSPHERE_RESULT_ROOT"
```

The readiness report records unrelated excluded runs, superseded records,
incomplete jobs, metric errors, and the campaign identifier. When multiple run
identities have the same campaign key, aggregation retains the terminal record
with the newest `run.json` modification time and records the number excluded.
A manuscript table is written only when all selected terminal records are
complete and all VUS metrics use the official backend.

The AU campaign uses separate CPU and GPU manifests while the arrays run. Use
`au_lsf_aggregate.bsub` after both arrays finish. It creates the combined
six-method manifest and runs this aggregation command against that manifest.

## 9. Result Retrieval

Copy the result tree and scheduler logs back to the MacBook with resumable
transfer:

```bash
rsync -avP USER@LOGIN_HOST:/shared/path/pendulum/results/official_baselines_v1/ \
  results/official_baselines_v1/
rsync -avP USER@LOGIN_HOST:/shared/path/pendulum/logs/ logs/hpc/
```

Do not merge result roots from different campaigns until their manifests and
adapter revisions have been checked.

## 10. GPU Status

The AU H100 and L40 scripts request one GPU and set `DYNSPHERE_DEVICE=cuda`.
CATCH uses its upstream device selection. MTAD-GAT, Anomaly Transformer, and
TranAD explicitly place their models, inputs, and targets on the selected
device. Each result record stores the selected device.

Cluster execution still requires the probe in Section 4.3 because local tests
cannot validate the installed CUDA build, driver compatibility, or GPU access
on an AU compute node. Do not submit a full GPU array until all four probe runs
finish successfully and record `"device": "cuda"`.

## 11. Acknowledgements

AU and PLGrid require acknowledgement of their computing resources. Use the
current language supplied by AU and the exact PLGrid acknowledgement from the
grant Settlement tab, including the grant number and providing centre.
