# Portable Official-Baseline Execution

This guide runs the same resumable experiment campaign on a MacBook,
American University Lovelace, and PLGrid. The Python campaign runner defines
the experiment matrix and shard membership. The scheduler scripts only request
resources and supply the array index.

## 1. Execution Stages

| Profile | Methods | Data | Seeds | Intended system |
|---|---|---|---|---|
| `macbook-safe` | DAGMM, TranAD, TSB-AD OmniAnomaly fallback | MSL, SMAP, SMD | 37 | MacBook |
| `hpc-seed37` | Six preferred external implementations | Current real data and complete TSB-AD-M manifest | 37 | AU or PLGrid |
| `hpc-followup` | Same external implementations | Current real data | 38, 39 | AU or PLGrid after method selection |

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
require batch scheduling and prohibit computation on login nodes. They list a
48-hour `normal` queue, a 240-hour `long` queue, a two-hour `short` queue, an
`H100` queue, and a `bigmem` queue. The
[system specification](https://www.american.edu/cas/hpc/software.cfm) lists
Python through Anaconda or Miniconda modules and current CPU/GPU hardware.
Verify the current module name with `module avail`.

### 4.1 Environment setup

From the repository directory, submit the setup as a CPU batch job:

```bash
cd "$DYNSPHERE_REPO"
mkdir -p logs
export DYNSPHERE_PYTHON_MODULE=YOUR_CURRENT_PYTHON_MODULE
bsub -J dynsphere-setup -q normal -n 4 \
  -R 'rusage[mem=16000]' -W 08:00 \
  -o logs/au-setup-%J.out -e logs/au-setup-%J.err \
  < benchmarks/official_baselines/hpc/setup_environment.sh
```

Omit `DYNSPHERE_PYTHON_MODULE` when `python3` is already available in batch
jobs. Inspect the setup log and
`results/official_baselines_v1/setup/provenance_report.json` before launching
the array.

### 4.2 Seed-37 array

The checked-in LSF script defines 16 one-based array elements. Exported
variables are inherited by the batch jobs:

```bash
export DYNSPHERE_PROFILE=hpc-seed37
export DYNSPHERE_SHARD_COUNT=16
bsub < benchmarks/official_baselines/hpc/au_lsf_array.sh
```

To use 32 shards, override both the scheduler range and campaign shard count:

```bash
export DYNSPHERE_SHARD_COUNT=32
bsub -J 'dynsphere[1-32]' < benchmarks/official_baselines/hpc/au_lsf_array.sh
```

Monitor with the Lovelace LSF commands available on the login node, commonly
`bjobs`, `bjobs -l JOB_ID`, and `bkill JOB_ID`. Confirm these commands in the
current MyAU HPC documentation.

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

Then submit one array element through the scheduler. On PLGrid use
`--array=0-0`; on AU use `-J 'dynsphere[1-1]'` and set
`DYNSPHERE_SHARD_COUNT=1`. Check the scheduler log, each method's stdout and
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

The readiness report records unrelated excluded runs, incomplete jobs, metric
errors, and the campaign identifier. A manuscript table is written only when
all selected terminal records are complete and all VUS metrics use the official
backend.

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

The present array scripts request CPU resources. The current external adapters
have not yet been verified to move every model and tensor to CUDA. Requesting
an H100, A100, GH200, L40, or V100 allocation would therefore consume a GPU
without established acceleration. GPU directives should be added after a
device-placement test confirms CUDA execution for each selected method.

## 11. Acknowledgements

AU and PLGrid require acknowledgement of their computing resources. Use the
current language supplied by AU and the exact PLGrid acknowledgement from the
grant Settlement tab, including the grant number and providing centre.
