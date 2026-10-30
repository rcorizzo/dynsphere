# Cluster execution

Use this guide with the [baseline setup instructions](README.md). Create
separate Linux environments for the pinned external implementations; do not
copy a virtual environment between operating systems.

## Preparation

1. Clone this repository into a directory on the cluster.
2. Install the controller and method environments.
3. Obtain the datasets described in [dataset preparation](../../docs/DATASETS.md).
4. Set the checkout path and create the scheduler log directory:

```bash
export DYNSPHERE_REPO="$PWD"
mkdir -p logs
```

Inspect a profile without model fitting:

```bash
.venv/bin/python -m benchmarks.official_baselines.run_campaign \
  --profile macbook-safe --dry-run
```

The `macbook-safe` profile runs sequentially. For the historical
multiseed benchmark profile, use `--profile paper-multiseed`.
The held-out protocol is documented separately in the
[held-out guide](../../research/submission_campaign/HPC_RUN_GUIDE.md).

## Scheduler templates

LSF and Slurm templates are under [hpc/](hpc/). Select the template matching
the required profile, CPU/GPU resources, and scheduler. Set the repository
and Python paths, queue or partition, account, wall-time, and concurrency to
values supported by the cluster. Check the template before submitting it;
module names and GPU types are site-specific.

Use the same code revision, dataset selection, prepared inputs, and method
configuration for every shard of a campaign. Inspect import and GPU
compatibility checks on an allocated compute node before a large run.
Do not submit overlapping arrays against the same trial assignments.

## Aggregation

```bash
.venv/bin/python -m benchmarks.official_baselines.aggregate_campaign \
  --campaign-manifest results/official_baselines_v1/campaigns/macbook-safe.json \
  --result-root results/official_baselines_v1
```

Substitute the manifest for the profile actually executed. Aggregation checks
campaign identities and recorded coverage. Keep all unsuccessful outcomes;
do not combine unrelated campaigns or different test-boundary policies.
Official VUS metrics require `vus==0.0.6`.

Transfer completed result directories without deleting existing files. Keep
the manifest, dependency/source inventories, scores, and terminal records
together so their hashes can be validated after transfer.
