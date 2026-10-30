# Source-native Exathlon evaluation

This protocol preserves application and execution boundaries in the pinned
source-native Exathlon release. It uses four applications, five test
executions, seeds 37--39, and ten score configurations. The reduced E1 score
reuses its E1 fit. Application 5 shares each fit across two test executions.

Pointwise metrics exclude 100 rows at both ends of each uninterrupted
segment. They are averaged over seeds, executions within application, and
applications equally. Official VUS 0.0.6 is evaluated on eligible contiguous
two-class segments and length-weighted within execution. Its application
coverage can differ from the pointwise cohort.

## Preparation

Obtain the source release using [dataset preparation](../../docs/DATASETS.md)
and prepare the pinned CATCH and MTAD-GAT environments using the
[baseline guide](../../benchmarks/official_baselines/README.md).

```bash
.venv/bin/python -m research.exathlon_native.run prepare \
  --tree /path/to/exathlon-tree.json
benchmarks/official_baselines/.envs/catch/bin/python \
  -m research.exathlon_native.run compatibility --method catch
benchmarks/official_baselines/.envs/mtad_gat/bin/python \
  -m research.exathlon_native.run compatibility --method mtad_gat
.venv/bin/python -m research.exathlon_native.run freeze
.venv/bin/python -m research.exathlon_native.run dry-run
```

Use the release inventory expected by the preparation command. After
freezing, changed source or inputs are rejected. Use a new campaign identity
for changes. Compatibility checks use synthetic inputs and do not contribute
benchmark scores.

## Execution and analysis

```bash
.venv/bin/python -m research.exathlon_native.run local
.venv/bin/python -m research.exathlon_native.evaluate
.venv/bin/python -m research.exathlon_native.evaluate --vus
```

For cluster execution, transfer identical prepared inputs and the manifest,
create compatible environments on the target host, set
`DYNSPHERE_NATIVE_REPO` to the checkout, and configure
`au_l40.bsub` for the local queue and GPU resources. Do not transfer
virtual environments across operating systems. Preserve all trial records.

Temporal metrics are cached and can be divided using
`--vus --metrics-only --shard-count N --shard-index I`. Run the summary
after all shards finish. Coverage reports distinguish completed,
unsuccessful, pending, and non-evaluable outcomes. Raw source data retain
their upstream license restrictions and are excluded from the repository.
