# Held-out multiseed comparison

The held-out comparison uses separate fitting, normal-calibration, and test
observations from BATADAL, HAI-20.07, MSL, SMAP, and SMD, with seeds 37--39.
Input scaling is fitted on fitting rows. Primary metrics exclude 100 rows
at both test boundaries. Preserve the source manifest's contamination
exclusions and use the same cases and boundaries for every method.

## Prerequisites

Install the controller dependencies, obtain the datasets in
[dataset preparation](../../docs/DATASETS.md), and prepare the pinned method
environments using the [baseline guide](../../benchmarks/official_baselines/README.md).

These runners require the prepared protocol manifest under
`results/protocol_validation_20261024/manifests/`. The default manifest
filename is specified by the runner. For an exact archived comparison,
restore that manifest and its bound inputs before execution. Use a new
output directory when inputs or source revisions differ.

## Local validation and execution

```bash
export DYNSPHERE_REPO="$PWD"
mkdir -p logs
.venv/bin/python -m research.submission_campaign.deep --help
.venv/bin/python -m research.submission_campaign.deep --dry-run \
  --methods mtad_gat catch tranad anomaly_transformer dagmm omni_anomaly_tsb_ad
```

Inspect the resulting source/runtime/input evidence before removing
`--dry-run`. Deep workers retain their configured architecture, window,
epoch budget, seed, and score direction. CATCH uses normal calibration for
validation. The [corrected CPU runner](../submission_recovery/README.md)
documents the DAGMM environment and OmniAnomaly fitting/validation interface.

## Cluster execution

Set `DYNSPHERE_REPO` to the checkout on the execution host. Create
Linux-compatible environments and verify dataset hashes locally on that host.
LSF templates are `au_h100.bsub` and `au_cpu.bsub`; Slurm templates
are `plgrid_gpu.sbatch` and `plgrid_cpu.sbatch`. Configure their
resources and site-specific module/account settings before submission.

`DYNSPHERE_DEEP_METHODS` selects a subset of methods. Keep shard
assignments disjoint and do not submit competing arrays to the same output.
The templates start training only when explicitly submitted.

## Aggregation

```bash
.venv/bin/python -m research.submission_campaign.aggregate \
  --deep-root results/submission_push_20261024/deep/DEEP_MANIFEST_PREFIX
```

Replace the prefix with the actual frozen manifest directory. To combine
the corrected CPU methods with the GPU methods, use the explicit original
and recovery arguments in the CPU baseline guide. Aggregators validate
inputs, source revisions, scores, and case/seed coverage. Preserve all
failed, incompatible, and timed-out records.
