# Corrected DAGMM and OmniAnomaly CPU evaluation

This runner evaluates the CPU deep baselines on the
[held-out comparison](../submission_campaign/HPC_RUN_GUIDE.md). Architecture,
training budgets, windows, and test boundaries remain fixed.

## Environment and validation

DAGMM requires its pinned TensorFlow environment plus pandas 1.5.3 and
pytz 2024.1. Install these packages as binary wheels in the separate
`dependencies/dagmm-overlay` directory. The runner records package
versions, import locations, and overlay file hashes; it does not modify a
shared environment.

For TSB-AD OmniAnomaly, the adapter supplies the fitting rows followed by
reserved normal-calibration rows. Its internal 80/20 validation split then
recovers the declared fitting boundary. No test data enter validation.
The upstream dataset class independently normalizes fitting, validation,
and test partitions. This transductive preprocessing behavior is retained
and must be identified in comparisons.

## Execution

Prepare compatible method environments using the
[baseline guide](../../benchmarks/official_baselines/README.md). Restore the
protocol manifest and bound input sources before using the frozen runner.

```bash
export DYNSPHERE_RECOVERY_REPO="$PWD"
mkdir -p logs
.venv/bin/python -m research.submission_recovery.run --help
```

LSF templates `au_setup.bsub`, `au_pilot.bsub`, and `au_cpu.bsub`
provide environment setup, a first-case technical check, and the CPU array.
Configure their queue, module and resource settings for the execution host.
Verify setup and pilot evidence before launching the full array. Each
attempt retains its terminal record; do not overlap arrays assigned to
the same trials.

## Matched aggregation

The aggregator requires explicit original and corrected result directories
and their manifests:

```bash
.venv/bin/python -m research.submission_recovery.aggregate \
  --original-root /path/to/original/runs \
  --original-manifest /path/to/original/manifest.json \
  --recovery-root /path/to/corrected/runs \
  --recovery-manifest /path/to/corrected/manifest.json \
  --output /path/to/new/summary
```

It validates source/configuration/input/score identities and reports
pointwise quality on common successful case/seed support. Earlier attempts
remain separate. Select an unused output directory.

For `au_l40_cpu_aggregate.bsub`, set
`DYNSPHERE_AGGREGATION_REPO`, `DYNSPHERE_ORIGINAL_ROOT`,
`DYNSPHERE_ORIGINAL_MANIFEST`, `DYNSPHERE_RECOVERY_ROOT`,
`DYNSPHERE_RECOVERY_MANIFEST`, and `DYNSPHERE_AGGREGATION_OUTPUT`
to the corresponding paths on the execution host before submission.
