# Isolated CPU recovery, October 24, 2026

This campaign preserves all original results and leaves arrays 44785 and
44786 unchanged. It runs from a separate checkout and writes a new result
identity. Shared environments, upstream checkouts, adapters, and the detector
remain unchanged. No paper table is updated automatically.

## Corrections

1. DAGMM's worker imports pandas through the frozen campaign helpers. Its
   environment lacks pandas and pytz. Install pandas 1.5.3 and pytz 2024.1 as
   binary wheels in `dependencies/dagmm-overlay`, used only by new workers.
   NumPy, TensorFlow, SciPy, and scikit-learn retain their original versions.
   Runtime evidence includes package versions, import origins, and content
   hashes. No shared environment installation is performed.
2. TSB-AD OmniAnomaly internally reserves 20% of its fitting input for
   validation. Applying this split to an already shortened fitting prefix
   leaves some MSL validation partitions below its fixed 100-observation
   window. The corrected runner passes the original fitting observations
   followed by the reserved normal-calibration observations. Its internal
   split recovers the exact original fitting boundary. Validation receives
   no test observations or test labels. Architecture, loss, learning rate,
   window, and epoch budget are unchanged.

OmniAnomaly's upstream `ReconstructDataset` normalizes fitting, validation,
and test partitions separately. That behavior is unchanged and constitutes
a transductive feature-normalization caveat. The correction does not establish
strictly training-only internal normalization for that baseline.

Both corrected methods receive a new identity and rerun all cases and seeds.
The previous OmniAnomaly successes remain in the original campaign. They
are excluded from the corrected comparison to keep one validation policy.

## Launch

Use the isolated recovery checkout, load `git/2.40.0`, and set
`DYNSPHERE_RECOVERY_REPO` to that checkout. Create its `logs` directory.
Submit `au_setup.bsub`, verify successful private imports, then submit
`au_pilot.bsub`. The pilot runs the first MSL case at seed 37 for both methods.
Verify two complete, independently validated results before submitting
`au_cpu.bsub`. The array has 16 shards, a concurrency limit of four, and
366 planned records: 360 eligible executions and six declared incompatible
records. It reuses the two successful pilot records after validation.

Retries create a new numbered attempt directory, preserving old attempts.
Do not run overlapping recovery arrays concurrently. Source changes generate
a different manifest. Existing source and result directories are preserved.

## Explicit aggregation

The new `aggregate` module requires original and recovery roots and frozen
manifests. It chooses the original four GPU methods and the corrected two
CPU methods explicitly. It rejects changed configurations, upstream sources,
inputs, score hashes, and duplicate method/case/seed entries. It recomputes
the seven local controls on the same 100-observation boundary exclusions.
Partial and all-method matched-support summaries remain separate. The output
directory must be new; aggregation never overwrites an earlier summary.

```bash
.venv/bin/python -m research.submission_recovery.aggregate \
  --original-root /absolute/original/results/deep/ORIGINAL_PREFIX \
  --original-manifest /absolute/original/results/deep-manifests/ORIGINAL_ID.json \
  --recovery-root /absolute/recovery/results/campaigns/RECOVERY_PREFIX \
  --recovery-manifest /absolute/recovery/results/manifests/RECOVERY_ID.json \
  --output /absolute/new/summary-directory
```

These summaries include AUPRC, AUROC, and standardized partial AUROC at
FPR <= 5%. They do not claim a new VUS evaluation or submission readiness.
