# Remaining-experiment ledger: October 22, 2026

Approved scope: complete the remaining evaluation stages listed in
EXTENSION_RESULTS_20261022.md, preserve completed records, and use manual
HPC submission for GPU-heavy work. The manuscript remains unchanged until
the scientific comparison is validated.

## Completed before this stage

- 1,188 local-method configurations with pointwise metrics, campaign 754b33fc172b.
- Six methods, 25 Exathlon traces, 23 GHL traces, 18 GutenTAG cases, three detector seeds.
- Existing manuscript campaign remains separately preserved.

## Current tasks

1. Implement and test a cached-score-only official VUS evaluator. Run the
   suitable local portion and prepare the larger real-data metric jobs.
2. Correct and audit TranAD preprocessing/alignment, then check Anomaly
   Transformer scoring/training against its released implementation.
3. Run matched component and computational-cost checks that are suitable locally.
4. Prepare the remaining audited-baseline jobs and exact manual submission instructions.
5. Audit source-native release availability and preserve its distinction from TSB-AD data.
6. Verify coverage and report completed work and external actions still required.

## Decisions and evidence

- Ruling: retain the current approved checkout and its unrelated changes.
  The active data and cached artifacts use this workspace; no reset, merge,
  push, or automatic commit is permitted by this execution decision.
- Ruling: VUS must consume immutable cached scores and the matching private
  labels. The older aggregator lacks the common first-row exclusion and is
  unsuitable unchanged for these extension results.
- Timing probe: official VUS took 2.24 seconds on one 7,999-row GutenTAG
  case, 28.44 seconds on one 32,299-row Exathlon case, and 28.04 seconds on
  one 150,000-row GHL case. Timing is configuration-specific.
- Ruling: reuse metrics only for identical score/label inputs, evaluator
  code, backend version, and evaluation support. This reduces repeated
  calculations without changing the metric definition.

## Progress

- Cached-score VUS evaluator: implemented, tested, and reviewed. GutenTAG
  complete for seven methods (378 records). Real-data preflight validates
  1,008 records and 330 unique inputs; manual HPC execution remains.
- Audited neural baselines: TranAD scaling/alignment configuration and
  released-capacity Anomaly Transformer profile implemented; isolated
  CPU smoke tests pass. Full comparison remains an HPC stage.
- Local controls: all 198 KMeans configurations complete, bringing the
  full local comparison to 1,386 records. Neural predictor pilot retained;
  its full workload is assigned to batch execution after timing it locally.
- Matched components and scaling: 2,178 E1 metric records and 72 controlled
  fresh-process cost measurements complete. Principal settings remain fixed.
- Warmup sensitivity: all 1,386 seven-method pointwise records recomputed
  with the same first 100 rows excluded; the main ordering is preserved.
- Source-native follow-up: official sources and preparation requirements
  audited; obtaining/preparing/evaluating the releases remains pending.
- Final experiment package and coverage report: prepared in
  EXTENSION_HPC_NEXT_STEPS_20261022.md and EXTENSION_PROGRESS_20261022.md.
- Verification: 187 tests passed; review findings addressed. The manuscript
  is unchanged. Completion of the entire submission evaluation is not claimed.
