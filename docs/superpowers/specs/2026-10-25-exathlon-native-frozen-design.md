# Broader source-native Exathlon evaluation

Status: approved by the user on 2026-10-25. Implementation and verification authorized.

## Purpose and interpretation

Evaluate whether application- and input-rate-matched independent normal executions provide useful normal-dynamics references for DynSphere, with strong deep comparisons. Preserve the existing five-family held-out study and all native pilot artifacts. This is a separate Exathlon study with custom source-native preparation.

Applications 1 and 2 have already been examined in native pilots. Applications 4 and 5 add native-preparation cases, although Exathlon results under other preparations have already informed development. Describe this history explicitly. The study cannot establish the causal benefit of execution separation because several preparation choices differ from the previous TSB-AD experiments.

The selected test executions cover bursty-input injections and a bursty-input-until-crash injection. They provide limited anomaly-type diversity and cannot support claims about every Exathlon fault category. Scoring is retrospective, consistent with the existing local detector and block-reconstruction baselines.

## Approaches

1. **Exact application and input-rate matching (recommended).** Include every disturbed execution with at least two undisturbed executions at the same application and nominal input rate. This gives four applications and five disturbed executions. It directly extends the pilot hypothesis with a bounded campaign.
2. **Application matching with input-rate shift.** This would cover eight applications and 31 disturbed executions, but most test executions have an input rate absent from normal fitting data. Reserve it for a separately frozen distribution-shift study.
3. **All-source pooling.** This introduces application heterogeneity and answers a different generalization question. It is outside this campaign.

The user approved exact matching and the detailed protocol below on 2026-10-25.

## Frozen cohort

Official repository: https://github.com/exathlonbenchmark/exathlon

Release commit: `4101f6087f902fa150e392b65c957976faa84e40`.

Inspect the complete pinned Git tree, including nested archive paths. Use nominal rate 100000 because it is the only rate satisfying the eligibility rule. Within each eligible application/rate group, use the two lowest numeric undisturbed execution IDs for fitting and calibration, respectively. Include every disturbed execution in that group. Selection uses file metadata only.

| Application | Fitting execution | Calibration execution | Test execution(s) |
| --- | --- | --- | --- |
| 1 | `1_0_100000_15` | `1_0_100000_16` | `1_2_100000_68` |
| 2 | `2_0_100000_20` | `2_0_100000_22` | `2_1_100000_60` |
| 4 | `4_0_100000_27` | `4_0_100000_28` | `4_1_100000_61` |
| 5 | `5_0_100000_33` | `5_0_100000_34` | `5_1_100000_63`, `5_1_100000_64` |

There are 13 distinct trace archives, totaling 521,464,728 compressed bytes when all split volumes are included, plus labels and source/license evidence. The application-4 test archive is nested at `data/raw/app4/4_1_100000_61/4_1_100000_61.zip`, with three additional `.z01` to `.z03` volumes. Verify every volume and assemble a separate ZIP artifact without modifying source bytes. The initial metadata estimate of 364,178,328 bytes counted the terminal ZIP files only; this correction preserves the approved cohort.

Freeze every selected archive's exact release path and Git blob hash in the machine-readable manifest. Verify downloaded bytes against the pinned blob, record SHA256, and retain the complete eligibility/exclusion inventory. Never substitute an execution after seeing scores. Numeric execution IDs determine partition assignment; they do not imply fitting-before-calibration chronology. Report actual time spans and whether each test follows both normal executions.

## Preparation and common support

- Reuse the inspected pilot's 19-feature published Spark bundle, one-second observations, duplicate-timestamp averaging, inactive-executor repair, and within-segment differences. Record that this differs from the complete official Exathlon preparation.
- Require finite nondecreasing source timestamps. A gap exceeding one second starts a new contiguous segment. No derivative, prediction window, score propagation, or temporal metric buffer crosses a segment or execution boundary.
- Fit imputation and scaling statistics exclusively on the fitting execution. Require normal-only fitting and calibration annotations. Test annotations are used only for evaluation.
- Keep the pilot's primary common support: exclude the first and last 100 observations of each test segment. Segments with at most 200 observations contribute no primary rows. Apply the same mask to every configuration, regardless of its scores.
- Apply the same 100/100 support to calibration scores. A lack of valid calibration support makes calibrated operating-point diagnostics unavailable; threshold-free test metrics can still be evaluated.
- For the optional calibrated diagnostic, select the empirical 95th percentile of retained normal calibration scores with NumPy's `method='higher'` quantile rule, matching the existing evaluation helper, and flag scores strictly above that threshold. Report the realized calibration exceedance fraction, test false-positive rate, and test recall. Ties can reduce the realized exceedance fraction. CATCH uses this normal execution for early stopping as well, so this diagnostic is not an independent threshold-validation experiment.
- Report retained rows, segments, events, positive/negative counts, and coverage fractions per execution. Preserve executions lacking both test classes as explicitly non-evaluable. Do not replace them or silently remove their manifest entries.
- Retain the existing 10/10 local-control boundary diagnostic, and extend it to the same local configurations if useful. Label it a separate support sensitivity with no matched deep-baseline claim. The common 100/100 comparison remains primary.

## Configurations, seeds, and budgets

Ten configurations: DynSphere-S4, DynSphere-E4, DynSphere-S1, DynSphere-E1, E1 motion/state-speed with radius-8 propagation, PCA reconstruction, KMeans distance, Ridge AR(1), CATCH, and ML4ITS MTAD-GAT.

Seeds: 37, 38, 39. Retain the existing local settings and upstream-pinned deep architectures. CATCH and MTAD-GAT use window 100, batch size 128, and at most 10 epochs, with their existing learning-rate and scoring settings. Preserve the existing CATCH coverage-count correction. No tuning against these test labels, no new Anomaly Transformer investigation, and no test-based choice of a DynSphere variant.

Fit once per application, configuration, and seed; reuse that fitted model for both application-5 tests. Derive the reduced score from the corresponding E1 fit. The maximum campaign is 150 scored configuration/execution/seed records, backed by 108 model fits (84 local and 24 deep). Three seeds measure algorithmic variation, not three independent datasets.

## Deep-baseline boundary handling

Existing deep workers cannot be reused unchanged. CATCH's current reserved-validation path requires an 80/20 row ratio, which independent normal executions generally do not satisfy.

- CATCH: preserve the upstream architecture, training loop, early stopping, and scoring. Add a scoped data-interface wrapper that supplies the exact independent fitting and calibration executions to the upstream split interface, and segment-aware fitting/validation loaders. Keep the normal-only fitting scaler. Record the wrapper and upstream source hashes.
- MTAD-GAT: preserve the current 10-epoch training and forecast-plus-reconstruction score. Build training windows separately within each fitting segment. Score each calibration/test segment independently. Its existing fixed-epoch training does not require calibration-based early stopping.
- CATCH scoring uses complete upstream scoring blocks within each segment. MTAD-GAT requires 100 preceding observed rows. Record actual observed support; never count padded or copied edge scores as observations. Assert that all primary common-mask rows have genuine finite scores for every successfully evaluated method.
- Test these interfaces with artificial gaps and arbitrary fit/calibration lengths before fitting models on the new cases. Do not modify shared historical adapters or campaigns.

## Metrics and aggregation

Primary: standardized partial AUROC through FPR 0.05, computed on the common retained test rows. Average seeds within execution, executions within application, then applications equally. Report equal-execution weighting as a sensitivity. This is one Exathlon study; its application mean is separate from the existing five-family mean.

Secondary: AP/AUPRC, AUROC, VUS-PR, and VUS-ROC. Use official `vus==0.0.6` with a fixed maximum buffer of 100 one-second observations. Evaluate VUS separately on each contiguous retained segment containing both classes, then compute a retained-length-weighted mean within execution. Record eligible segment lengths and VUS coverage. Single-class segments remain in pointwise evaluation; their VUS is undefined. This segment-weighted temporal statistic must be labeled explicitly and cannot be compared numerically to the pilot's concatenated-gap VUS.

Preserve all non-evaluable records and baseline exceptions. Publish available coverage plus the strict all-method/all-seed matched subset; report whether matching changes the application set. Do not claim complete-cohort superiority from incomplete runs. If no complete matched cohort exists, state that result explicitly.

Method-specific available aggregates require all three seeds for each included execution. Keep partial-seed outcomes in the per-run inventory, without mixing them into these aggregates. If only one application-5 test execution is eligible in a summary, that execution represents the application and its 1/2 coverage is reported. The strict comparison uses the same eligible executions for every configuration. Always report execution and application denominators, and label a summary missing an application as a reduced-cohort result.

Report applications 1/2 separately from 4/5, together with the complete four-application summary. Restrict uncertainty claims given only four eligible applications. Any bootstrap is paired at application level, averages seeds first, and is identified as exploratory with only four clusters.

## Artifacts, execution, and acceptance

Add isolated reusable code under `research/exathlon_native/`, focused tests under `tests/test_exathlon_native*.py`, and a new result root `results/exathlon_native_frozen_20261025/`. Keep source archives in a verified cache; never overwrite pilot outputs or historical campaigns. New model implementation hashes, preparation hashes, exact inputs, package versions, baseline sources, configuration, support, and the full run list are frozen before scoring.

Prepare immutable arrays locally, then run local controls on the laptop. Prepare an AU LSF L40/long runner for CATCH and MTAD-GAT, with bounded concurrency and explicit 24-hour per-fit timeout. Check existing jobs before any submission; leave them unchanged. Use an existing authenticated SSH connection if available. If authentication has expired, provide the runner and request reconnection without collecting credentials.

Start with an isolated interface/compatibility check using synthetic arrays, with outputs outside the frozen campaign. Proceed to the complete frozen run list only after it passes. A dry run must show 150 planned score records, 108 fits, the five exact test executions, and three seeds. Fail closed on source/input/configuration drift. Resume only after verifying completed artifacts, with a single-writer lock and atomic output replacement for new results.

Acceptance requires tests for metadata eligibility, nested archives, duplicate timestamps, gaps, independent calibration lengths, training-only scaling, score alignment, VUS boundaries, degenerate segments, source drift, resume integrity, and application-weighted aggregation. Verify that pilot and manuscript result hashes are unchanged. New results remain internal until coverage, numerical validity, and interpretation are reviewed.

## Decisions retained for the implementer

Choose module decomposition, chunked CSV reading, cache format, and precise worker layout while preserving the interfaces and scientific requirements above. Benchmark one compatible deep fit to estimate runtime; record the estimate before scheduling the remaining fits. Wall-clock estimates must be empirical. Record resource-related incomplete runs without changing model settings or replacing cases.
