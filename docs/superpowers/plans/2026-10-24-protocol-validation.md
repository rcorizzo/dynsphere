# Held-out protocol validation implementation plan

> **For agentic workers:** Use superpowers:executing-plans for native execution. The user approved the experimental campaign on October 24, 2026.

**Goal:** Preserve historical results and complete an internal, independently identified first-stage held-out and low-false-alarm evaluation.

**Architecture:** An isolated research package reads immutable sources and fits the existing four DynSphere variants plus PCA, KMeans, and ridge AR(1). It does not change historical adapters, scores, or manuscript files. Input and implementation hashes identify every run.

**Tech Stack:** Existing Python environment, NumPy, pandas, scikit-learn, threadpoolctl, pytest.

**Spec:** The approved October 24 discussion: preserve existing PowerSystem results; evaluate disjoint support; audit NASA training prefixes; report partial AUROC and training-calibrated operating points. This is stage one. Deep-baseline corrections and full matched deep reruns are subsequent stages.

## Global constraints

- Native execution in the existing checkout; no commit, branch change, deletion, HPC submission, or manuscript edit.
- Write new files under research/evaluation_protocols, tests, and results/protocol_validation_20261024 only.
- Seeds 37, 38, 39; all 59 NASA cases from the existing campaign, BATADAL, HAI, and one source-grouped PowerSystem diagnostic.
- NASA: reject any contaminated declared training prefix; reserve its final 20% for normal calibration, fit only on its first 80%, evaluate the untouched suffix.
- BATADAL and HAI: reserve final 20% of the released normal training file for calibration; preserve separate released test files.
- PowerSystem: data1..9 training, data10..12 calibration, data13..15 test; source-file order has unverified temporal meaning. Remove exact feature-vector overlaps from earlier partitions, retaining the test partition and recording counts. No claim of scenario independence or temporal generalization.
- Normal-only selection creates separate entities at every original-row gap. No differentiation across files, gaps, or calibration/test boundaries.
- Imputation and standardization are fitted exclusively on fitting observations. All-missing fitting columns are discarded.
- Discard first 100 and last 10 observations per test entity and first/last 10 per calibration entity. Derivatives and centered score propagation use up to 10 future samples. These are retrospective scores, with no zero-latency deployment claim.
- Report AUPRC, AUROC, standardized partial AUROC at 1% and 5% FPR, normal-calibration thresholds at 1% and 5%, realized test FPR, point recall, and event recall. No VUS in this bounded first-stage run.
- Identical support for all seven methods. Failed/incompatible records remain visible. No performance-based source selection.

## Review focus

1. Contaminated training prefixes must be rejected before fitting.
2. Imputation, scaling, and alert thresholds must not use test labels or test statistics.
3. Exact PowerSystem overlap removal and discontinuity boundaries must survive normal-row selection.
4. Test/calibration support and score alignment must be identical for every method.
5. Resume must validate source, implementation, and saved artifact hashes; old successful records cannot hide failures.

## Task 1: Protocol preparation and low-FPR metrics

- [ ] Add tests for contaminated prefixes, train-only preprocessing, group overlap, boundary masks, score ties, and single-class inputs.
- [ ] Run tests and observe missing implementation failures.
- [ ] Implement preparation and metric functions in research/evaluation_protocols/campaign.py.
- [ ] Run focused tests to green.

## Task 2: Frozen campaign and model execution

- [ ] Add tests for entity-aware model parity, deterministic manifest identity, score artifact validation, and output support.
- [ ] Observe failures, implement the seven model wrappers and resumable runner, rerun focused tests.
- [ ] Freeze all source hashes and execute an initial one-case-per-source seed-37 pilot, then remaining seeds/cases if the checks pass.
- [ ] Summarize all source-method results and historical partial-AUROC tables separately; identify real-data illustration candidates without changing the paper.

## Task 3: Verification and handoff

- [ ] Run the full project test suite and one fresh final review of the new package.
- [ ] Address important findings with regression tests, verify experiment artifact hashes and metric reproduction, and write an internal report.
- [ ] Report completed and pending stages explicitly; no automatic manuscript promotion.
