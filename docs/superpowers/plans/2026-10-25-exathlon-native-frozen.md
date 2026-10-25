# Source-native Exathlon Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for native execution, task by task, with test-first verification.

**Goal:** Execute the approved 150-record source-native comparison without changing previous experiments.

**Architecture:** Isolated source preparation, segment-aware model interfaces, immutable score artifacts, and independently cached evaluation. The manifest binds the complete cohort, inputs, code, configurations, and upstream sources before scoring.

**Tech Stack:** Existing Python/NumPy/pandas/sklearn environment; existing separate CATCH/MTAD-GAT Torch environments; official VUS 0.0.6; AU LSF.

**Spec:** `docs/superpowers/specs/2026-10-25-exathlon-native-frozen-design.md` (approved).

## Global Constraints

- Exact release commit `4101f6087f902fa150e392b65c957976faa84e40`; applications 1, 2, 4, 5; five test executions; seeds 37, 38, 39.
- Ten score configurations, 108 maximum fits, 150 score records. Primary support excludes 100 rows at both ends of every segment.
- Normal fitting and calibration executions are independent. No window or temporal metric crosses gaps.
- Primary standardized pAUROC at FPR 0.05 uses equal application weighting. Official segment-weighted VUS has maximum buffer 100.
- New code and outputs only. Preserve all pilot, previous campaign, and manuscript files.
- Native implementation; no per-task subagents. One fresh final reviewer.

## Review Focus

- Nested raw archives and missing/ambiguous published features: fail explicitly without substituting a trace.
- Repeated timestamps and short segments: average duplicates before transforms, exclude unsupported rows consistently.
- Arbitrary independent calibration lengths: CATCH must receive the intended partition without an implicit 80/20 split.
- Missing seeds or constant-class segments: preserve denominators, avoid incomplete-seed aggregate means.
- Source drift, corrupt resume artifacts, and concurrent workers: fail closed and preserve previous attempts.

### Task 1: Source preparation and manifest

**Files:** Create `research/exathlon_native/{__init__,protocol,data}.py`; test `tests/test_exathlon_native.py`.

**Interfaces:** `select_cohort(tree)` returns all groups plus exclusions; `read_trace(path, events)` returns features, labels, entities, audit; `prepare(tree, output, source_cache)` writes immutable arrays; `freeze(output)` binds inputs and implementation.

- [ ] Write tests for eligibility, nested paths, duplicates/gaps, training-only preprocessing, and immutable/source checks.
- [ ] Run `.venv/bin/python -m pytest tests/test_exathlon_native.py -q`; observe missing implementation.
- [ ] Implement the pinned selection, inspected pilot transforms, archive verification, input audits, and manifest.
- [ ] Run the tests; expect all new tests to pass.

### Task 2: Segment-aware models and execution

**Files:** Create `research/exathlon_native/{models,run}.py`; test `tests/test_exathlon_native_models.py`.

**Interfaces:** `window_starts(entities, window, forecast)` returns valid within-segment starts; deep fit/score classes consume label-free arrays and entities; run controller consumes manifest and generates hashed, locked, resumable fit bundles.

- [ ] Write failing tests for windows, arbitrary calibration split, observed score support, seeded reproducibility, source drift, corrupt resume, and dry-run counts.
- [ ] Implement scoped CATCH split/provider wrappers and boundary-aware MTAD-GAT training/scoring, preserving upstream architectures and losses.
- [ ] Implement immutable per-fit attempts and controller commands `prepare`, `freeze`, `dry-run`, `local`, `deep-worker`, `deep`, `compatibility`.
- [ ] Run focused tests and actual isolated synthetic compatibility checks in both baseline environments; expect finite genuine supported scores, with no cross-gap windows.

### Task 3: Evaluation and aggregation

**Files:** Create `research/exathlon_native/evaluate.py`; test `tests/test_exathlon_native_metrics.py`.

**Interfaces:** `pointwise(labels,scores,calibration)`; `segment_vus(labels,scores,entities,mask,cache)`; `aggregate(frame)` returns per-execution, application, and equal-application comparisons plus coverage.

- [ ] Write failing tests for segment-separated VUS, single-class coverage, independent quantile diagnostics, and complete-three-seed aggregation.
- [ ] Implement official VUS cache, strict matched support, available outcomes, and equal-execution sensitivity.
- [ ] Run focused tests, then the full project test command. Record any pre-existing failures explicitly.

### Task 4: Frozen execution and handoff

**Files:** Create `research/exathlon_native/au_l40.bsub`, `README.md`; generate results only below the new result root.

- [ ] Snapshot preservation hashes; prepare verified inputs and freeze the final implementation and 150 records.
- [ ] Run all 84 local fits and retain 120 local score records; calculate pointwise and temporal metrics separately.
- [ ] Benchmark a deep synthetic compatibility fit. Prepare the 24-fit L40 array with bounded concurrency, 24-hour per-fit timeouts, source verification, and explicit baseline environments.
- [ ] Verify an existing authenticated HPC connection and inspect active jobs before submission. If absent, provide the upload/submission commands and ask for reconnection; leave existing jobs unchanged.
- [ ] Obtain one final fresh code review, address important issues with regression tests, verify preservation hashes, and report actual completed/pending counts.

## Execution record

Native execution is retained from the user's earlier preference. New modules provide file-level isolation because the existing research helpers are untracked and required by the approved design; a clean worktree would omit them. Existing changes are preserved. No broad staging or history manipulation is permitted.
