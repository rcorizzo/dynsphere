# Window-conditioned DynSphere-E Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user prefers native execution.

**Goal:** Implement and evaluate the approved compact window-conditioned normal-dynamics prototype without altering existing detectors or results.

**Architecture:** A convolution/GRU encoder learns history representation with forecasting and optional reconstruction objectives. Frozen representations define normal regimes and ridge observation-increment predictors. A calibrated Gaussian mixture supplies one anomaly score.

**Tech Stack:** Existing Python environment, NumPy, SciPy, scikit-learn, PyTorch, pytest, and the unchanged official VUS backend. No dependency installation is planned.

**Spec:** `docs/superpowers/specs/2026-10-23-window-conditioned-dynsphere-design.md` (approved October 23, 2026).

## Global Constraints

- Keep existing root-level detector modules, official baseline adapters, metric sources, frozen campaigns, scores, and manuscript unchanged.
- Create an isolated `research/window_dynsphere/` package and a dedicated `results/window_dynsphere_v1/` output root.
- Defaults: window 100, causal convolution kernel 5 and width 32, GRU width 32, latent dimension 8, decoder/head width 32.
- Train 10 epochs with Adam, learning rate 1e-3, batch size 128, gradient clipping 5, CPU, one Torch thread, and at most 20,000 fixed entity-balanced fitting targets.
- Split normal training rows 80/20 by entity before internal scaling. The previous normal neural predictor must use the same split for architectural comparisons.
- Compare K in {1,4} and auxiliary reconstruction weight in {0,1}; primary K=1, weight=1. No test-based tuning.
- Ridge alpha=1 multiplies the penalty on mean squared loss; the intercept is unpenalized.
- Follow the spec's exact regime merging, temperature, weighted residual calibration, 32-pseudo-observation variance shrinkage, and variance floor 1e-6.
- Support regular sampling only. Scores for the first 100 rows of each test entity are NaN with an explicit validity mask.
- Pilot: 15 frozen cases, seed 37. Report unavailable or incompatible cases. No automatic full campaign or HPC submission.
- Preserve the existing extension source fingerprint `22357e3c923ef3219421f06a4eb49fd40e0d211347f4c2573994e5f083f4f28c`.
- Preserve unrelated working-tree changes. Do not commit, switch branches, or create a worktree containing an incomplete historical snapshot without explicit user direction.

## Review Focus

1. Sparse target subsampling must still compute velocity from adjacent windows, without treating gaps between sampled targets as one time step (Task 1).
2. An extreme calibration suffix must not change the fitting scaler, selected targets, or fitted neural parameters (Tasks 1 and 3).
3. Constant or underflowed regime weights must yield normalized densities and finite calibrated variances (Task 2).
4. A resumed run with a changed input, source, configuration, or damaged score file must never silently reuse stale results (Task 4).
5. Missing principal datasets and single-class post-warmup labels must be recorded explicitly, without selecting replacement cases (Task 4).

## File responsibilities

- `research/window_dynsphere/data.py`: validation, entity-local splits, target selection, bounded history batches.
- `research/window_dynsphere/network.py`: module-scope serializable neural model and encoder training.
- `research/window_dynsphere/density.py`: deterministic cluster merging, ridge experts, covariance calibration, mixture score.
- `research/window_dynsphere/detector.py`: estimator interface, orchestration, scoring, diagnostics, channel-independent control.
- `research/window_dynsphere/pilot.py`: frozen case selection, source/input identities, controls, fresh-process jobs, metrics, resume checks, reports.
- `research/window_dynsphere/README.md`: usage, evaluation limitations, result interpretation.
- `research/window_dynsphere/__init__.py`: explicit estimator export.
- `tests/test_window_dynsphere_data.py`, `tests/test_window_dynsphere_density.py`, `tests/test_window_dynsphere_detector.py`, `tests/test_window_dynsphere_pilot.py`: task-owned tests.

### Task 1: Causal window data and representation

**Interfaces:** `validate_sequences(x, entity_ids, times=None)` returns validated float observations, entity-local indices, and regular times. `prepare_training(x, entity_ids, times, window, max_targets, seed)` returns fitting/calibration target indices, fitting-only scaler, selected fitting targets, and metadata. `history_batches(x_scaled, entity_indices, target_indices, window, batch_size)` yields targets and histories ending immediately before each target. `HistoryEncoder(n_features, window=100, hidden_dim=32, latent_dim=8, reconstruction_weight=1.0)` exposes `encode(windows)`, `forecast(z)`, and optional `reconstruct(z)`.

- [ ] Write failing tests for entity isolation, chronological 80/20 splits, calibration-only changes leaving fitting inputs unchanged, finite input validation, regular/irregular times, minimum lengths, exact window alignment, deterministic balanced target selection, and adjacent-window velocity under sparse selection.
- [ ] Run `.venv/bin/python -m pytest tests/test_window_dynsphere_data.py -q`; confirm the tests fail because the new package is absent.
- [ ] Implement data contracts in `data.py` and module-scope neural classes in `network.py`. Fit only selected normal fitting targets, using the spec's exact losses and fixed training budget. Keep local random generators; preserve and restore externally visible RNG/thread settings where possible.
- [ ] Add tests for output dimensions, disabled reconstruction at weight zero, finite one-epoch training, deterministic replay, and no future input in histories. Small test-only dimensions and windows are allowed.
- [ ] Re-run the task tests until passing; record the red/green results. No commit is required.

### Task 2: Local dynamics and calibrated mixture density

**Interfaces:** `LocalDynamicsDensity(n_regimes=1, ridge_alpha=1.0, variance_floor=1e-6, pseudo_count=32, random_state=37).fit(context, descriptor, increments)` fits hard-assigned ridge experts and records requested/effective regimes. `.calibrate(context, descriptor, previous_x, targets)` fits weighted normal residual biases and diagonal variances. `.predict(context, descriptor, previous_x, targets=None)` returns predictive means, normalized weights, optional per-expert log densities, and negative log mixture density.

- [ ] Write failing tests for a one-regime Gaussian density against an independent hand calculation, regime probabilities summing to one, ridge penalty normalization, unpenalized intercepts, and hard cluster-specific fitting.
- [ ] Run `.venv/bin/python -m pytest tests/test_window_dynsphere_density.py -q`; verify missing implementation causes failure.
- [ ] Implement `density.py` in float64, using stable log-sum-exp and the spec's exact temperature, merging, bias, and covariance equations.
- [ ] Add tests for all clusters initially undersized, deterministic merging/ties, constant descriptors, effectively zero calibration mass, extreme distances, constant channels, and finite bounded variances.
- [ ] Re-run all density tests and compare a small mixture score with an independent direct calculation. Preserve requested/effective regime counts in metadata.

### Task 3: Estimator and matched controls

**Interfaces:** `WindowConditionedDynamicsDetector(window=100, hidden_dim=32, latent_dim=8, n_regimes=1, reconstruction_weight=1.0, epochs=10, batch_size=128, max_targets=20000, random_state=37, channel_independent=False)` provides `fit(x, entity_ids, times=None)`, `decision_function(x, entity_ids, times=None)`, and `decision_components(x, entity_ids, times=None, diagnostics=False)`. Components always include scores and validity mask; detailed arrays are opt-in. `.forecast_head_scores(...)` provides the separately calibrated compact neural-head control from the weight-zero fit.

- [ ] Write failing tests for earlier-score invariance to later observations, predictive-mean invariance to the current target, entity isolation, fitting-only scaling, and inference NaN/validity alignment.
- [ ] Run `.venv/bin/python -m pytest tests/test_window_dynsphere_detector.py -q`; verify missing estimator causes failure.
- [ ] Implement `detector.py` by composing Tasks 1 and 2. Compute latent velocities from actual adjacent windows, including under target subsampling. Freeze the encoder before local fitting. Fit normal residual calibration separately and preserve its split indices.
- [ ] Implement the channel-independent control exactly as specified: shared univariate encoder, channel-local ridge contexts, factorized density, and explicit parameter-count reporting. Cache encoder fits across K=1/K=4 with identical source/configuration and targets.
- [ ] Add tests for label-free signatures, no train/test history concatenation, short/constant inputs, K=1/K=4 behavior, zero-weight reconstruction, bounded batching, batch-size agreement, serialization/reload agreement, replay, and calibration-suffix changes leaving neural parameters unchanged.
- [ ] Run all three new test files plus `tests/test_predictive_generative_detector.py`, `tests/test_pendulum_components.py`, and `tests/test_official_baseline_local_controls.py`; require all passing before real data.

### Task 4: Frozen pilot, execution, and analysis

**Interfaces:** `freeze_pilot(output_dir: Path) -> Path` selects and writes the exact 15-case manifest before fitting. `run_case(manifest_path: Path, case_id: str, configuration_id: str, result_root: Path) -> dict` executes one fresh-process configuration and atomically records its outcome. CLI: `.venv/bin/python -m research.window_dynsphere.pilot --dry-run`, with `--run`, `--resume`, and `--aggregate` as explicit actions. A valid resume verifies provenance and score/label artifact hashes.

- [ ] Write failing tests for SHA-256 ordering of the first three Exathlon/GHL cases, minimum generation seed per all six GutenTAG mechanisms, and fixed BATADAL/HAI/PowerSystem membership.
- [ ] Run `.venv/bin/python -m pytest tests/test_window_dynsphere_pilot.py -q`; confirm failure before implementing the runner.
- [ ] Implement separate pilot loading using the existing frozen case metadata and dataset loaders through read-only imports. Hash prototype sources, consumed loader sources, package versions, input arrays, seed, configuration, and the manifest. Store outputs only under `results/window_dynsphere_v1/`.
- [ ] Add four main K/weight configurations, one channel-independent configuration, window-ridge, compact neural forecast-head scoring from the existing weight-zero fit, and a previous-neural-predictor control using the same 80/20 split. Avoid duplicate neural fits. Window-ridge uses streamed/matrix-free least squares if a full window matrix would exceed 512 MiB; record solver and tolerance.
- [ ] Add tests for resume corruption/identity changes, missing data, input hash differences, invalid warmup support, absent principal cases, and single-class post-warmup labels. These outcomes remain explicit records with no replacement sampling.
- [ ] Run the pilot tests and all new package tests. Dry-run the manifest and verify 15 planned cases, exact family/mechanism coverage, frozen seed 37, no external submissions, and unchanged extension source digest.
- [ ] Run one small synthetic smoke case in a fresh process, then the bounded pilot serially with one computational thread per process. Record fit/score times, peak memory, losses, occupancy, covariance ranges, and status. Pause additional heavy local fitting if concurrent VUS workers saturate the laptop.
- [ ] Evaluate official VUS and pointwise metrics on identical 100-row-excluded support. Re-score existing cached references when available. Record training-budget differences. Average cases within sources; preserve every mechanism and rejected case.
- [ ] Write `research/window_dynsphere/README.md` and a dated results report identifying exploratory status, all configurations, remaining issues, and whether the evidence supports a larger frozen campaign. Do not update manuscript claims automatically.

## Execution handoff

Native execution is retained from the user's earlier preference. The implementation plan requires review before coding. The approved design permits the isolated prototype and bounded pilot; full-campaign and external-HPC execution remain outside scope.

Execution status, October 23, 2026: user approved this plan. Tasks 1 through
3 and the Task 4 runner, control, provenance, resume, and test deliverables
are implemented. The independent review's substantive issues have regression
tests and fixes; the complete suite passes (229 tests). The frozen pilot has
15 cases and 120 records. Fitting and analysis are complete: 112 completed
executions on 14 cases and eight explicit PowerSystem incompatibilities.
The numerical audit verifies source hashes, input-array digests, score/label
hashes, row support, and independent pointwise metric reproduction. The
implementation ledger retains exact verification evidence. The pilot does
not support expanding this candidate into a full campaign at present.
