# Official Baselines and Expanded Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a source-traceable, leakage-free, resumable evaluation path for the requested official baselines, matched DynSphere controls, current datasets, and the 180-trajectory TSB-AD-M evaluation split, while preserving all historical results.

**Architecture:** A new `benchmarks.official_baselines` package will own provenance, dataset preparation, the model-input contract, isolated external execution, metrics, and aggregation. External implementations will run unchanged at pinned commits through repository-owned adapters; local controls will use the same prepared arrays and result schema. Labels will remain in a private evaluation directory, while external processes receive only the model-input directory.

**Tech Stack:** Python 3.10+, NumPy, pandas, scikit-learn, SciPy, PyTorch where required by upstream sources, `vus`, `uv`, JSON, subprocess execution, and `pytest`.

**Spec:** `docs/superpowers/specs/2026-10-11-official-baselines-expanded-evaluation-design.md`

## Global Constraints

- Preserve `benchmarks/deep_tsad_smoke.py`, `benchmarks/run_deep_tsad_paper.py`, all existing result files, and all manuscript tables until replacement evidence receives explicit approval.
- Write every new run beneath `results/official_baselines_v1/`; never reuse a historical result directory.
- Use PCA reconstruction, KMeans distance, DAGMM, CATCH, MTAD-GAT, OmniAnomaly, Anomaly Transformer, TranAD, `predictive_euclidean_single`, and one-regime DynSphere.
- Use author-released sources for CATCH, MTAD-GAT, OmniAnomaly, Anomaly Transformer, and TranAD; use TAB for DAGMM. TSB-AD or TAB fallbacks require explicit method identifiers and separate provenance.
- Pin every external repository to a full 40-character Git commit identifier. External checkouts remain unchanged beneath ignored `benchmarks/official_baselines/.external/`.
- Fit every scaler and model using normal training observations only. Apply fitted transformations to test observations without refitting.
- Give external adapters only `train.npy`, `test.npy`, and public metadata. Keep `labels.npy` under a separate private evaluation path.
- Return one finite anomaly score per test row. Record any length mismatch, nonfinite output, timeout, or incomplete output as a failed run.
- Use seeds 37, 38, and 39 for DynSphere-Contextual, MSL, SMAP, SMD, BATADAL, HAI-20.07, and PowerSystem. Use seed 37 for the initial complete 180-trajectory TSB-AD-M evaluation.
- Treat AUPRC and VUS-PR as primary metrics. Record AUROC, VUS-ROC, standardized partial AUROC at 5% FPR, fixed-budget event metrics, runtime, throughput, and peak memory when available.
- Aggregate trajectories within each source family first, then compute an unweighted mean across source families.
- Use formal, precise manuscript language and the project language guardrails. Avoid the prohibited terms and contrastive constructions recorded by the author.
- Commit only files named by the active task. Existing unrelated modifications in both repositories belong to the user.

## File Structure

- `benchmarks/official_baselines/provenance.py`: registry models, validation, source resolution, and run identity.
- `benchmarks/official_baselines/registry.json`: method sources, pinned commits, environments, commands, defaults, and explicit fallbacks.
- `benchmarks/official_baselines/contract.py`: private evaluation case, public model bundle, output validation, and run records.
- `benchmarks/official_baselines/datasets.py`: contextual, real-world, and TSB-AD-M loading with training-only scaling.
- `benchmarks/official_baselines/manifests/TSB-AD-M-Eva.csv`: exact official 180-file evaluation manifest.
- `benchmarks/official_baselines/manifests/provenance.json`: manifest URL, pinned source commit, SHA-256 digest, and row count.
- `benchmarks/official_baselines/local_controls.py`: PCA, KMeans, Euclidean predictor, and one-regime DynSphere scoring.
- `benchmarks/official_baselines/external.py`: isolated process execution, timeouts, logs, and explicit fallback selection.
- `benchmarks/official_baselines/adapters/`: one command-line adapter per external implementation family plus a fake adapter fixture.
- `benchmarks/official_baselines/setup_external.py`: checkout and environment setup with a reproducibility report.
- `benchmarks/official_baselines/run_expanded_evaluation.py`: matrix construction, dry runs, execution, caching, and resume.
- `benchmarks/official_baselines/metrics.py`: metric computation using private labels.
- `benchmarks/official_baselines/aggregate.py`: trajectory, family, paired, completion, and resource summaries.
- `benchmarks/official_baselines/README.md`: setup, smoke, full-run, resume, and audit commands.
- `6a24546f7bd64dcf98a38b0c/METHODOLOGY_CLARITY_ASSESSMENT.md`: post-experiment paper audit without edits to the manuscript source.

## Review Focus

1. A TSB filename prefix that contains positive labels must raise an integrity error, except for the named SMD-22 maximal-clean-prefix rule; Task 3 pins this behavior.
2. A manifest with a header, duplicate path, unknown path, or a row count other than 180 must be rejected before execution; Task 3 pins this behavior.
3. A score vector with warm-up truncation, an incorrect length, NaN, or infinity must produce a failed record and must never reach metric computation; Tasks 2 and 5 pin this behavior.
4. A mutable source reference or an unavailable preferred source must never trigger implicit substitution; Tasks 1 and 5 pin this behavior.
5. A changed source commit, hyperparameter, seed, data digest, or scaling rule must produce a new run identity, so resume can skip only an exact completed run; Tasks 1 and 7 pin this behavior.

---

### Task 1: Provenance Registry and Immutable Run Identity

**Files:**
- Create: `benchmarks/official_baselines/__init__.py`
- Create: `benchmarks/official_baselines/provenance.py`
- Create: `benchmarks/official_baselines/registry.json`
- Modify: `.gitignore`
- Test: `tests/test_official_baseline_provenance.py`

**Interfaces:**
- Consumes: repository root and the selected registry path.
- Produces: `ImplementationSpec`, `load_registry(path: Path) -> dict[str, ImplementationSpec]`, `validate_registry(specs) -> None`, `resolve_local_revision(repo: Path) -> str`, and `run_identity(case_identity: str, implementation: ImplementationSpec, hyperparameters: Mapping[str, object], seed: int, data_digest: str, scaling_rule: str) -> str`.

- [ ] **Step 1: Write failing registry tests.** Assert that the ten required displayed methods exist, external revisions match `[0-9a-f]{40}`, commands are argument lists, local revisions resolve to the current Git commit, and missing fields or branch names raise `ValueError`.
- [ ] **Step 2: Write the resume-identity risk test.** Assert that changing any one of commit, hyperparameters, seed, data digest, or scaling rule changes the SHA-256 run identity.
- [ ] **Step 3: Run `python3 -m pytest tests/test_official_baseline_provenance.py -q`.** Expected result: import or assertion failure because the package is absent.
- [ ] **Step 4: Resolve and record immutable upstream commits.** Use `git ls-remote <repository-url> HEAD` for each preferred source, copy each returned 40-character commit into `registry.json`, and include no placeholder or mutable revision.
- [ ] **Step 5: Implement the registry models and validation.** Local entries use `source_kind: "local"` and resolve the current repository commit at run time; external entries use `source_kind: "external"` and a pinned commit. Add explicit fallback entries with distinct identifiers.
- [ ] **Step 6: Ignore only generated external state.** Add `benchmarks/official_baselines/.external/`, `.envs/`, and `.work/` to `.gitignore`.
- [ ] **Step 7: Run the provenance tests.** Expected result: all tests pass and the registry contains exact source identities.
- [ ] **Step 8: Commit.** Stage the five named files and commit with `feat: add official baseline provenance registry`.

### Task 2: Private Evaluation Case and Public Model-Input Contract

**Files:**
- Create: `benchmarks/official_baselines/contract.py`
- Test: `tests/test_official_baseline_contract.py`

**Interfaces:**
- Consumes: NumPy arrays and dataset metadata.
- Produces: `EvaluationCase`, `ModelInputBundle`, `write_evaluation_case(case: EvaluationCase, root: Path) -> ModelInputBundle`, `validate_scores(path: Path, expected_rows: int) -> np.ndarray`, and `write_run_record(path: Path, record: Mapping[str, object]) -> None`.

- [ ] **Step 1: Write the bundle-isolation test.** Construct a six-row fixture and assert that `private/labels.npy` exists, `model_input/` contains exactly `train.npy`, `test.npy`, and `metadata.json`, and public metadata contains no labels or label path.
- [ ] **Step 2: Write output-validation tests.** Assert that exact-length finite scores pass; truncated, oversized, multidimensional, NaN, infinity, and missing score files raise `OutputValidationError`.
- [ ] **Step 3: Run `python3 -m pytest tests/test_official_baseline_contract.py -q`.** Expected result: import failure.
- [ ] **Step 4: Implement the dataclasses and atomic writers.** Write temporary files inside the target run directory and replace final files only after serialization succeeds. Include schema version `official-baseline-v1`.
- [ ] **Step 5: Implement score validation and normalized run records.** Required status values are `pending`, `running`, `complete`, `failed`, `timeout`, and `incompatible`.
- [ ] **Step 6: Run the contract tests.** Expected result: all tests pass.
- [ ] **Step 7: Commit.** Stage the contract and its test and commit with `feat: define isolated experiment contract`.

### Task 3: Leakage-Free Dataset Preparation and Official Manifest

**Files:**
- Create: `benchmarks/official_baselines/datasets.py`
- Create: `benchmarks/official_baselines/manifests/TSB-AD-M-Eva.csv`
- Create: `benchmarks/official_baselines/manifests/provenance.json`
- Test: `tests/test_official_baseline_datasets.py`

**Interfaces:**
- Consumes: `load_tsb_csv`, `declared_prefix_indices`, contextual files, the existing real-world loaders, and the official manifest.
- Produces: `load_contextual_case(seed: int) -> EvaluationCase`, `load_real_world_case(dataset_id: str, seed: int) -> EvaluationCase`, `load_tsb_cases(data_root: Path, manifest_path: Path, seed: int) -> Iterator[EvaluationCase]`, `read_eval_manifest(path: Path) -> tuple[str, ...]`, and `fit_training_scaler(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, object]]`.

- [ ] **Step 1: Write manifest integrity tests.** Assert 180 unique filenames, a recognized header, stable order, official SHA-256 metadata, rejection of duplicates, rejection of missing local files, and rejection of any extra row.
- [ ] **Step 2: Write split-integrity tests.** Assert declared normal-prefix selection for an MSL fixture, an error for a contaminated ordinary prefix, and the documented maximal-clean-prefix result only for `078_SMD_id_22_Facility_tr_500_1st_326.csv`.
- [ ] **Step 3: Write the scaling test.** Use a training mean of 1 and a test mean of 101; assert the scaler center equals the training statistic and the transformed test remains far from zero.
- [ ] **Step 4: Run `python3 -m pytest tests/test_official_baseline_datasets.py -q`.** Expected result: import failure.
- [ ] **Step 5: Copy the official manifest and record provenance.** Pin the upstream manifest commit, store its direct source URL and SHA-256 digest, and preserve its 180 data rows exactly.
- [ ] **Step 6: Implement dataset loaders.** Reuse parsing helpers from `benchmarks/tsb_ad_m/run_tsb_ad_m.py`; replace the older full-series fit path with declared-prefix training. Store row identifiers and the exceptional SMD rule in private metadata.
- [ ] **Step 7: Run dataset tests plus `python3 -m pytest tests/test_strategy_a_real_world.py -q`.** Expected result: all tests pass.
- [ ] **Step 8: Commit.** Stage the loader, manifest, provenance record, and test and commit with `feat: add leakage-free expanded datasets`.

### Task 4: Local Baselines and Matched Controls

**Files:**
- Create: `benchmarks/official_baselines/local_controls.py`
- Test: `tests/test_official_baseline_local_controls.py`

**Interfaces:**
- Consumes: scaled `EvaluationCase.train`, `EvaluationCase.test`, and seed.
- Produces: `score_pca_reconstruction(train, test, seed) -> tuple[np.ndarray, dict]`, `score_kmeans_distance(train, test, seed) -> tuple[np.ndarray, dict]`, `score_euclidean_predictor(train, test, seed) -> tuple[np.ndarray, dict]`, `score_one_regime_dynsphere(train, test, seed) -> tuple[np.ndarray, dict]`, and `LOCAL_SCORERS`.

- [ ] **Step 1: Write hand-computed PCA and KMeans tests.** PCA keeps the smallest component count reaching 95% training variance, bounded to `[1, min(n_train - 1, n_features)]`; KMeans uses `min(4, n_train)` clusters, `n_init=20`, and scores distance to the nearest center.
- [ ] **Step 2: Write matched-control tests.** Assert that the Euclidean control invokes the existing `predictive_euclidean_single` scorer and that one-regime DynSphere instantiates `MultiConceptPendulumDetector(n_concepts=1, random_state=seed)`.
- [ ] **Step 3: Write common output tests.** Each scorer must return finite scores with `len(test)` rows and metadata containing the fixed configuration.
- [ ] **Step 4: Run `python3 -m pytest tests/test_official_baseline_local_controls.py -q`.** Expected result: import failure.
- [ ] **Step 5: Implement the four local scorers.** Use `method_scorer("predictive_euclidean_single")` for the retained one-step predictor and existing DynSphere code for the one-regime control. Do not copy detector logic into the adapter.
- [ ] **Step 6: Run the local-control tests.** Expected result: all tests pass.
- [ ] **Step 7: Commit.** Stage the scorer module and test and commit with `feat: add matched local controls`.

### Task 5: Isolated External Execution

**Files:**
- Create: `benchmarks/official_baselines/external.py`
- Create: `benchmarks/official_baselines/adapters/__init__.py`
- Create: `benchmarks/official_baselines/adapters/fake_adapter.py`
- Test: `tests/test_official_baseline_external.py`

**Interfaces:**
- Consumes: `ImplementationSpec`, `ModelInputBundle`, output directory, seed, timeout, and optional explicit fallback identifier.
- Produces: `ExternalRunResult` and `run_external(spec: ImplementationSpec, bundle: ModelInputBundle, output_dir: Path, seed: int, timeout_seconds: int, fallback_id: str | None = None) -> ExternalRunResult`.

- [ ] **Step 1: Write fake-adapter success and privacy tests.** The fixture records its received file list and writes deterministic scores; assert that it receives no label file or private directory argument.
- [ ] **Step 2: Write process-risk tests.** Exercise timeout, nonzero exit, missing output, truncated warm-up output, nonfinite scores, and a partial `run.json`; assert a normalized failed or timeout record for each condition.
- [ ] **Step 3: Write fallback tests.** An unavailable preferred source must return `incompatible` unless `fallback_id` is supplied; an explicit fallback must record both requested and executed identities.
- [ ] **Step 4: Run `python3 -m pytest tests/test_official_baseline_external.py -q`.** Expected result: import failure.
- [ ] **Step 5: Implement argument-list subprocess execution.** Use no shell, pass deterministic seed environment variables, capture bounded stdout and stderr tails, and write status transitions atomically.
- [ ] **Step 6: Run the external execution tests.** Expected result: all tests pass.
- [ ] **Step 7: Commit.** Stage the external runner, adapter fixture, and test and commit with `feat: add isolated external execution`.

### Task 6: Official Source Setup and Real Adapters

**Files:**
- Create: `benchmarks/official_baselines/setup_external.py`
- Create: `benchmarks/official_baselines/adapters/tab_dagmm.py`
- Create: `benchmarks/official_baselines/adapters/catch.py`
- Create: `benchmarks/official_baselines/adapters/mtad_gat.py`
- Create: `benchmarks/official_baselines/adapters/omni_anomaly.py`
- Create: `benchmarks/official_baselines/adapters/anomaly_transformer.py`
- Create: `benchmarks/official_baselines/adapters/tranad.py`
- Test: `tests/test_official_baseline_adapters.py`

**Interfaces:**
- Consumes: registry entries and the Task 2 model-input directory.
- Produces: `prepare_source(method_id: str, external_root: Path) -> SetupRecord`, `verify_source(method_id: str, external_root: Path) -> SetupRecord`, and adapter CLIs accepting `--input-dir`, `--output-dir`, `--seed`, and `--config-json`.

- [ ] **Step 1: Write checkout and environment-plan tests.** With local fake Git repositories, assert detached checkout at the pinned commit, refusal to continue with modified upstream files, reuse of a verified environment, and a structured incompatibility report when the declared Python or dependency set cannot be created.
- [ ] **Step 2: Write adapter contract tests using stub upstream modules.** For each of the six adapters, assert training uses `train.npy`, scoring uses `test.npy`, score direction means larger is more anomalous, output length matches test rows, and warm-up alignment is recorded.
- [ ] **Step 3: Run `python3 -m pytest tests/test_official_baseline_adapters.py -q`.** Expected result: import failure.
- [ ] **Step 4: Implement source setup.** Create one `uv` environment per implementation family, install the pinned checkout's declared dependencies, record Python and package versions, and leave upstream source unchanged.
- [ ] **Step 5: Implement DAGMM and CATCH adapters.** Map the canonical arrays to TAB DAGMM and the author CATCH entry points, preserving fixed registry hyperparameters.
- [ ] **Step 6: Implement MTAD-GAT and deep temporal adapters.** Apply each upstream window convention, map window scores back to test rows with the declared warm-up policy, and record score interpretation.
- [ ] **Step 7: Run adapter unit tests.** Expected result: all stub-backed tests pass without downloading external sources.
- [ ] **Step 8: Commit.** Stage setup, six adapters, and tests and commit with `feat: integrate source-traceable baseline adapters`.

### Task 7: Resumable Experiment Matrix

**Files:**
- Create: `benchmarks/official_baselines/run_expanded_evaluation.py`
- Test: `tests/test_official_baseline_runner.py`

**Interfaces:**
- Consumes: provenance registry, dataset loaders, local scorers, external runner, method filters, dataset filters, manifest path, seeds, timeout, and result root.
- Produces: `ExperimentJob`, `build_matrix(config: RunConfig) -> tuple[ExperimentJob, ...]`, `execute_job(job: ExperimentJob) -> dict[str, object]`, and CLI options `--methods`, `--datasets`, `--seeds`, `--tsb-manifest`, `--dry-run`, `--resume`, `--timeout-seconds`, and `--result-root`.

- [ ] **Step 1: Write matrix tests.** Assert three seeds for contextual and the six current datasets, seed 37 for remaining TSB evaluation trajectories, deterministic ordering, and deduplication when MSL, SMAP, or SMD jobs occur in both selections.
- [ ] **Step 2: Write resume tests.** A complete matching run is skipped; pending, failed, timeout, corrupt, or identity-mismatched runs execute again. A changed revision, hyperparameter, seed, data digest, or scaling rule must yield a distinct directory.
- [ ] **Step 3: Write preservation tests.** Point the runner at a fixture containing historical files and assert dry-run and execution create content only inside the new versioned result root.
- [ ] **Step 4: Run `python3 -m pytest tests/test_official_baseline_runner.py -q`.** Expected result: import failure.
- [ ] **Step 5: Implement matrix construction and execution.** Evaluate labels only after validated scores are present. Keep per-job public input, private evaluation data, logs, scores, and run record in separate subdirectories.
- [ ] **Step 6: Implement dry-run reporting.** Print job counts by method, dataset, source family, seed, and implementation identity; return a nonzero status for missing data, invalid provenance, or an invalid manifest.
- [ ] **Step 7: Run runner tests.** Expected result: all tests pass.
- [ ] **Step 8: Commit.** Stage the runner and test and commit with `feat: add resumable expanded evaluation runner`.

### Task 8: Metrics and Hierarchical Aggregation

**Files:**
- Create: `benchmarks/official_baselines/metrics.py`
- Create: `benchmarks/official_baselines/aggregate.py`
- Test: `tests/test_official_baseline_aggregation.py`

**Interfaces:**
- Consumes: private labels, validated scores, and complete run records.
- Produces: `compute_metrics(labels, scores, event_budget, vus_window) -> dict[str, object]`, `aggregate_runs(result_root: Path, output_dir: Path, bootstrap_seed: int = 20261011, bootstrap_replicates: int = 10000) -> AggregateArtifacts`, and CSV/JSON/LaTeX outputs.

- [ ] **Step 1: Write metric tests.** Compare AUPRC, AUROC, and 5% standardized partial AUROC with direct scikit-learn calls; require the official `vus` backend for manuscript-ready status; test a fixed-budget event fixture.
- [ ] **Step 2: Write weighting tests.** Use unequal trajectory counts across two families and assert trajectory means within family followed by an unweighted two-family mean.
- [ ] **Step 3: Write paired-analysis tests.** Assert pairing by dataset, trajectory, and seed; report paired mean difference, 95% bootstrap interval, win rate, and rank-biserial effect size. Exclude incomplete pairs and report their count.
- [ ] **Step 4: Write completion tests.** Assert separate totals for complete, failed, timeout, and incompatible runs and prevent manuscript-ready table generation when a required method has no valid run.
- [ ] **Step 5: Run `python3 -m pytest tests/test_official_baseline_aggregation.py -q`.** Expected result: import failure.
- [ ] **Step 6: Implement metrics and aggregation.** Reuse `range_metrics.range_vus_scores` with the official backend installed; record backend and window in every row.
- [ ] **Step 7: Run aggregation tests.** Expected result: all tests pass.
- [ ] **Step 8: Commit.** Stage metrics, aggregation, and tests and commit with `feat: add hierarchical evaluation summaries`.

### Task 9: Reproduction Documentation and Full Local Verification

**Files:**
- Create: `benchmarks/official_baselines/README.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: Tasks 1 through 8.
- Produces: exact setup, validation, smoke, execution, resume, aggregation, and provenance-audit commands.

- [ ] **Step 1: Document the historical status.** State that compact local proxies remain audit artifacts and that only `official_baselines_v1` outputs qualify for the new comparison.
- [ ] **Step 2: Document the staged commands.** Include registry validation, external setup, matrix dry run, fake-adapter test, one-trajectory real smoke tests, current-dataset execution, 180-trajectory execution, resume, and aggregation.
- [ ] **Step 3: Run `python3 -m pytest -q`.** Expected result: the complete repository test suite passes.
- [ ] **Step 4: Run a dry matrix validation.** Expected result: all ten requested methods are listed, 180 TSB manifest entries are recognized, and no historical path is targeted.
- [ ] **Step 5: Commit.** Stage only the two documentation files and commit with `docs: add official baseline reproduction guide`.

### Task 10: Staged Real Execution and Result Audit

**Files:**
- Generate: `results/official_baselines_v1/setup/provenance_report.json`
- Generate: `results/official_baselines_v1/runs/`
- Generate: `results/official_baselines_v1/aggregate/`
- Generate: `results/official_baselines_v1/EXECUTION_STATUS.md`

**Interfaces:**
- Consumes: the completed runner, local datasets, pinned external sources, and isolated environments.
- Produces: validated per-run records, scores, logs, aggregate tables, paired analyses, and a human-readable execution ledger. Generated results remain outside Git unless the user later requests selected artifacts to be tracked.

- [ ] **Step 1: Prepare and verify all preferred external sources.** Record exact commits, environments, package versions, hardware, and any reproducible incompatibility.
- [ ] **Step 2: Run one real trajectory per method.** Use seed 37, validate score direction and row alignment, inspect score distributions, and record peak resource use.
- [ ] **Step 3: Resolve adapter defects through the systematic debugging workflow.** Repeat the affected smoke test and the relevant unit suite after each correction; commit each adapter correction separately.
- [ ] **Step 4: Run contextual plus MSL, SMAP, SMD, BATADAL, HAI-20.07, and PowerSystem for seeds 37, 38, and 39.** Resume from existing complete identities and preserve all failed records.
- [ ] **Step 5: Aggregate and inspect the current-dataset stage.** Verify completion counts, official VUS provenance, paired coverage, score direction, and any extreme metric or runtime value before expanding execution.
- [ ] **Step 6: Run all 180 official TSB-AD-M evaluation trajectories at seed 37.** Use resume, per-run timeouts, and source-family progress reporting.
- [ ] **Step 7: Produce the final aggregate artifacts.** Generate per-run, per-trajectory, per-family, unweighted family macro, paired interval, effect-size, runtime, memory, completion, and incompatibility outputs.
- [ ] **Step 8: Compare new outputs with preserved historical tables.** Record differences in `EXECUTION_STATUS.md`; make no manuscript table replacement in this task.

### Task 11: Methodology-Clarity Assessment

**Files:**
- Create: `../6a24546f7bd64dcf98a38b0c/METHODOLOGY_CLARITY_ASSESSMENT.md`

**Interfaces:**
- Consumes: `main.tex`, `vldb2027/dynsphere_vldb.tex`, figures, algorithms, executed run metadata, adapter provenance, and final aggregate outputs.
- Produces: a location-specific paper audit with severity, evidence, and proposed correction for each ambiguity. This task does not edit either manuscript source.

- [ ] **Step 1: Trace the anomaly definition and deployment assumptions.** Cite exact sections, equations, and figure labels; test whether the contextual illustration matches the formal score and executed input contract.
- [ ] **Step 2: Trace the complete method.** Cover representation, normalization, derivatives, concepts, transitions, score channels, score calibration, composite score, training, inference, causality, and complexity.
- [ ] **Step 3: Trace experimental reproducibility.** Compare manuscript statements with the registry, dataset metadata, preprocessing, metrics, seeds, and aggregate weighting actually used.
- [ ] **Step 4: Classify every finding.** Use `blocking`, `major`, `minor`, or `clear`; include the manuscript location, observed ambiguity, supporting code or run artifact, and a concise proposed correction.
- [ ] **Step 5: Apply the project language guardrails to the assessment.** Run targeted text searches for prohibited terms and contrastive constructions, then revise any matches.
- [ ] **Step 6: Verify links and locations.** Confirm every cited equation, figure, table, subsection, file, and result path exists.
- [ ] **Step 7: Commit only the assessment in the paper repository.** Use commit message `docs: assess DynSphere methodology clarity` and leave manuscript edits for a separately approved task.

## Completion Gate

Before reporting completion:

1. Run `python3 -m pytest -q` and retain the passing output.
2. Validate all registry entries and reproduce the dry-run matrix count.
3. Confirm every preferred adapter has a successful real smoke test or a structured, reproducible incompatibility record.
4. Confirm the 180-file manifest, run completion report, aggregate outputs, and private-label isolation.
5. Confirm historical results and manuscript tables are byte-for-byte unchanged.
6. Review `METHODOLOGY_CLARITY_ASSESSMENT.md` against the executed evidence and the author language guardrails.
