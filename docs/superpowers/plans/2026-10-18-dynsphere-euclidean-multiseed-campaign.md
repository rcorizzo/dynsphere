# DynSphere Euclidean Variant and Bundled Multi-Seed Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the matched DynSphere-E geometry variant and run one reproducible three-seed campaign containing DynSphere-family methods, local controls, and source-traceable external baselines on the seven paper datasets.

**Architecture:** A shared geometry-aware DynSphere core will preserve the existing spherical API and add a Euclidean implementation with matched concepts, dynamics, calibration, weights, and propagation. The experiment system will expose one `paper-multiseed` profile, split execution into H100, external-CPU, and local-CPU component manifests, then merge them into one 2,457-run umbrella manifest for aggregation.

**Tech Stack:** Python 3.9+, NumPy, pandas, scikit-learn, SciPy, PyTorch for external deep methods, official `vus`, JSON, Bash, IBM LSF, SLURM, and pytest.

**Spec:** `docs/superpowers/specs/2026-10-18-dynsphere-euclidean-variant-design.md`

## Global Constraints

- Preserve the existing `MultiConceptPendulumDetector` constructor, default values, score semantics, component names, and numerical output within strict floating-point tolerance.
- Add `MultiConceptEuclideanDetector` as the matched Euclidean implementation. Keep `predictive_euclidean_single` as a separate predictive baseline.
- Use four concepts for DynSphere-S and DynSphere-E and one concept for DynSphere-S1 and DynSphere-E1.
- Keep all existing result directories and campaign manifests. Write new runs beneath `results/official_baselines_v1/` using distinct identities and manifests.
- Evaluate DynSphere-Contextual, MSL, SMAP, SMD, BATADAL, HAI-20.07, and PowerSystem with seeds 37, 38, and 39.
- Include 13 methods: PCA reconstruction, KMeans distance, DAGMM, CATCH, MTAD-GAT, TSB-AD OmniAnomaly, Anomaly Transformer, TranAD, the one-step Euclidean predictor, DynSphere-S, DynSphere-E, DynSphere-S1, and DynSphere-E1.
- Generate all final multi-seed runs from one committed repository revision so every seed for a method has the same adapter revision.
- Use AUPRC and VUS-PR as primary metrics. Use AUROC, VUS-ROC, standardized partial AUROC at five percent false-positive rate, and event metrics as secondary evidence.
- Use no evaluation labels for configuration selection, geometry selection, concept-count selection, score weights, or propagation width.
- Preserve unrelated working-tree changes and commit only files named by the active task.
- Follow the project language guardrails in all documentation and generated manuscript text.

## File Structure

- `pendulum_detector.py`: shared geometry-aware DynSphere implementation and backward-compatible public detector classes.
- `tests/test_pendulum_components.py`: spherical regression snapshot, Euclidean definitions, component additivity, boundary behavior, and one-concept behavior.
- `benchmarks/official_baselines/local_controls.py`: local scoring entry points for DynSphere-S, DynSphere-E, DynSphere-S1, and DynSphere-E1.
- `benchmarks/official_baselines/registry.json`: immutable identities and fixed configurations for the four DynSphere methods.
- `benchmarks/official_baselines/campaign.py`: `paper-multiseed` campaign profile.
- `benchmarks/official_baselines/aggregate.py`: named factorial contrasts and final campaign summaries.
- `benchmarks/official_baselines/hpc/au_lsf_multiseed_h100_array.bsub`: four GPU methods, 756 jobs over 16 shards.
- `benchmarks/official_baselines/hpc/au_lsf_multiseed_cpu_array.bsub`: two external CPU methods, 378 jobs over 16 shards.
- `benchmarks/official_baselines/hpc/au_lsf_multiseed_local_array.bsub`: seven local methods, 1,323 jobs over 32 shards.
- `benchmarks/official_baselines/hpc/au_lsf_multiseed_aggregate.bsub`: component-manifest merge and final aggregation.
- `benchmarks/official_baselines/hpc/plgrid_slurm_array.sh`: optional method filtering for equivalent SLURM component submissions.
- `benchmarks/official_baselines/HPC_EXECUTION.md`: exact AU and PLGrid setup, submission, monitoring, recovery, and aggregation commands.
- `tests/test_official_baseline_local_controls.py`, `tests/test_official_baseline_provenance.py`, `tests/test_official_baseline_campaign.py`, `tests/test_official_baseline_aggregation.py`, and `tests/test_official_baseline_hpc_scripts.py`: integration and campaign coverage.

## Review Focus

1. A row with a zero PCA norm must produce finite spherical and Euclidean scores; Task 1 adds the regression case.
2. An entity boundary must reset derivatives and transitions in both geometries; Task 2 adds the multi-entity assertion.
3. A concept with too few samples must use the existing all-training fallback in both geometries; Task 2 adds the small-concept test.
4. A scheduler rerun with an existing manifest must preserve identical campaign membership and reject a changed method set; Task 4 adds the manifest-collision test.
5. A component campaign with incomplete or mixed-revision records must keep the umbrella campaign outside manuscript-ready status; Task 5 adds the readiness assertion.

---

### Task 1: Freeze the Spherical Detector and Introduce the Shared Core

**Files:**
- Modify: `pendulum_detector.py`
- Modify: `tests/test_pendulum_components.py`

**Interfaces:**
- Consumes: existing `MultiConceptPendulumDetector`, NumPy arrays, entity identifiers, and optional times.
- Produces: private geometry operations used by both public detectors while retaining the full existing `MultiConceptPendulumDetector` interface.

- [ ] **Step 1: Add a spherical characterization test.** Use the deterministic sequence fixture with `n_concepts=3`, `latent_dim=4`, `propagation_radius=0`, and seed 37. Assert the complete score vector and every existing component against values captured from the current implementation with `rtol=1e-10` and `atol=1e-10`.
- [ ] **Step 2: Add the zero-latent-norm regression case.** Fit and score a finite constant or symmetric sequence and assert finite values for every reported channel.
- [ ] **Step 3: Run `.venv/bin/python -m pytest tests/test_pendulum_components.py -q`.** Expected result: both characterization tests pass against the unchanged detector.
- [ ] **Step 4: Refactor the detector into a private shared core.** Limit geometry-specific behavior to state construction, derivative projection, position distance, equilibrium construction, and potential calculation. Retain all spherical public names and constructor defaults.
- [ ] **Step 5: Run the component tests again.** Expected result: the frozen spherical arrays remain within the declared tolerance.
- [ ] **Step 6: Commit.** Stage only `pendulum_detector.py` and `tests/test_pendulum_components.py`; commit with `refactor: share DynSphere geometry core`.

### Task 2: Implement DynSphere-E and the Two-by-Two Geometry Controls

**Files:**
- Modify: `pendulum_detector.py`
- Modify: `tests/test_pendulum_components.py`

**Interfaces:**
- Consumes: the Task 1 shared core.
- Produces: `MultiConceptEuclideanDetector(n_concepts: int = 4, latent_dim: int = 5, ..., random_state: int = 37)` with `fit`, `decision_components`, `decision_function`, and `predict` methods matching the spherical API.

- [ ] **Step 1: Write failing Euclidean-definition tests.** Assert unnormalized PCA states, ordinary Euclidean derivatives, Euclidean center distance, and potential `0.5 * ||y - g_E||^2` on hand-computed fixtures.
- [ ] **Step 2: Write failing common-behavior tests.** For both public detectors, assert finite row-aligned scores, exact additive reconstruction of the pre-propagation score, reset behavior at entity boundaries, and constant transition evidence with one concept.
- [ ] **Step 3: Write the small-concept fallback test.** Request more concepts than a small regime can support and assert finite scores plus the same all-training fallback policy under both geometries.
- [ ] **Step 4: Write the magnitude-distinction test.** Apply a controlled positive latent-magnitude change and assert unchanged spherical direction together with changed Euclidean position and potential evidence.
- [ ] **Step 5: Run `.venv/bin/python -m pytest tests/test_pendulum_components.py -q`.** Expected result: the new Euclidean tests fail because `MultiConceptEuclideanDetector` is absent.
- [ ] **Step 6: Implement `MultiConceptEuclideanDetector`.** Use the exact definitions and fixed weights in the approved specification. Return `position` and `euclidean_dynamics` as geometry-specific component names and retain the common channel names.
- [ ] **Step 7: Run the component tests.** Expected result: all spherical and Euclidean tests pass.
- [ ] **Step 8: Commit.** Stage the two named files and commit with `feat: add matched Euclidean DynSphere`.

### Task 3: Register the DynSphere Family and Define the Final Matrix

**Files:**
- Modify: `benchmarks/official_baselines/local_controls.py`
- Modify: `benchmarks/official_baselines/registry.json`
- Modify: `benchmarks/official_baselines/campaign.py`
- Modify: `tests/test_official_baseline_local_controls.py`
- Modify: `tests/test_official_baseline_provenance.py`
- Modify: `tests/test_official_baseline_campaign.py`

**Interfaces:**
- Consumes: `MultiConceptPendulumDetector`, `MultiConceptEuclideanDetector`, and the existing experiment contract.
- Produces: `score_dynsphere`, `score_euclidean_dynsphere`, `score_one_regime_dynsphere`, `score_one_regime_euclidean_dynsphere`, four registry entries, and the `paper-multiseed` profile.

- [ ] **Step 1: Write failing local-scorer tests.** Assert that the four method identifiers instantiate the correct geometry and concept count, return finite aligned scores, and record `geometry`, `n_concepts`, fixed weights, and propagation radius in metadata.
- [ ] **Step 2: Write failing registry tests.** Require display names `DynSphere-S`, `DynSphere-E`, `DynSphere-S1`, and `DynSphere-E1`, local adapter commands, and fixed default hyperparameters.
- [ ] **Step 3: Write the campaign-matrix test.** Assert that `paper-multiseed` contains 13 methods, seven dataset groups, seeds 37 through 39, 63 cases per seed and method, and 2,457 logical jobs.
- [ ] **Step 4: Run `.venv/bin/python -m pytest tests/test_official_baseline_local_controls.py tests/test_official_baseline_provenance.py tests/test_official_baseline_campaign.py -q`.** Expected result: failures for the absent scorers, registry entries, and profile.
- [ ] **Step 5: Implement the four local scorers.** Keep detector logic in `pendulum_detector.py`; adapters construct the requested class and return scores plus configuration metadata.
- [ ] **Step 6: Add the four registry entries and `paper-multiseed` profile.** The profile uses all 13 methods, `CURRENT_DATASETS`, seeds `(37, 38, 39)`, and a 14,400-second per-run timeout.
- [ ] **Step 7: Run the three targeted test modules.** Expected result: all tests pass and the dry matrix count is exactly 2,457.
- [ ] **Step 8: Commit.** Stage the six named files and commit with `feat: register DynSphere geometry campaign`.

### Task 4: Add Bundled AU and PLGrid Schedulers

**Files:**
- Create: `benchmarks/official_baselines/hpc/au_lsf_multiseed_h100_array.bsub`
- Create: `benchmarks/official_baselines/hpc/au_lsf_multiseed_cpu_array.bsub`
- Create: `benchmarks/official_baselines/hpc/au_lsf_multiseed_local_array.bsub`
- Create: `benchmarks/official_baselines/hpc/au_lsf_multiseed_aggregate.bsub`
- Modify: `benchmarks/official_baselines/hpc/plgrid_slurm_array.sh`
- Modify: `tests/test_official_baseline_hpc_scripts.py`

**Interfaces:**
- Consumes: `paper-multiseed`, `run_au_array.sh`, three component method sets, and the standard result root.
- Produces: three distinct component manifests and `paper-multiseed-au-complete.json`.

- [ ] **Step 1: Write failing scheduler-content tests.** Assert the H100 script selects CATCH, MTAD-GAT, Anomaly Transformer, and TranAD; the external CPU script selects DAGMM and TSB-AD OmniAnomaly; the local script selects the seven local methods; every component uses a distinct manifest.
- [ ] **Step 2: Write shard and queue tests.** Require 16 H100 shards, 16 external-CPU shards, 32 local shards, the AU `h100` queue for GPU work, and the AU `long` queue for both CPU groups.
- [ ] **Step 3: Write merge and collision tests.** Assert merge order, 2,457 unique logical jobs, and rejection when an existing component-manifest path contains a different method set or repository revision.
- [ ] **Step 4: Write the PLGrid filtering test.** Require optional `DYNSPHERE_METHODS` parsing and forwarding to `run_campaign --methods` while retaining zero-based SLURM array indexing.
- [ ] **Step 5: Run `.venv/bin/python -m pytest tests/test_official_baseline_hpc_scripts.py -q`.** Expected result: failures for missing scripts and filtering.
- [ ] **Step 6: Implement the four AU scripts and PLGrid filtering.** Set `DYNSPHERE_PROFILE=paper-multiseed`, explicit method lists, component manifest paths, device selection, and shard counts. The aggregate script merges all three manifests before invoking `aggregate_campaign`.
- [ ] **Step 7: Run the HPC script tests and `bash -n benchmarks/official_baselines/hpc/*.bsub benchmarks/official_baselines/hpc/*.sh`.** Expected result: tests pass and every script has valid shell syntax.
- [ ] **Step 8: Commit.** Stage the six scheduler files and the test; commit with `feat: bundle multi-seed HPC campaign`.

### Task 5: Add Factorial Contrasts and Strict Readiness Checks

**Files:**
- Modify: `benchmarks/official_baselines/aggregate.py`
- Modify: `benchmarks/official_baselines/aggregate_campaign.py`
- Modify: `tests/test_official_baseline_aggregation.py`

**Interfaces:**
- Consumes: the complete umbrella campaign manifest and validated per-run records.
- Produces: `geometry_factorial.csv`, existing aggregate tables, and a readiness report that requires all 13 methods and a single adapter revision per method.

- [ ] **Step 1: Write failing factorial tests.** Use a small paired fixture and require four named contrasts: `E4-S4`, `E1-S1`, `S4-S1`, and `E4-E1`, each with paired mean difference, confidence interval, win rate, and pair count.
- [ ] **Step 2: Write failing readiness tests.** Missing runs, missing methods, metric errors, fallback VUS, or multiple adapter revisions for one method must set `manuscript_ready` to false and identify the exact affected records.
- [ ] **Step 3: Run `.venv/bin/python -m pytest tests/test_official_baseline_aggregation.py -q`.** Expected result: failures for absent factorial output and revision validation.
- [ ] **Step 4: Implement the named contrast table.** Reuse the existing paired comparison machinery and filter it through a fixed method-pair mapping; do not add a second bootstrap implementation.
- [ ] **Step 5: Extend readiness reporting.** Record expected and observed methods, per-method adapter revisions, incomplete identities, metric warnings, and the factorial artifact path.
- [ ] **Step 6: Run the aggregation tests.** Expected result: all tests pass.
- [ ] **Step 7: Commit.** Stage the three named files and commit with `feat: aggregate DynSphere factorial evidence`.

### Task 6: Document, Verify, and Prepare Execution Handoff

**Files:**
- Modify: `benchmarks/official_baselines/HPC_EXECUTION.md`
- Modify: `benchmarks/official_baselines/README.md`
- Modify: `README.md`
- Modify: `post-ICDM-rev.md`

**Interfaces:**
- Consumes: Tasks 1 through 5.
- Produces: exact AU and PLGrid commands, campaign counts, monitoring commands, completion checks, aggregation commands, and the updated revision ledger.

- [ ] **Step 1: Document the final method table and run counts.** State 13 methods, seven dataset groups, three seeds, 63 cases per seed and method, and 2,457 logical runs.
- [ ] **Step 2: Document AU submission.** Include setup verification, dry matrix validation, three component `bsub` commands, monitoring, selective rerun, and aggregation.
- [ ] **Step 3: Document PLGrid submission.** Include environment variables, three method-filtered SLURM submissions, manifest paths, monitoring, selective rerun, synchronization, and aggregation.
- [ ] **Step 4: Run the language-guardrail scan.** Search the four documentation files for the prohibited vocabulary and contrastive constructions and revise every occurrence introduced by this task.
- [ ] **Step 5: Run the focused suite.** Execute `.venv/bin/python -m pytest tests/test_pendulum_components.py tests/test_official_baseline_local_controls.py tests/test_official_baseline_provenance.py tests/test_official_baseline_campaign.py tests/test_official_baseline_aggregation.py tests/test_official_baseline_hpc_scripts.py -q`; require all tests to pass.
- [ ] **Step 6: Run the broader regression suite.** Execute `.venv/bin/python -m pytest tests/test_official_baseline_datasets.py tests/test_official_baseline_runner.py tests/test_strategy_a_real_world.py -q`; require all tests to pass.
- [ ] **Step 7: Generate a dry campaign manifest in a temporary result root.** Require 2,457 jobs with method counts of 189 each and seed counts of 819 each. Delete only the temporary result root after validation.
- [ ] **Step 8: Commit.** Stage the four documentation files and commit with `docs: add bundled multi-seed execution guide`.

## Execution Gate

The campaign may be submitted only after:

1. the spherical regression snapshot passes;
2. every focused and broader regression test passes;
3. the dry matrix reports exactly 2,457 logical jobs;
4. all component manifests are absent or byte-identical to the newly generated documents;
5. the AU environment probe confirms CUDA for H100 methods;
6. the setup provenance report validates all external source revisions; and
7. the code revision used to generate the manifests is materialized on the execution cluster.

After execution, merge the H100, external-CPU, and local manifests, aggregate
the umbrella campaign, inspect readiness, and generate paper tables only when
all 13 methods have complete matched coverage or every disclosed exclusion has
an explicit scientific justification.

