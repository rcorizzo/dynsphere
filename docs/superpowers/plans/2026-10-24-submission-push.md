# VLDB submission preparation implementation plan

> **For agentic workers:** Use superpowers:executing-plans, native execution. User approved all proposed work on October 24, 2026.

**Goal:** Prepare an evidence-bounded method-paper abstract and manuscript, finish the state/motion experiment, and execute or prepare matched deep comparisons today.

**Architecture:** New research/submission_campaign modules consume the frozen held-out protocol manifest and local score artifacts. Existing detector and baseline files remain unchanged. A separate worker primes TranAD imports before requested seeding and explicitly seeds CATCH after imports. New manuscript sections identify completed and pending comparisons separately.

**Tech Stack:** Existing Python environments, NumPy/pandas/scikit-learn/PyTorch, existing pinned external sources, built-in LaTeX compiler.

**Spec:** Approved October 24 strategy: method-focused abstract, matched state-only/motion-only/combined study, matched deep comparisons, same-hardware cost analysis, artifact preparation. Prior language guardrails and project STYLE.md apply. No new architecture or dataset search.

## Global constraints

- Native execution in current checkout; no Git mutation, external publication, submission, SSH, or HPC submission. Prepare manual HPC runners when laptop timing is unsuitable.
- Preserve all existing scores, detector/adapters, and frozen manifests. New output root results/submission_push_20261024.
- Existing manifest 502a1002c7788742def0a4bc13aeac1885b7376aec38ad5f3ddab3b07b2a0423 defines sources, train/calibration/test splits, seeds, and local methods.
- State/motion study: E1 only, three seeds, all 61 eligible held-out cases including separately labeled PowerSystem. State = position + regime distance. Motion = latent residual + raw residual + state-speed + speed. Combined = both (transition is zero). Evaluate radii 0 and 8 for every score, using identical frozen row/calibration masks. Reconstruct the frozen full E1 score exactly before accepting results.
- This grouping is a diagnostic: regime distance depends on state/velocity assignment and pseudo-energy mixes state and speed. No claim of pure probabilistic independence.
- Deep methods: all six existing external methods prepared; MTAD-GAT and CATCH prioritized. Fixed existing budgets, audited-v2 TranAD, released-budget-v2 Anomaly Transformer. No performance-based tuning or dataset selection.
- Deep primary scoring: untouched test suffix, same fitting 80% as local methods; no use of calibration/test labels by workers. Evaluate all deep and cached local methods on a conservative common 100-row trim at both ends. Calibration operating points remain the separate completed local study; deep threshold calibration is outside this first comparison.
- PowerSystem deep scoring remains a separate unsupported-temporal-boundary diagnostic, excluded from the main matched deep campaign. Contaminated SMD source is recorded as incompatible for every method.
- Deep worker outputs are label-free, include source/adapter hashes and runtime identity, and are strictly validated. Fresh-process seed replay checks precede production runs. Local probes have bounded timeouts; prepare AU LSF and PLGrid Slurm runners for expensive work.
- Abstract contains completed evidence only. Retain historical tables with explicit protocol labels. Affiliation and artifact publication choices remain user-owned.

## Review focus

1. Source or implementation changes must invalidate new run reuse without mutating old records.
2. Grouped component scores must reconstruct the frozen combined score and use matched propagation.
3. Deep workers must see label-free inputs only and shared observed support must exclude padded tails.
4. Seed corrections must survive upstream import side effects and replay in fresh processes.
5. Paper claims must distinguish completed local comparisons, historical comparisons, and pending matched deep results; no stale macro headline or causal monitoring claim.

## Task 1: Mechanism and cost studies

- [x] Write tests for state/motion reconstruction, propagation parity, invalid multi-regime reuse, and frozen score disagreement.
- [x] Observe failures, implement studies.py, run tests to green, then execute all eligible cases and seeds.
- [x] Reuse existing matched local fit/score timings; produce source-level end-to-end cost summaries and preserve earlier controlled scaling measurements.

## Task 2: Matched deep campaign

- [x] Write tests for conservative shared support, worker output validation, and seed replay behavior; verify transferable source resolution through isolated archive extraction.
- [x] Observe failures, implement isolated deep worker/controller with frozen identities, bounded subprocesses, and explicit failures.
- [x] Run fresh-process replay probes and a predeclared first-case-per-source local feasibility pilot. Run suitable cases locally; prepare complete manual AU/PLGrid runners and transfer instructions for expensive cases. Full matched deep execution remains pending manual submission.

## Task 3: Paper and artifact

- [x] Preserve source snapshot, revise title/abstract/introduction/conclusion and add completed held-out/component/extension evidence with protocol labels.
- [x] Save a matching plain-text abstract for CMT and an artifact manifest/run guide; no online publishing or submission.
- [ ] Current source compiles successfully. Visual layout/page-count verification requires a fresh PDF export; the saved PDF predates this revision.
- [x] Run full tests and one fresh independent review, address important findings with regression tests, record exact completed/pending outputs.
