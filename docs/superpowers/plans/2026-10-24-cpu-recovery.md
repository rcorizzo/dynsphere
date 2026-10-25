# Isolated CPU recovery implementation plan

> Execute natively using superpowers:executing-plans. The user explicitly
> approved corrections and reruns while preserving all results and leaving
> active jobs unchanged. No further design approval is required.

**Goal:** Recover DAGMM execution and correct OmniAnomaly validation on the
frozen held-out sources without changing jobs 44785/44786 or their inputs.

**Architecture:** Add an isolated recovery controller/worker and an explicitly
selected cross-campaign aggregator. Reuse unchanged frozen helpers. A private
DAGMM pandas/pytz directory is added only to new worker processes. OmniAnomaly
receives fitting plus reserved normal-calibration observations; its existing
80/20 split recovers the exact fitting boundary. Model architecture, loss,
window, epoch budget, and upstream files remain unchanged.

**Tech Stack:** Existing Python environments, pinned TAB/TSB-AD sources,
NumPy, pandas, scikit-learn, TensorFlow, PyTorch, AU LSF.

**Spec:** User approval on October 24: perform the two corrections and reruns,
preserve existing results, and leave active jobs unchanged. The frozen base
manifest remains 502a1002c7788742def0a4bc13aeac1885b7376aec38ad5f3ddab3b07b2a0423.

## Global constraints

- No editing, cancelling, suspending, or resubmitting jobs 44785/44786.
- No changes to shared .envs, .external, existing detector/adapters, or
  research/submission_campaign code. Native implementation uses new files.
- Use a new HPC directory /home/rcorizzo/pendulum-recovery-20261024 and new
  result identities. Preserve original CPU records, including successes.
- Install only pandas==1.5.3 and pytz==2024.1 as binary wheels in a private
  target directory. Keep NumPy, TensorFlow, and sklearn versions unchanged.
- Both methods rerun all 61 declared non-PowerSystem cases and three seeds:
  366 planned records, 360 eligible executions, six incompatibilities.
- Workers receive features only. Independently validate frozen input hashes,
  configurations, source evidence, runtime evidence, and score hashes.
- Keep upstream OmniAnomaly normalization behavior unchanged; its dataset
  utility normalizes fitting, validation, and test arrays separately. Record
  this transductive feature-normalization caveat explicitly.
- Preserve the current paper. No Git commit/push is authorized for this
  experiment-code task. Remote transfer and new recovery jobs are authorized.

## Review focus

1. Shared environments and live campaign source must remain byte-identical.
2. Short cases must use the reserved normal validation observations with the
   exact fitting boundary, sufficient windows, and no test observations.
3. New identities must bind actual recovery code, runtime overlay contents,
   original upstream content, configurations, and frozen input arrays.
4. Aggregation must select repaired DAGMM/OmniAnomaly and original GPU
   methods explicitly and refuse wrong-source, wrong-config, or mixed-policy
   records with identical case names.
5. Retry/resume and pilot gating must preserve all existing records and
   avoid concurrent writes to the same recovery run.

## Tasks

1. Write failing tests for reserved boundary checks, overlay environment
   isolation, provenance validation, and explicit cross-campaign selection.
   Implement research/submission_recovery modules; run focused tests.
2. Add setup, pilot, and 16-shard CPU scripts. Run the full local test suite
   and one independent review. Fix important findings in one test-backed pass.
3. Transfer the unchanged original runner snapshot plus new recovery files
   into the new HPC directory. Install the private dependency overlay and
   verify imports, pinned versions, and unaffected active source hashes.
4. Submit a bounded first-MSL-case pilot for both methods. On verified
   success, submit the full recovery CPU array and inspect early records.
5. Record exact job IDs, identities, verification evidence, pending work, and
   unchanged existing-job status. No automatic monitoring is requested.
