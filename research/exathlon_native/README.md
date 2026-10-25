# Frozen native Exathlon study

This study is separate from the paper's existing five-family comparison. The approved protocol is in `docs/superpowers/specs/2026-10-25-exathlon-native-frozen-design.md`.

## Scope

Four applications, five disturbed executions, three seeds, ten score configurations. There are 84 local and 24 deep fits, producing 120 local and 30 deep score records. The reduced E1 score reuses the original E1 fit. Application 5 reuses each model for both test executions.

The first two applications have prior native pilot outcomes. The two additional applications have no native pilot outcomes. Other preparations of Exathlon have already informed development. Every selected execution remains in the inventory, including non-evaluable or unsuccessful configurations.

Primary common support excludes the first and last 100 rows of each uninterrupted segment. Primary pAUROC through FPR 0.05 averages seeds, then executions within application, then applications equally. Secondary temporal metrics use official VUS 0.0.6, buffer 100, separately for each retained segment containing both classes. Segment results are length-weighted within execution; their support is reported explicitly.

## Local execution

From the code repository:

```bash
.venv/bin/python -m research.exathlon_native.run prepare --tree /tmp/dynsphere-native-exathlon-tree-4101.json
benchmarks/official_baselines/.envs/catch/bin/python -m research.exathlon_native.run compatibility --method catch
benchmarks/official_baselines/.envs/mtad_gat/bin/python -m research.exathlon_native.run compatibility --method mtad_gat
.venv/bin/python -m research.exathlon_native.run freeze
.venv/bin/python -m research.exathlon_native.run dry-run
.venv/bin/python -m research.exathlon_native.run local
.venv/bin/python -m research.exathlon_native.evaluate
.venv/bin/python -m research.exathlon_native.evaluate --vus
```

After freezing, source/input changes are rejected. A new implementation requires a new output root and a newly identified campaign. Compatibility checks use synthetic data and two epochs; they do not contribute benchmark scores. The complete upstream training budget remains ten epochs.

## AU HPC

Use a new directory, e.g., `/home/rcorizzo/pendulum-native-20261025`. Transfer the frozen source bundle, prepared inputs, manifest, and compatibility evidence. Raw source archives are unnecessary on HPC once prepared arrays are verified. Reuse the Linux environments and pinned upstream clones from the previous L40 checkout through links in the new directory. Never copy the Mac environments to Linux.

After extracting the supplied bundle on AU HPC, from the new directory:

```bash
module load git/2.40.0
mkdir -p benchmarks/official_baselines logs
ln -s /home/rcorizzo/pendulum-l40-private-pilot-20261025/.venv .venv
ln -s /home/rcorizzo/pendulum-l40-private-pilot-20261025/benchmarks/official_baselines/.envs benchmarks/official_baselines/.envs
ln -s /home/rcorizzo/pendulum-l40-private-pilot-20261025/benchmarks/official_baselines/.external benchmarks/official_baselines/.external
export DYNSPHERE_NATIVE_REPO=/home/rcorizzo/pendulum-native-20261025
.venv/bin/python -m research.exathlon_native.run dry-run
```

Verify upstream hashes before submission:

```bash
.venv/bin/python -c 'from research.exathlon_native.protocol import validate; validate(upstream=True); print("Frozen input, code, and upstream checks passed")'
```

First benchmark one full deep fit on L40, with the same frozen settings, and capture its actual runtime:

```bash
bsub -J 'dynsphere-native-timing[1]' < research/exathlon_native/au_l40.bsub
```

This runs shard 1 (application 1, CATCH, seed 37). Review its fit record, hardware evidence, and runtime before submitting the remaining 23 fits. No estimate is inferred from the synthetic check. Once it passes:

```bash
bsub -J 'dynsphere-native-l40[2-24]%2' < research/exathlon_native/au_l40.bsub
```

The controller enforces a 24-hour timeout per fit. Two array tasks may run concurrently. Completed fits are verified and reused. Failed attempts remain on disk; resubmitting that shard creates a separate attempt. Existing jobs and previous result directories are unaffected.

## Download and summarize

Transfer only the new study directory back to the matching local result root. Preserve the frozen inputs and manifest. Then run:

```bash
.venv/bin/python -m research.exathlon_native.evaluate
.venv/bin/python -m research.exathlon_native.evaluate --vus
```

Temporal calculations are cached and can be divided with `--vus --metrics-only --shard-count N --shard-index I`; run the summary command after all shards finish. `summary/readiness.json` reports completed, pending, unsuccessful, non-evaluable, and temporal-metric coverage. Manuscript readiness remains false pending scientific review.

Data originate from the pinned Exathlon release and retain its license restrictions. Raw archives and generated source inputs are excluded from the code commit and retained in the local results directory.
