# Independent Linux explanations and embedding PCA

This runner evaluates the same controlled-edit explanation protocol as
[explanation_study](../explanation_study/README.md), using independent
Linux CPU fits. It adds separately timed representation diagnostics.
Mac and Linux fits and their timing measurements are not pooled.

## Prerequisites

Prepare all method environments and restore the bound study inputs,
configuration records, source snapshots, and archived DAGMM checkpoints.
The preparation command requires a relocated continuation manifest with
its parent-manifest identity, path mapping, runtime inventories, and file
hashes. This generated manifest is not included in the code repository.

## Prepare a new replication identity

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python \
  -m research.explanation_replication.run \
  --manifest results/linux-explanations/manifest.json \
  --prepare /path/to/continuation.json \
  --output results/linux-explanations
```

Preparation validates inherited scientific fields and bound files; it does
not fit a model. An incompatible or changed source/input snapshot requires a
newly prepared manifest.

## Execute and validate

```bash
.venv/bin/python -m research.explanation_replication.run \
  --manifest results/linux-explanations/manifest.json \
  --output results/linux-explanations --stage pilot --index 1
```

Pilot indices 1--65 cover seed 37. Dispatch each index once using separate
CPU allocations and a concurrency limit suited to the host. After every
pilot outcome is terminal:

```bash
.venv/bin/python -m research.explanation_replication.run \
  --manifest results/linux-explanations/manifest.json \
  --output results/linux-explanations --gate
```

Only a passed technical gate permits remaining indices 1--130
(`--stage remaining`), covering seeds 38--39. No ranking criterion
enters promotion. All failures and timeouts remain terminal.

```bash
.venv/bin/python -m research.explanation_replication.aggregate \
  --output results/linux-explanations
```

The aggregate reads the manifest from the selected output directory.
Use the exact method runtimes for representation recomputation. Representation
PCA is fitted only on normal training embeddings. Native widths, pooling,
missing metrics, and support restrictions accompany its interpretation.
The representation stage has its own timeout and cannot change explanation
completion or the original fitting/scoring measurements.
