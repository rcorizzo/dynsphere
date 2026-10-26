# Matched CPU quality and cost study

This isolated study retains the detector definitions, input sources, fitting
split and deep-method budgets of the held-out comparison. New outputs never
replace the existing primary, extension or native Exathlon results.

The cohort covers all five primary families: BATADAL, HAI-20.07, MSL, SMAP
and SMD. One case per family is selected by lower-median test length among
eligible cases, with case-id tie breaking. The selection uses the
frozen manifest and sequence lengths, independently of anomaly scores. Each
of nine methods uses seeds 37–39 and complete input sequences (135 runs). Shared input
preparation is outside model timers. Partial AUROC, AP and AUROC use the same
100-observation exclusion at both test boundaries.

Each measurement runs sequentially in a fresh process on one CPU thread,
using the existing pinned method environment. Initialization and fitting are
timed together; full test scoring is measured separately. Imports, startup,
shared input preparation, serialization and quality evaluation are excluded
from those times. Peak resident memory is the fresh-process maximum through
scoring, including the runtime, imports, inputs and retained training state.
It is recorded before serialization validation.

Serialized sizes use pickle protocol 4 for the fitted scoring estimator.
Retained diagnostics and checkpoints are included. CATCH's training-only
optimizers and data loaders are omitted. The restored estimator must
reproduce its score vector. Sizes exclude installed packages and source code.
This measures the retained scoring state of these implementations.

From the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 DYNSPHERE_DEVICE=cpu \
  .venv/bin/python -m research.quality_cost.run \
  --output results/quality_cost_20261026/new-campaign --dry-run
```

Inspect the frozen manifest, then replace `--dry-run` with `--resume` to run
the frozen campaign. An existing campaign cannot be overwritten. Resume
checks its local and upstream source hashes, configurations, worker identities
and input hashes. It recovers completed worker artifacts after interruption,
and preserves recorded failures. Each worker has a frozen one-hour wall-time
limit; a timeout is reported explicitly, without reducing the training budget.
A changed
implementation requires a new output directory and a new campaign identity.

`per_run.csv` retains all outcomes. `per_case.csv` averages quality over seeds
and reports median cost per case and method. Failed measurements have no
imputed numerical scores. A complete study requires all planned runs.
