# Four-method matched CPU timing extension

This adds DAGMM (TAB), OmniAnomaly (TSB-AD), Anomaly Transformer, and TranAD
to the existing nine-method CPU study. The original135 outcomes, input files,
model implementations and results are read-only. The extension has60 trials:
the same five cases, seeds37–39, one CPU thread and full existing budgets.
Each fresh worker has a3600-second whole-process limit.

The controller reuses the input bundles selected before the original study.
It records local and upstream source hashes, all installed package versions,
the exact configurations, input hashes, stored scores and same-run quality.
Failures, timeouts and interrupted attempts are retained. Resume recovers a
validated completed worker when its parent record is missing; an unfinished
attempt becomes an explicit interrupted outcome with no automatic retry.
Shared hardware locks prevent concurrent extension controllers and workers,
including across different output directories. Unrelated computations must
also remain stopped during measured runs.

Initialization/fitting and full scoring have separate timers. Imports, input
loading, serialization and metric evaluation are excluded. Transformer
measurements execute the unchanged adapter statements in their existing
order. Their test-window materialization therefore occurs inside the fitting
interval. Scoring-state reconstruction is verified against each stored score
vector, replaying random states for stochastic inference.

Anomaly Transformer and TranAD serialize the network and required inference
metadata using pickle4. OmniAnomaly uses its fitted estimator with training
optimizer/scheduler removed. DAGMM requires its native TensorFlow checkpoint,
including graph and optimizer variables. These storage formats are identified
explicitly; byte-size comparisons require this qualification. Peak process
RSS is captured through first scoring, before serialization checks.

Compatibility inputs are synthetic and excluded from reported measurements.
Every pilot retains the production model configuration and compares its
scores with the unchanged adapter in addition to serialization restoration.
Anomaly Transformer uses110 fitting rows for its compatibility check; the
other pilots use400. All production trials use complete original sequences.

From the code repository:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m research.quality_cost_extension.run \
  --output results/quality_cost_20261026/four-methods-cpu-v1 --dry-run
```

After inspecting the frozen manifest and separate compatibility checks, use
the same command with `--resume` replacing `--dry-run`. A new campaign never
overwrites an existing output directory. `--aggregate` performs stored-score
checks without training. All retained outcomes appear in `per_run.csv` and
`per_case.csv`; no missing values are imputed. Summaries belong to this
five-case descriptive cohort, separately from the59-case accuracy study.
