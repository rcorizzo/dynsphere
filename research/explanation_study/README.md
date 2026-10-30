# Frozen explanation comparison

This isolated exploratory study uses the five metadata-selected CPU timing
cases, all thirteen original configurations and seeds37-39. Seed37 is the
pilot and is counted once in the full195-model matrix. No original result,
configuration, source or environment is modified.

The comparison uses the same grouped sensor/time occlusion for every model.
It measures held-out joint-intervention fidelity, known edit localization,
small-noise stability, reference sensitivity and explanation cost. Native
DynSphere score accounting is checked separately. PCA explained variance is
not used as a comparative explanation metric.

Each detector retains its full original training budget and arrays. Scoring
uses explicit512-row contexts with a max-score target at rows240:272. This
is a separate context-window protocol. Replacements use real training values
in rows128:384. They are statistical interventions with no claim of physical
realizability. Known edited inputs do not establish physical fault causes.

The one-seed pilot must record65 terminal outcomes. Source/input/runtime
bindings, repeatable inference, original-score agreement, native accounting
and minimum comparator coverage are technical promotion requirements.
Performance rankings never enter the gate. Resource timeouts stay explicit;
they do not demonstrate inferior explanation quality. A failed gate stops
the controller. All195 outcomes and all predefined metrics are retained.

Run from the existing code checkout, using its pinned environments:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m research.explanation_study.run \
  --output results/explanation_20261027/v3 --dry-run

# After verifying the frozen inputs and source, run exactly one controller:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 .venv/bin/python -m research.explanation_study.run \
  --output results/explanation_20261027/v3

# Read-only analysis of existing terminal outcomes:
.venv/bin/python -m research.explanation_study.run \
  --output results/explanation_20261027/v3 --aggregate
```

The two timed worker stages each have a3600-second budget; the process-group
hard cap is7230seconds. A shared hardware lock prevents concurrent workers.
Terminal JSON publication is atomic and exclusive. On recovery, a live PID
or held worker lock blocks execution; completed child records are reconciled,
and unfinished attempts are retained as interrupted without automatic retries.
Do not rerun an interrupted or failed measurement under a changed source.

Every original/full score vector, fitted state, inference RNG state, event
input, attribution, held-out perturbation response and metric is bound to
the manifest or terminal record by hashes. Aggregation recomputes numerical
metrics from saved evidence. Undefined constant-response metrics remain
unavailable. All-model and pairwise aggregates identify exact common support
and require all three seeds; an incomplete support is explicitly labeled.
Detected-edit and stable-score subsets are descriptive per-method summaries,
because their membership can differ between methods.

The shuffled-attribution stability value shares its permutation between
original/noisy inputs, so rank correlation is preserved. It is not an
independent stability control. Stability is interpreted jointly with
faithfulness and localization.
