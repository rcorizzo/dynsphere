# Full local-method extension: October 22, 2026

## 1. Completion and provenance

The complete selected local-method campaign finished on the MacBook. All
1,188 configurations completed with valid pointwise metrics. The workload
contains 25 Exathlon traces, 23 GHL traces, and 18 official GutenTAG generated
cases, six methods, and detector seeds 37, 38, and 39. The 18 GutenTAG cases
cover six mechanisms at generation seeds 101, 102, and 103.

The runner reused 132 validated pilot records and fitted 1,056 additional
configurations. Two single-threaded CPU workers handled the two shards. No
new HPC job was submitted for this extension. The live HPC queue was not
queried, so this statement does not describe unrelated cluster activity.

Both shard completion records report 594 complete configurations. The combined
analysis verifies all 1,188 expected case/method/seed identities, absence of
duplicates, finite metrics, and preservation of the pilot-only readiness flag.
The source revision and actual source-content fingerprints are retained.

## 2. Evaluation protocol

Exathlon and GHL use the fixed TSB-AD-M manifest selection. Models train on
the clean filename-declared prefix and score only the held-out suffix.
Each file remains a separate sequence. GutenTAG uses separate normal training
and anomalous test files from pinned official generator version 1.5.0.
Scaling is fitted on training observations. All methods exclude the first
scored row from metrics, accounting for the AR(1) control's missing predecessor.

DynSphere retains its existing weights, latent dimension, and one/four-concept
settings. Centered derivatives and propagation make the evaluation offline.
The controls use PCA reconstruction and one-step linear ridge regression
with penalty 1. The AR(1) residual has no temporal propagation. No settings
were selected using the new test results.

The reported AUPRC is average precision. Metrics are averaged over detector
seeds within case and then cases within source. The GutenTAG source mean
also averages independent generation replicates within the fixed mechanism
design. These generated cases do not represent independent real deployments.
No confidence intervals, deep-baseline comparison, or VUS results are claimed
for this stage.

## 3. Full selected-workload results

| Method | Exathlon AUPRC | GHL AUPRC | GutenTAG AUPRC |
|---|---:|---:|---:|
| DynSphere-S | 0.9025 | 0.0134 | 0.2659 |
| DynSphere-E | 0.9131 | 0.0111 | 0.3742 |
| DynSphere-S1 | 0.9135 | 0.0135 | 0.1399 |
| DynSphere-E1 | 0.9373 | 0.0111 | 0.3230 |
| PCA reconstruction | 0.9109 | 0.0137 | 0.2352 |
| Linear ridge AR(1) | 0.9152 | 0.0254 | 0.3153 |

1. DynSphere-E1 has the highest Exathlon AUPRC among the six tested methods.
   Its differences from AR(1) and PCA are approximately 0.0221 and 0.0264.
   This extends the favorable pilot result to the full 25-trace selection.
2. DynSphere-E has the highest GutenTAG AUPRC. The comparison provides a
   controlled example in which four concepts outperform the Euclidean
   one-concept configuration. Effects remain dependent on anomaly mechanism.
3. All DynSphere variants perform poorly on GHL. AR(1) has the highest GHL
   AUPRC and AUROC (0.5807). This collection remains in the report with its
   original configuration and selection rule.
4. These results are encouraging for the proposed detector on Exathlon and
   the fixed controlled study. Broader superiority requires comparison with
   the audited external baselines and a complete metric analysis.

## 4. Remaining evaluations

1. **External-baseline audit and extension.** Verify model-compatible
   preprocessing, row alignment, convergence, and score extraction. TranAD
   and Anomaly Transformer require particular attention based on the earlier
   review. Then evaluate the selected deep baselines on the same disjoint
   workload. Historical scores require compatibility checks before reuse.
2. **Official VUS metrics.** Reuse cached score vectors. Median labeled-event
   lengths across Exathlon/GHL are approximately 1,000 samples (ranges
   953–1,916 and 773–1,299, respectively). Evaluating buffered metrics on
   these long series can be substantially more expensive than fitting the
   local models. Use an exact score/label cohort with the shared first-row
   exclusion. The older general aggregator evaluates every stored row and
   does not implement this extension's exclusion, so it must not be used
   unchanged for the extension tables.
3. **Source-native validation.** Obtain and audit source-native Exathlon/GHL
   releases, preserving train/test splits and application/trace identities.
   The current results describe TSB-AD versions of these collections.
4. **Method justification and deployment cost.** Run matched component
   controls, including propagation, on the final protocol. Measure scaling,
   scoring throughput, and memory with controlled hardware and thread counts.
5. **Manuscript integration.** Assess coverage, source-level effects, and
   uncertainty before replacing or extending the paper tables. Existing paper
   results remain unchanged.

## 5. Why the earlier work used HPC

The previous multi-seed comparison attempted 2,457 configurations, including
756 GPU-baseline executions for CATCH, MTAD-GAT, Anomaly Transformer, and
TranAD. It obtained 2,453 valid score vectors with four documented DAGMM
exceptions. Those results were transferred from AU HPC to the local project.

The earlier recorded median DynSphere run cost was approximately 0.10–0.15
seconds. CATCH's 189 H100 runs had summed recorded elapsed time of about
11.2 hours before parallel execution. These are end-to-end measurements
including preparation and startup, with heterogeneous devices. They provide
workload context and do not establish a controlled hardware speedup.

Some inexpensive methods were included in HPC batches for a common
environment and convenient scheduling. The present six-method CPU campaign
does not require GPU hardware. Neural-baseline training and expensive metric
evaluation remain stronger reasons to use HPC.

## 6. Value of additional datasets

The immediate priority is completion of the above validity and comparison
checks. Another independent real deployment could later test the fixed
configuration's generalization. Additional transformed copies of an existing
collection provide limited independent evidence. Source-native validation
primarily tests sensitivity to preparation and splitting.

Further synthetic configurations can test specific hypotheses about noise,
event duration, severity, and geometry. They should have predeclared selection
rules and a clear scientific question. The current six-mechanism, one-severity
GutenTAG design provides a starting point for that analysis. No additional
dataset is needed simply to increase the reported number of collections.

## 7. Artifacts and environment

- [Frozen campaign manifest](/Users/robertocorizzo/git/pendulum/code/results/extension_v1/campaigns/754b33fc172b.json)
- [Combined completion and coverage record](/Users/robertocorizzo/git/pendulum/code/results/extension_v1/summaries/754b33fc172b/combined/completion.json)
- [Combined per-run metrics](/Users/robertocorizzo/git/pendulum/code/results/extension_v1/summaries/754b33fc172b/combined/per_run.csv)
- [Per-source results](/Users/robertocorizzo/git/pendulum/code/results/extension_v1/summaries/754b33fc172b/combined/per_family.csv)
- [GutenTAG mechanism results](/Users/robertocorizzo/git/pendulum/code/results/extension_v1/summaries/754b33fc172b/combined/per_mechanism.csv)

Detector environment: Python 3.9.23, NumPy 2.0.2, pandas 2.3.3,
scikit-learn 1.6.1, SciPy 1.13.1, and PyTorch 2.8.0 on macOS ARM64.
The installed VUS package is 0.0.6; full-campaign VUS computation is pending.
The isolated generator environment is recorded with the generated data.

All six-method pointwise configurations are complete. The manuscript-readiness
flag remains false. Existing result directories and manuscript tables were
preserved, and no new source code or detector behavior was changed during
this full-campaign run.
