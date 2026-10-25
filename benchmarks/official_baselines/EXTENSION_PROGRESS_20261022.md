# Extension evaluation: verified progress, October 22, 2026

## 1. Completed work

1. Seven-method pointwise comparison: **1,386 complete configurations**, using
   25 Exathlon traces, 23 GHL traces, 18 GutenTAG cases, and three detector
   seeds. The original six-method campaign is preserved. All new KMeans
   input digests match the frozen E1 reference for the same case and seed.
2. Official GutenTAG VUS: **378 complete records**, seven methods, no metric
   errors. Identical score/label inputs share exact metric results. The
   first test row is excluded consistently.
3. Matched E1 ablation: **2,178 metric records**, 11 configurations for all
   198 case/seed references. Every reconstructed reference score agrees
   with its frozen cached score. The study retains every selected source.
4. Controlled computational-cost study: **72 fresh-process measurements**,
   with 1,000/10,000/50,000 training rows, 3/16 channels, one/four concepts,
   spherical/Euclidean geometry, and three seeds. Test length is 20,000.
   Numerical-library threads are fixed to one.
5. TranAD audit: a separate `audited-v2` configuration corrects training
   scaling and target-time correspondence. The historical configuration is
   preserved. Anomaly Transformer has a released-capacity/budget profile.
   Both produced finite scores in isolated CPU smoke tests.
6. AU LSF and PLGrid SLURM runners, selection dry-runs, and transfer guidance
   are prepared. No jobs were submitted remotely.
7. Common 100-row warmup sensitivity: **1,386 complete pointwise records**.
   Exathlon E1 AUPRC remains 0.9373; GutenTAG E4 changes from 0.3742 to
   0.3757. The main source-level ordering is preserved. Deep-baseline
   sensitivity remains pending their score vectors.

Verification: the full test suite passed **187 tests**, with 20 existing
upstream/degenerated-input warnings. Independent code review identified
metric-time field overwriting and insufficient protection against later
rank-preserving score mutation. The evaluator now preserves fitting time,
records metric time separately, and verifies a persistent initial input seal.
The original campaign table did not contain score hashes, so historical
integrity before the first seal is limited to available metadata and metrics.

## 2. Results and implications

| Source | DynSphere-S AUPRC | DynSphere-E AUPRC | DynSphere-E1 AUPRC | AR(1) AUPRC | KMeans AUPRC |
|---|---:|---:|---:|---:|---:|
| Exathlon (TSB-AD preparation) | 0.9025 | 0.9131 | 0.9373 | 0.9152 | 0.9386 |
| GHL (TSB-AD preparation) | 0.0134 | 0.0111 | 0.0111 | 0.0254 | 0.0140 |
| GutenTAG | 0.2659 | 0.3742 | 0.3230 | 0.3153 | 0.2551 |

On GutenTAG, mean official VUS-PR is 0.4334 for DynSphere-E, 0.3705 for
DynSphere-E1, 0.3364 for AR(1), 0.3163 for DynSphere-S, 0.2676 for KMeans,
and 0.2500 for PCA. These are seed means within case, followed by case means.
They provide evidence for the Euclidean variant on the fixed controlled
design. The audited deep-baseline comparison is still required.

KMeans and E1 have similar Exathlon AUPRC. GHL favors AR(1) among the
completed methods. Consequently, a general superiority claim remains
unsupported. GHL is retained in the report.

The E1 ablation raises Exathlon AUPRC from 0.9373 to 0.9886 when raw-dynamics
evidence is removed. GutenTAG AUPRC changes from 0.3230 to 0.3415, and GHL
from 0.0111 to 0.0112. Removing latent-dynamics evidence changes GutenTAG
AUPRC to 0.3468. Propagation also has source-dependent effects. These results
provide a reason to examine the relative calibration of the evidence
channels. They do not justify a claim that every component is necessary.
The principal detector configuration remains fixed; any proposed revision
needs confirmation on a separately fixed workload.

For 50,000 training rows and 16 channels, Euclidean E1 has median fitting
time 0.312 seconds and scoring time 0.044 seconds for 20,000 test rows.
Euclidean E4 has medians 1.502 and 0.062 seconds. These measurements exclude
generation, common scaling, imports, and process startup. Peak RSS includes
the entire fresh process and its libraries. Hardware and exact worker code
are recorded with the results. These measurements establish local workload
costs; they provide no cross-hardware neural-baseline speedup estimate.

## 3. Remaining execution

The planned 14-method comparison has 2,772 detector configurations.
The remaining full groups contain 792 GPU-baseline configurations,
396 external-CPU configurations, and 198 neural-predictor configurations.
Two predictor pilot records are already complete and can be reused when
their complete identities match. A third interrupted pilot record is
preserved as incomplete. The pilot was stopped after observing about
23 seconds per 8,000-row case, before proceeding to longer sequences.

Real-data official VUS requires **1,008 seven-method records**, corresponding
to **330 distinct metric inputs** in the verified preflight. The runners
consume cached scores and fit no detectors. Further VUS evaluations will
be needed for the remaining neural-baseline scores after those fits finish.

Source-native Exathlon/GHL validation remains pending. The source audit
records release availability and preparation requirements. Original Exathlon
has 2,283 features and application-level normal/disturbed traces, which need
a separate fixed preparation protocol. No source-native accuracy result
is claimed. The present Exathlon/GHL results are TSB-AD preparations;
GHL and GutenTAG are synthetic collections.

The final deep-baseline table also needs a shared 100-row warmup sensitivity
analysis. The seven local methods' corresponding pointwise analysis uses
cached scores and requires no fitting. Its outputs are kept separately from
the one-row-exclusion principal extension results.

The manuscript remains unchanged and readiness remains false. Integration
follows full coverage checks, convergence inspection, paired comparison,
and review of source-level limitations.

## 4. Main artifacts

- Seven-method comparison: `results/extension_v1/summaries/seven-methods-20261022/combined/`
- GutenTAG VUS: `results/extension_v1/vus/seven-methods-20261022/gutentag/combined/`
- E1 component study: `results/extension_v1/component_study/754b33fc172b/`
- Computational cost: `results/extension_v1/scaling/20261022/`
- Warmup sensitivity: `results/extension_v1/support_sensitivity/20261022/`
- Manual transfer/submission: `benchmarks/official_baselines/EXTENSION_HPC_NEXT_STEPS_20261022.md`
- Native release audit: `benchmarks/official_baselines/SOURCE_NATIVE_AUDIT_20261022.md`
