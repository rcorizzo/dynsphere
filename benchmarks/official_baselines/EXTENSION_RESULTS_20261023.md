# AU extension results and validation: October 23, 2026

## 1. Coverage and integrity

All 48 training shards and 16 metric shards were downloaded and inspected.

| Job | Group | Planned | Complete | Input rejected |
|---|---|---:|---:|---:|
| 44537 | GPU baselines | 792 | 780 | 12 |
| 44538 | External CPU baselines | 396 | 390 | 6 |
| 44539 | Euclidean neural predictor | 198 | 195 | 3 |
| 44540 | Cached-score VUS | 1,008 | 1,008 | 0 |

The 21 rejected configurations share one input file:
`034_GHL_id_3_Sensor_tr_50000_1st_122001.csv`. Its HPC copy has invalid
binary labels and a different SHA-256 hash from the frozen local input.
The other 47 selected TSB files and all 18 GutenTAG train/test pairs have
matching source hashes. The exception is a data-copy discrepancy, and these
records cannot support a conclusion about model numerical reliability.

For every completed HPC run, the audit verified record identity, score
shape and finiteness, private-label correspondence, stored train/test
digests, and agreement with the frozen local inputs. Scaled arrays agree
within absolute/relative tolerance 1e-12. The largest absolute difference
is 2.84e-14, consistent with the differing numerical environments.
Pointwise AP and AUROC were independently recomputed and checked against
the recorded values. The existing 1,386 local records were also checked.

There are **2,751 available score records** of 2,772 planned. The provisional
14-method comparison uses **2,730 records on 65 complete cases**: 25 Exathlon,
22 GHL, and 18 GutenTAG cases, with three requested detector seeds. The
local scores for the remaining GHL case are preserved outside this matched
table. Every method in the table uses identical case and row support.

## 2. Matched pointwise results

Values are mean AUPRC, averaging detector seeds within case and then cases
within source. GHL has 22 cases in this provisional table. Exathlon and GHL
are TSB-AD preparations; GHL and GutenTAG are synthetic collections.

| Method | Exathlon | GHL | GutenTAG |
|---|---:|---:|---:|
| DynSphere-S, four regimes | 0.9025 | 0.0136 | 0.2659 |
| DynSphere-E, four regimes | 0.9131 | 0.0112 | 0.3742 |
| DynSphere-S1 | 0.9135 | 0.0137 | 0.1399 |
| DynSphere-E1 | 0.9373 | 0.0112 | 0.3230 |
| Euclidean neural predictor | 0.9377 | 0.0279 | 0.3663 |
| Linear ridge AR(1) | 0.9152 | 0.0257 | 0.3153 |
| PCA reconstruction | 0.9109 | 0.0138 | 0.2352 |
| KMeans distance | 0.9386 | 0.0141 | 0.2551 |
| DAGMM (TAB) | 0.3081 | 0.0306 | 0.2155 |
| OmniAnomaly (TSB-AD) | 0.9653 | 0.0319 | 0.1123 |
| CATCH, overlap correction | 0.7845 | 0.0084 | 0.4515 |
| MTAD-GAT | 0.9430 | 0.0179 | 0.9073 |
| Anomaly Transformer, released-budget profile | 0.1383 | 0.0142 | 0.0990 |
| TranAD, audited scaling/alignment | 0.1283 | 0.0142 | 0.0500 |

The Euclidean variants remain competitive on Exathlon. E1 is close to
KMeans and the neural predictor, while TSB-AD OmniAnomaly has higher AUPRC
and AUROC (0.9653 and 0.9967). On GutenTAG, MTAD-GAT substantially exceeds
the proposed variants. CATCH also has higher average AUPRC than DynSphere-E.
GHL remains weak for DynSphere; several controls have higher AUPRC.
The expanded comparison does not support broad superiority claims.

The common 100-row-exclusion sensitivity analysis preserves these main
comparisons. Exathlon E1 remains at 0.9373 AUPRC. GutenTAG DynSphere-E
changes to 0.3757 and MTAD-GAT remains at 0.9073. GHL TSB-AD OmniAnomaly
changes to 0.0332, compared with 0.0112 for E1. The primary table retains
the extension's original shared one-row exclusion; the sensitivity results
are saved separately.

The complete six-mechanism GutenTAG breakdown is retained. For example,
DynSphere-E reaches 0.9815 AUPRC on variance anomalies, compared with
0.9904 for CATCH and 0.8959 for MTAD-GAT. On mode-correlation anomalies,
its AUPRC is 0.7320, compared with 0.0625 for CATCH and 0.8826 for MTAD-GAT.
These restricted strengths should be interpreted alongside the other
mechanisms and the full average. Generation uses one fixed base design
and severity per mechanism, with three generation seeds.

## 3. VUS coverage

The 1,008 real-data-preparation metric records are complete, with 330
distinct score/label inputs. All metric keys, the persistent input seal,
cache outputs, and row coverage were verified. One Exathlon case and one
GHL case were independently recomputed with the official VUS backend;
differences from the HPC results were at most 8.2e-17.

Together with the 378 local GutenTAG records, VUS covers the original seven
methods on all 66 cases. This table therefore uses all 23 GHL cases.

| Method | Exathlon VUS-PR | GHL VUS-PR | GutenTAG VUS-PR |
|---|---:|---:|---:|
| DynSphere-S | 0.8971 | 0.0191 | 0.3163 |
| DynSphere-E | 0.9031 | 0.0151 | 0.4334 |
| DynSphere-S1 | 0.9005 | 0.0198 | 0.1794 |
| DynSphere-E1 | 0.9234 | 0.0151 | 0.3705 |
| Linear ridge AR(1) | 0.9095 | 0.0365 | 0.3364 |
| PCA reconstruction | 0.9128 | 0.0165 | 0.2500 |
| KMeans distance | 0.9290 | 0.0178 | 0.2676 |

VUS for the seven newly trained methods is now complete. Four single-threaded
CPU workers evaluated all 1,365 completed records, representing 1,023 distinct
metric inputs, without evaluation errors or detector retraining. Final
aggregation checked exact coverage, finite official VUS values, duplicate
identities, and agreement with the frozen pointwise metrics.

Checkpoints and final outputs are isolated under
`results/extension_audit_20261023/new_methods_vus/`. Each worker writes a
`completion-shard-XXX.json` and `metrics-shard-XXX.csv`. A separate
finalization process waits for all four workers to succeed, checks exact
run coverage and pointwise-metric agreement, and then writes the combined
14-method tables on the same 65-case cohort. The final success marker
`evaluation_complete.json` exists. Logs are
`logs/new-methods-vus-*-20261023.log`.

The combined outputs retain 2,751 available records. The matched comparison
uses 2,730 records: 14 methods, 65 cases, and three requested seeds. Its
GHL average uses the same 22 cases for every method. Consequently, its GHL
values differ slightly from the seven-method table above, which retains
all 23 GHL cases.

| Method | Exathlon VUS-PR | GHL VUS-PR | GutenTAG VUS-PR |
|---|---:|---:|---:|
| DynSphere-S | 0.8971 | 0.0194 | 0.3163 |
| DynSphere-E | 0.9031 | 0.0153 | 0.4334 |
| DynSphere-E1 | 0.9234 | 0.0152 | 0.3705 |
| KMeans distance | 0.9290 | 0.0180 | 0.2676 |
| Linear ridge AR(1) | 0.9095 | 0.0372 | 0.3364 |
| Euclidean neural predictor | 0.9281 | 0.0315 | 0.3912 |
| CATCH | 0.8294 | 0.0130 | 0.4601 |
| MTAD-GAT (ML4ITS) | 0.9323 | 0.0207 | 0.9569 |
| OmniAnomaly (TSB-AD) | 0.9733 | 0.0512 | 0.1300 |

DynSphere-E1 remains competitive on Exathlon, with VUS-PR close to KMeans,
the neural predictor, and MTAD-GAT. The case-paired comparison nevertheless
places MTAD-GAT above E1 on all 25 Exathlon cases, with a mean difference
of 0.0089. TSB-AD OmniAnomaly has the highest
Exathlon VUS-PR. GutenTAG retains a substantial gap between MTAD-GAT and
every DynSphere variant. DynSphere-E improves on spherical DynSphere and
the one-regime Euclidean variant on that controlled collection. GHL remains
weak for the proposed variants. The range-aware evaluation therefore
supports restricted strengths and does not establish broad superiority.

The full six-mechanism breakdown is preserved. DynSphere-E obtains 0.9983
VUS-PR on variance anomalies and 0.8297 on mode-correlation anomalies.
MTAD-GAT obtains 0.9195 and 0.9127, respectively. Its other four mechanism
averages range from 0.9415 to 0.9961. These findings motivate a controlled
history-representation experiment; they do not establish which MTAD-GAT
component explains its advantage.

The complete matched table also includes DAGMM, Anomaly Transformer,
TranAD, PCA, and DynSphere-S1. Their records are retained with the previously
identified reproducibility limitations. Metric completion alone does not
make the manuscript submission-ready.

## 4. Reproducibility concerns found during inspection

**TranAD seed control.** Its upstream `src.models` module executes
`torch.manual_seed(1)` during import. The adapter sets the requested seed
before importing that module. An isolated diagnostic confirmed a requested
seed of 37 becomes 1 after import. All 65 completed cases have identical
score vectors for requested seeds 37, 38, and 39. The records retain valid
scores, but they do not establish independent-seed robustness. Seed control
requires correction and a separately identified rerun before that claim.

**CATCH replayability.** The shared adapter seeds Python and NumPy. CATCH's
adapter discards its explicit seed argument and does not explicitly seed
PyTorch. Its three score vectors differ in each completed case, but replay
from the recorded seed is unverified. PyTorch initialization control needs
an explicit audit before reproducible multi-seed claims.

**MTAD-GAT implementation scope.** The evaluated model uses the ML4ITS
PyTorch implementation, which describes itself as inspired by MTAD-GAT
and explicitly does not claim to reproduce the original paper's results.
Its defaults include GATv2 attention and a GRU reconstruction decoder.
The campaign adapter trains forecasting and reconstruction objectives
together and adds their squared errors at evaluation. This implementation
must be identified explicitly in comparisons; it is a third-party
implementation, and its accuracy does not isolate attention's contribution.

**Weak transformer scores.** TranAD has constant scores for eight Exathlon
cases (24 requested-seed records). Anomaly Transformer remains close to
chance on several sources despite its expanded capacity and budget.
These observations require further scoring/training diagnostics. Scores
were retained with their original direction, configuration, and cohort.

The manuscript and detector source were unchanged during this audit.
Readiness remains false because of the missing input, seed-control issues,
incomplete VUS comparison, and pending source-native validation.

## 5. Data-copy correction, performed manually

To restore the exact frozen input, run this from a MacBook terminal:

```bash
rsync -av --backup --suffix=.before-extension-repair-20261023 \
  /Users/robertocorizzo/git/pendulum/code/benchmarks/tsb_ad_m/data/TSB-AD-M/TSB-AD-M/034_GHL_id_3_Sensor_tr_50000_1st_122001.csv \
  rcorizzo@prodhpcloginnode01.american.edu:/home/rcorizzo/pendulum/benchmarks/tsb_ad_m/data/TSB-AD-M/TSB-AD-M/
```

This preserves the existing HPC file with a backup suffix. On the HPC,
verify its hash:

```bash
cd /home/rcorizzo/pendulum
sha256sum benchmarks/tsb_ad_m/data/TSB-AD-M/TSB-AD-M/034_GHL_id_3_Sensor_tr_50000_1st_122001.csv
```

Expected SHA-256:
`03359ca9d437a9d2acad8e7f6dfff9977956b03349ea9c8899560a3d5a5f2f17`.

This correction alone leaves 21 missing fits. With identical code,
environment, Git revision, and other inputs, the existing runners can reuse
the 1,365 completed fits. The seed-control corrections are a separate issue
and must receive new implementation identities. Avoid changing the broad
source fingerprint and then blindly resubmitting the whole campaign,
because that can invalidate otherwise reusable run identities.

## 6. Audit outputs

The derived outputs are isolated under `results/extension_audit_20261023/`:

- `verification.json`: coverage, provenance, tolerances, and remaining issues.
- `per_run_all_available.csv`: all 2,751 available score records.
- `per_run_matched.csv` and `per_family_matched.csv`: the provisional 65-case comparison.
- `per_run_matched_warmup100.csv` and `per_family_matched_warmup100.csv`: common warmup sensitivity.
- `vus_seven_methods_per_run.csv` and `vus_seven_methods_per_family.csv`: complete seven-method VUS.
- `imported_run_integrity.csv`: per-run input/label/metric checks.
- `noncomplete_records.json` and `raw_source_mismatches.json`: the exact missing configurations and source discrepancy.
- `imported_score_diagnostics.csv` and `seed_vector_diagnostics.json`: score behavior and requested-seed checks.
- `gutentag_mechanism_all_methods.csv`: all mechanisms and methods.
- `paired_against_e1.csv`: case-paired differences after seed averaging; no independence-based significance claim is made.
