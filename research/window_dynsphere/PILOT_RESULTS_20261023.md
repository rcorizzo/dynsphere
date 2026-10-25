# Window-conditioned DynSphere-E: bounded pilot results

## 1. Scope and completion

The approved prototype and bounded pilot are implemented. The full test
suite passes: 229 tests, with 20 existing dependency/degenerate-input warnings.
An independent review identified numerical normalization and resume-integrity
issues. Their regression tests failed before the fixes and passed afterward.
The original detector modules, metric implementations, manuscript, and
historical score files remain unchanged. The existing extension source
fingerprint remains `22357e3c923ef3219421f06a4eb49fd40e0d211347f4c2573994e5f083f4f28c`.

Frozen pilot manifest:
`results/window_dynsphere_v1/manifests/bb72fb42ddc0907b4829f4fd9c198f52d2977663284dc379f6670163a4dfe229.json`.

All 120 planned records are present. Of these, 112 completed and eight are
incompatible. The incompatible records are all PowerSystem configurations:
the historical loader uses the same raw source for training and testing.
The pilot retains that predetermined membership and requires a disjoint
protocol before evaluating it. No replacement case was selected.

The completed matched pilot includes eight configurations on 14 cases:
three Exathlon, three GHL, six GutenTAG mechanisms at generation seed 101,
BATADAL, and HAI-20.07. Detector seed is 37. Every reported comparison uses
the common 100-row exclusion. The pilot is exploratory, and its smaller
cohort differs from the complete 65-case extension comparison.

## 2. Matched pilot VUS-PR

Values are case means within each source. All eight configurations complete
every included case. Exathlon and GHL are the frozen TSB-AD preparations.

| Configuration | BATADAL | Exathlon | GHL | GutenTAG | HAI-20.07 |
|---|---:|---:|---:|---:|---:|
| Window K1, reconstruction training (primary) | 0.5846 | 0.8460 | 0.0205 | 0.2200 | 0.3324 |
| Window K1, forecasting-only training | 0.5816 | 0.8461 | 0.0210 | 0.2005 | 0.3317 |
| Window K4, reconstruction training | 0.3935 | 0.8423 | 0.0172 | 0.2092 | 0.3298 |
| Window K4, forecasting-only training | 0.3947 | 0.8429 | 0.0158 | 0.1964 | 0.3292 |
| Channel-independent control | 0.2771 | 0.8468 | 0.0204 | 0.2916 | 0.3343 |
| Compact neural forecast head | 0.7070 | 0.8505 | 0.0231 | 0.5013 | 0.1511 |
| Previous neural predictor, rerun | 0.7036 | 0.8547 | 0.0229 | 0.3750 | 0.2948 |
| Window ridge | 0.6311 | 0.8488 | 0.0270 | 0.2907 | 0.3758 |

The proposed local-regime variant does not provide a consistent improvement
over the previous neural predictor. Four regimes reduce VUS-PR relative to
one regime in all five source averages, for both reconstruction settings.
Auxiliary reconstruction provides a modest GutenTAG gain, with small or
mixed effects elsewhere. This does not support expanding this candidate
into a full multi-seed campaign at present.

The compact neural forecast head improves on the previous predictor on the
six GutenTAG pilot cases, but loses performance on HAI. Its higher BATADAL
VUS-PR is accompanied by slightly lower pointwise AUPRC. The primary variant's
HAI VUS-PR exceeds the previous predictor's value, but its AUPRC is lower
(0.3091 versus 0.3531). These metric differences preclude a general accuracy
claim for either experimental candidate.

## 3. Historical references on the same pilot support

Historical cached predictions were independently re-evaluated on the same
100-row exclusion and checked for label alignment. Complete extension
references are available for the 12 Exathlon/GHL/GutenTAG cases.

| Reference | Exathlon | GHL | GutenTAG |
|---|---:|---:|---:|
| DynSphere-E | 0.8458 | 0.0240 | 0.3813 |
| DynSphere-E1 | 0.8457 | 0.0233 | 0.3832 |
| MTAD-GAT (ML4ITS) | 0.8595 | 0.0239 | 0.9690 |
| Previous neural predictor, cached | 0.8547 | 0.0224 | 0.3750 |

Historical BATADAL/HAI reference tables are present, but their score arrays
were not found in the checked result roots. These eight missing references
remain explicit in `reference_issues.json`. The previous neural predictor
was rerun within the pilot, so its BATADAL/HAI values in Section 2 are available.
Historical training budgets differ from the prototype's capped window
training and must remain declared.

The compact forecast head remains substantially below MTAD-GAT on the
GutenTAG pilot (0.5013 versus 0.9690). The results have not established that
window encoding or reconstruction training reproduces MTAD-GAT's advantage.
They also suggest that replacing the trained forecast head with the current
ridge dynamics stage merits scrutiny. Predictor capacity, fitting objective,
and calibration differ, so the comparison does not isolate one explanation.

## 4. Complete extension VUS findings

The separate full VUS evaluation is complete for all 1,365 new records,
representing 1,023 distinct metric inputs. The combined 14-method comparison
uses 2,730 matched records on 65 cases: 25 Exathlon, 22 GHL, and 18 GutenTAG,
with three requested detector seeds. All available records are preserved.

DynSphere-E1 achieves 0.9234 VUS-PR on Exathlon, compared with 0.9323 for
MTAD-GAT and 0.9733 for TSB-AD OmniAnomaly. MTAD-GAT has higher VUS-PR than
E1 on all 25 Exathlon cases despite the relatively small mean difference.
On GutenTAG, DynSphere-E obtains 0.4334 and MTAD-GAT obtains 0.9569. GHL
remains weak for the proposed variants.

DynSphere-E retains a mechanism-specific strength on variance anomalies:
0.9983 VUS-PR, compared with 0.9195 for MTAD-GAT. Its mode-correlation result
is 0.8297, compared with 0.9127 for MTAD-GAT. The complete six-mechanism
average and every source remain reported. These observations do not support
broad superiority claims.

## 5. Outputs and remaining limitations

Pilot summaries are under
`results/window_dynsphere_v1/summaries/bb72fb42ddc0907b4829f4fd9c198f52d2977663284dc379f6670163a4dfe229/`.
The numerical audit is under `results/window_dynsphere_v1/audit/`.
The full extension VUS summaries are under
`results/extension_audit_20261023/new_methods_vus/`.

Runtime summaries distinguish actual fitting/refitting cost from a reused
representation. Fresh-process peak memory includes Python and dependency
overhead. The primary configuration averages 38.7 seconds for fitting/refitting
and 16.4 seconds for scoring across the 14 completed cases. The largest
fresh-process memory measurement across all configurations is approximately
4.61 GiB, for the channel-independent control. These timings are unsuitable
for a direct fresh-training comparison when a representation was reused.
Batch-shape score agreement uses absolute tolerance 1e-4 and
relative tolerance 1e-6 for float32 recurrent computation.

One deferred metadata issue remains: the shared metric helper emits a nested
`manuscript_ready` flag for metric-backend eligibility. Pilot records and
completion summaries explicitly retain top-level `manuscript_ready=false`.
The nested helper flag must not be used to assess paper readiness.

The 21 missing extension fits on the corrupted HPC GHL copy, TranAD seed
reset, and CATCH seed replayability remain separate unresolved issues.
The new prototype has one detector seed and no held-out confirmation source.
Its implementation and pilot are complete, but adoption as a replacement
method and a larger campaign are unsupported by the current evidence.
