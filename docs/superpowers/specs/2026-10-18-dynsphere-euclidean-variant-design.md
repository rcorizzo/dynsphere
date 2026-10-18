# Matched DynSphere Euclidean Variant Design

**Status:** Approved in conversation on October 18, 2026

## 1. Objective

Add a Euclidean member of the DynSphere family that changes latent geometry
while preserving the remaining detector structure. The resulting experiment
must separate the effects of geometry and concept count. It must also preserve
the numerical behavior and public interface of the existing spherical
detector.

## 2. Scientific motivation

The current one-step Euclidean predictor is a useful baseline, but it differs
from DynSphere in its temporal objective, mode structure, probabilistic score,
and evidence aggregation. A comparison between that predictor and DynSphere
cannot isolate latent geometry.

The proposed variant introduces a controlled two-by-two comparison:

| Method | Geometry | Concepts |
| --- | --- | ---: |
| DynSphere-S | spherical | 4 |
| DynSphere-E | Euclidean | 4 |
| DynSphere-S1 | spherical | 1 |
| DynSphere-E1 | Euclidean | 1 |

The existing one-step Euclidean predictor remains an independent predictive
baseline. The earlier `ManifoldDynamicsDetector` remains a separate
likelihood-based experimental method.

## 3. Shared detector structure

All four DynSphere configurations will share:

1. training-only feature standardization;
2. PCA with the same latent dimension;
3. sequence-specific velocity and acceleration estimation;
4. clustering of concatenated latent state and velocity;
5. concept-specific ridge prediction of latent acceleration;
6. concept-specific ridge prediction of raw standardized-feature acceleration;
7. robust median/MAD calibration of scalar evidence channels;
8. a first-order concept-transition model with identical smoothing;
9. identical component weights;
10. selection of the minimum complete local concept score; and
11. identical entity-wise rolling-maximum propagation.

The four-concept configurations use `n_concepts=4`. The one-concept
configurations use `n_concepts=1`; their transition evidence is consequently
constant.

## 4. Geometry-specific definitions

Let `u_t` denote the PCA latent vector.

### 4.1 Spherical geometry

DynSphere-S retains the current definitions:

\[
z_t = \frac{u_t}{\max(\lVert u_t\rVert_2,\epsilon)}.
\]

Velocity and acceleration are finite differences projected onto the tangent
space of the unit sphere. Position evidence uses angular distance to a concept
center. The global potential is

\[
V_t^{(S)} = 1-z_t^\top g_S,
\]

and the state-speed statistic is

\[
E_t^{(S)} = \frac12\lVert v_t\rVert_2^2 + V_t^{(S)}.
\]

### 4.2 Euclidean geometry

DynSphere-E retains the unnormalized PCA vector:

\[
y_t = u_t.
\]

Velocity and acceleration are ordinary Euclidean finite differences, with the
same sequence-boundary handling used by DynSphere-S. Position evidence uses

\[
d_{tm}^{(E)} = \lVert y_t-\mu_m\rVert_2,
\]

where `mu_m` is the mean latent state for concept `m`. The global equilibrium
is the mean normal latent state `g_E`, and the Euclidean potential is

\[
V_t^{(E)} = \frac12\lVert y_t-g_E\rVert_2^2.
\]

The Euclidean state-speed statistic is

\[
E_t^{(E)} = \frac12\lVert \dot y_t\rVert_2^2 + V_t^{(E)}.
\]

Concept-specific median/MAD calibration places the spherical and Euclidean
position, dynamics, state-speed, and velocity channels on the same robust
score scale before the fixed weights are applied.

## 5. Software architecture

### 5.1 Shared core

A geometry-aware internal detector core will own preprocessing, concept
learning, ridge models, robust calibration, transition estimation, local-score
selection, and propagation. Geometry-specific operations will be limited to:

1. state construction;
2. derivative projection;
3. position distance;
4. equilibrium construction; and
5. potential calculation.

### 5.2 Backward compatibility

`MultiConceptPendulumDetector` will remain the public spherical class with its
current constructor defaults and component names. Existing spherical calls
must produce numerically equivalent scores and component arrays within strict
floating-point tolerance.

A new `MultiConceptEuclideanDetector` class will expose the matched Euclidean
configuration. Its generic position and latent-dynamics channels will map to
the same analysis categories as the spherical angular and spherical-dynamics
channels.

### 5.3 Experiment registration

The official evaluation registry will add:

1. `dynsphere`;
2. `dynsphere_euclidean`;
3. `dynsphere_one_regime`; and
4. `dynsphere_euclidean_one_regime`.

The existing `predictive_euclidean_single` entry will remain unchanged. Every
run record will identify its geometry, concept count, component weights, and
implementation revision.

## 6. Experiment campaigns

### 6.1 DynSphere geometry campaign

The geometry campaign will evaluate the four DynSphere configurations and the
one-step Euclidean predictor on:

1. DynSphere-Contextual;
2. MSL;
3. SMAP;
4. SMD;
5. BATADAL;
6. HAI-20.07; and
7. PowerSystem.

Seeds 37, 38, and 39 will be used. The campaign contains 945 logical runs
before resumable reuse: 63 cases, five methods, and three seeds. It will use a
dedicated manifest and CPU scheduler array. Existing result directories will
be preserved.

### 6.2 Official-baseline follow-up

The official-baseline campaign will add seeds 38 and 39 for DAGMM, CATCH,
MTAD-GAT, the declared OmniAnomaly implementation, Anomaly Transformer, and
TranAD on the same seven dataset groups. This campaign contains 756 logical
runs and uses separate CPU and H100 manifests.

### 6.3 Aggregation

Campaign manifests will be merged only after completion checks. Aggregation
will average seeds within trajectories, trajectories within source families,
and source families with equal weight. Non-complete runs and metric warnings
will remain visible in the readiness report.

## 7. Statistical comparisons

The primary factorial contrasts are:

1. geometry with four concepts: DynSphere-E minus DynSphere-S;
2. geometry with one concept: DynSphere-E1 minus DynSphere-S1;
3. concept count under spherical geometry: DynSphere-S minus DynSphere-S1;
4. concept count under Euclidean geometry: DynSphere-E minus DynSphere-E1.

AUPRC and VUS-PR are primary metrics. AUROC, VUS-ROC, standardized partial
AUROC at five percent false-positive rate, and event metrics are secondary.
Multi-file sources will use paired trajectory-level intervals. Single-system
datasets will use paired seed summaries and temporal block uncertainty where
score vectors are available.

All configurations will be reported. Test labels will not be used to select a
geometry, concept count, score weight, or propagation radius.

## 8. Validation requirements

Implementation is complete only when:

1. a frozen spherical regression test confirms preservation of current scores;
2. both geometries return finite, row-aligned scores;
3. each component decomposition sums exactly to the unpropagated score;
4. a controlled latent-magnitude perturbation leaves the spherical state
   direction unchanged and changes the Euclidean position and potential
   evidence as specified;
5. entity boundaries prevent cross-sequence derivatives and transitions;
6. one-concept variants have constant transition evidence;
7. registry and campaign tests confirm distinct method identities and
   manifests;
8. scheduler scripts pass shell syntax checks; and
9. a small end-to-end campaign completes and aggregates all five local methods.

## 9. Paper positioning

DynSphere-S is the retained standalone method. DynSphere-E is its matched
Euclidean geometry ablation. DynSphere-S1 and DynSphere-E1 isolate the effect
of multiple concepts. The one-step Euclidean predictor is an external
predictive baseline. DynSphere++ remains an optional deployment ensemble.

The paper will describe the factorial comparison before presenting its
results. Claims about spherical geometry or multiple concepts will follow the
paired evidence and will remain source dependent.
