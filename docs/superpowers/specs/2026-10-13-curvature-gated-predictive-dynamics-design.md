# Curvature-Gated Predictive Dynamics: Method Design

**Date:** 2026-10-13  
**Status:** Proposed for review before implementation

## 1. Objective

The proposed method should preserve the strongest observed property of the
current Euclidean predictor while giving the spherical component a precise and
testable role. The central hypothesis is that normal temporal dynamics may
contain locally flat and positively curved regimes. A detector that learns
which geometry explains each transition can improve contextual and event-level
anomaly detection when both regimes are present.

The design addresses two concerns from the prior submission. First, the model
has one predictive probabilistic objective instead of an additive collection
of independently motivated scores. Second, every geometric component has a
matched control and an observable contribution.

## 2. Scope

The initial implementation will include:

1. A causal encoder and decoder shared by all geometric experts.
2. A flat expert with curvature fixed at zero.
3. A spherical expert with positive curvature.
4. A causal gate that assigns transition probabilities to the experts.
5. A single predictive negative log-likelihood anomaly score.
6. Flat-only, spherical-only, and mixed-geometry configurations.
7. Diagnostics for expert use, posterior probabilities, and collapse.

The initial implementation will exclude hyperbolic geometry, anomaly-label
tuning, ensembles of separately trained detectors, and a new benchmark-specific
post-processing rule.

## 3. Model

### 3.1 Shared causal representation

For a multivariate observation sequence, a causal encoder maps the history to
a latent state:

\[
z_t = h_\theta(x_{1:t}).
\]

Both experts receive the same state and context. This constraint supports a
matched comparison because differences originate in the transition geometry.

### 3.2 Geometry-specific transitions

The flat expert predicts an additive successor:

\[
\hat z_{t+1}^{(0)} = z_t + v_t^{(0)}.
\]

The spherical expert predicts a tangent velocity and maps it to a positively
curved latent space through the exponential map:

\[
\hat z_{t+1}^{(+)} = \operatorname{Exp}^{\kappa}_{z_t}
\left(v_t^{(+)}\right), \qquad \kappa > 0.
\]

The spherical expert therefore represents a transition assumption. It does not
serve as a separate score appended after prediction.

### 3.3 Causal geometry gate

A gate estimates expert probabilities using information available at time
\(t\):

\[
\pi_t = \operatorname{softmax}(g_\phi(z_t, c_t)).
\]

The gate cannot use future observations or anomaly labels. Entropy and
load-balancing regularization may be applied to reduce early expert collapse.
Their weights will be selected using normal validation likelihood.

### 3.4 Common observation model and anomaly score

A shared decoder produces the conditional observation distribution for each
expert. With Gaussian emissions, expert \(k\) defines
\(p_k(x_{t+1}\mid x_{1:t})\). The mixture likelihood is

\[
p(x_{t+1}\mid x_{1:t}) = \sum_k \pi_{t,k}
p_k(x_{t+1}\mid x_{1:t}).
\]

The anomaly score is the predictive negative log-likelihood:

\[
s_{t+1} = -\log \sum_k \pi_{t,k}
p_k(x_{t+1}\mid x_{1:t}).
\]

Training uses normal data only. This single score replaces independent
position, velocity, energy, and transition score terms.

## 4. Interpretation and Diagnostics

Each prediction will expose:

1. Prior gate probabilities \(\pi_{t,k}\).
2. Posterior expert responsibilities after observing \(x_{t+1}\).
3. Per-expert predictive log likelihoods.
4. Transition curvature and tangent velocity norms.
5. Expert utilization by dataset, source, and time segment.

An expert is considered inactive when its mean responsibility or effective
sample size remains below a prespecified threshold. The threshold is a
diagnostic and cannot alter test scores. If the spherical expert is inactive
across datasets, the evidence supports retaining the flat predictor as the
principal method.

## 5. Experimental Evaluation

### 5.1 Required comparisons

The study will compare:

1. The current one-regime Euclidean predictor.
2. The matched flat-only configuration of the new model.
3. The matched spherical-only configuration.
4. The mixed flat and spherical model.
5. The submitted DynSphere detector.
6. The expanded external baseline set.

The matched configurations will use the same encoder, decoder, capacity,
training budget, thresholding procedure, and data partitions.

### 5.2 Data and metrics

The initial evaluation will include the contextual synthetic benchmark,
BATADAL, HAI, MSL, SMAP, SMD, and the selected TSB-AD-M campaign. Reported
metrics will include AUPRC, AUROC, VUS-PR, VUS-ROC, event precision, event
recall, and event F1 when supported by the dataset protocol.

The primary study uses seed 37. Seeds 38 and 39 follow for configurations that
remain competitive. Paired bootstrap confidence intervals will be computed at
the dataset or source level where the sample count permits them.

### 5.3 Advancement criteria

The mixed model advances to the main paper only if all of the following hold:

1. Its aggregate performance matches the flat control within uncertainty.
2. It improves at least one prespecified contextual, event-level, or source
   family with corresponding uncertainty estimates.
3. Both experts receive meaningful responsibility on at least one real dataset.
4. Gains persist under matched capacity and training budget.
5. Hyperparameters are selected without test anomaly labels.

If these criteria are not met, the flat predictor remains the primary method
and the curvature study is reported as an informative ablation.

## 6. Paper Framing

The working name is **Curvature-Gated Predictive Dynamics**. The pendulum
provides geometric intuition for constrained motion and contextual anomalies,
but the method is defined through predictive transition geometry. The submitted
DynSphere detector is presented as the predecessor that motivated the unified
probabilistic model.

Claims will be limited to evidence from the matched controls. The paper will
not describe the Euclidean control as DynSphere unless the implemented model
actually shares the proposed architecture and training objective.

## 7. Implementation Boundaries

The implementation should introduce:

1. Geometry interfaces for flat and spherical transitions.
2. A shared causal encoder and decoder.
3. A differentiable mixture likelihood and causal gate.
4. Stable exponential-map calculations near zero velocity.
5. Configuration files for the three matched variants.
6. Result fields for responsibilities, curvature, and device metadata.
7. Unit tests for geometric limits, causality, likelihood consistency, and
   deterministic inference.

The zero-curvature limit of the spherical transition should converge
numerically to the flat transition. This property supplies a direct consistency
test and connects the two experts within one model family.

## 8. Risks and Decision Rules

1. **Expert collapse:** apply normal-data regularization, record utilization,
   and reject claims that rely on an inactive expert.
2. **No predictive gain:** retain the flat model and report the curvature
   comparison without promoting the mixed model.
3. **Capacity confounding:** match trainable parameter counts and training
   budgets in all principal comparisons.
4. **Unstable geometry:** test the zero-curvature and small-velocity limits and
   use numerically stable manifold operations.
5. **Excessive computation:** use the laptop for unit tests and reduced cases,
   followed by AU or PLGrid for the full study.

## 9. Review Gate

Implementation begins after confirmation of the following choices:

1. The two-expert scope is sufficient for the first study.
2. Predictive negative log likelihood is the sole primary anomaly score.
3. The advancement criteria are acceptable.
4. The pendulum remains motivation and is not the formal definition of the
   detector.

