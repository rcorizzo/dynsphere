# Window-conditioned DynSphere-E: experimental design

Status: approved in conversation on October 23, 2026; implementation plan pending review.

## 1. Intent and scope

The user accepted a compact neural encoder and its additional training cost.
The recommended direction is the window-conditioned Euclidean variant.
The numerical and architectural defaults below are proposed implementation
choices. They have not been selected using new test results.

The objective is to test whether a history representation that preserves
channel relationships and temporal structure improves regime-dependent
normal prediction. The working identifier is `window_dynsphere_euclidean`.
The name describes an experimental extension of the family. It does not
establish methodological novelty or an improvement over existing methods.

The first implementation is a research prototype with a bounded pilot.
The existing DynSphere-S/E detectors, baseline adapters, frozen campaigns,
scores, and manuscript remain unchanged. GPU execution, external submission,
large hyperparameter studies, and new paper claims are outside this stage.

## 2. Approaches considered

1. Window-conditioned local dynamics, recommended: a compact temporal encoder
   learns normal observation structure, followed by local linear dynamics
   and a normalized predictive density. This has a direct local-dynamics
   interpretation and supports controlled component comparisons.
2. Explicit graph-conditioned dynamics: add a learned channel graph. This
   introduces graph construction and regularization choices, and has close
   connections to existing graph-based anomaly detectors. It is deferred.
3. Frequency-conditioned dynamics: incorporate frequency representations.
   This has a narrower mechanism-level motivation and is deferred.

The accepted compact-encoder direction is developed as option 1. An explicit
attention layer is excluded from the initial prototype. Dense channel mixing
provides a simpler initial test of learned channel relationships.

The original MTAD-GAT paper uses feature and temporal attention with joint
forecasting and reconstruction. The evaluated ML4ITS implementation has
GATv2 and a GRU reconstruction decoder. Its leaderboard result motivates
these hypotheses but does not identify the contribution of attention.

Sources: https://arxiv.org/abs/2009.02040 and
https://github.com/ML4ITS/mtad-gat-pytorch.

## 3. Relationship to existing detectors

`PredictiveGenerativeDetector` already implements a causal GRU encoder,
Euclidean or spherical successor fields, Gaussian observation likelihoods,
and latent-mode filtering. Forecasting, recurrent encoding, and a unified
likelihood are established features of the repository.

The proposed experiment changes three specific aspects: a fixed history
window, an auxiliary reconstruction objective for the history representation,
and a two-stage fit with explicit local ridge predictors. It does not claim
that any of these ingredients is independently novel. Its contribution can
only be assessed through relevant prior work and controlled experiments.

Compared with the original DynSphere-E, the prototype changes the PCA state,
the prediction target, and the multi-channel score. The retained principle
is prediction conditioned on normal state-velocity regimes. Comparisons
with E1/E4 must account for these changes; results will be reported separately.

## 4. Data, boundaries, and causal alignment

Inputs follow the repository interface: observations, entity identifiers,
and optional times. Rows within each entity must be time ordered, with finite
features and strictly increasing times. The initial prototype supports
regular sampling only. Irregular spacing produces an explicit unsupported
input error; the implementation must not silently impose unit time steps.

For each normal training entity, the first 80% of rows form the fitting
partition and the last 20% form the calibration partition. Split indices
are computed before scaling or window construction. The feature scaler is
fitted on the fitting partition alone. An externally supplied training-only
scaler may already have been applied by a dataset loader; this additional
internal fitting-partition transformation must be recorded.

Let W_t contain the L observations ending at t-1. The model predicts x_t.
Neither the encoder input, regime weights, nor predictive mean may use x_t
or later observations. The score at t can use x_t as the observed target.
Auxiliary reconstruction targets are entirely inside W_t.

Windows never cross entity or train/test boundaries. Calibration windows
can use earlier fitting-partition rows from the same normal entity. The
test suffix starts with its own history, consistent with the extension
protocol. The first L test rows are marked invalid, using an explicit mask
and NaN scores. Evaluation excludes these rows for every compared method.

Set L=100. Each entity needs at least L+32 fitting targets and 32 calibration
targets. Insufficient data raises a documented incompatibility. The pilot
must report every rejected case; no automatic window reduction is allowed.

## 5. Model and fitting procedure

### 5.1 Compact history representation

Use one temporal convolution (kernel 5, 32 output channels) followed by a
single-layer GRU with hidden width 32. Convolution padding uses only past
positions. A linear projection of the final GRU state gives z_t in R^8.
There is no unit-sphere normalization or per-window amplitude normalization.
The convolution mixes observed channels. The latent dimension is fixed at
8, including when the observation dimension is smaller.

A small forecast head predicts x_t from z_t. A second decoder reconstructs
the complete history window from z_t. Both use one hidden layer of width 32.
The history loss is the mean squared reconstruction error over all L*d
entries. The forecast loss is the mean squared error over d target channels.
The objective is forecast_loss + lambda * history_loss, with lambda=1.
The lambda=0 control removes the reconstruction objective and decoder.

Train for 10 epochs using Adam, learning rate 1e-3, batch size 128, and
gradient-norm clipping at 5. Use at most 20,000 target windows from each
case, selected deterministically and approximately uniformly over the
fitting partition, with entity-balanced allocation. Record selected target
indices. Shuffle only this fixed window set using the requested seed.
Use float32 neural computation, CPU by default, and one Torch thread.
No early stopping or test-based model selection is performed.

### 5.2 Normal regimes and local dynamics

Freeze the encoder after training. For adjacent, valid, entity-local windows,
define v_t = z_t - z_(t-1). The predictor context is
q_t = [z_t, v_t, x_(t-1)]. The regime descriptor is [z_t, v_t].
The earliest target with a full window uses v_t=0; record this convention.

Standardize regime descriptors and predictor contexts using the fitting
targets only. Fit KMeans with the requested seed, n_init=10, and K=1 or 4
on the same selected fitting targets. The primary configuration is K=1.
The four-regime configuration is a prespecified mechanism comparison.

For each regime, use only the selected fitting targets assigned to that
cluster after merging. Fit a ridge predictor of the standardized observation
increment x_t-x_(t-1) from q_t, including an unpenalized intercept. Use
alpha=1 on mean squared loss, with a consistently normalized penalty.
The conditional mean is x_(t-1) plus the predicted increment. The neural
forecast head trains the representation and is discarded from this local
predictor after fitting. This two-stage procedure must be explicit in
documentation and saved metadata.

K=1 has weight one. For K=4, use soft weights proportional to
exp(-squared_distance_to_center / (2*T)). Distances use the standardized
regime descriptor. Set T to the median fitting-target squared distance to
the assigned center, lower bounded by 1e-6. Weights depend on W_t alone.
Any cluster with fewer than max(32, 2*(context_dimension+1)) fitting targets
is merged into its nearest qualifying cluster. If no cluster qualifies,
merge the smallest cluster into its nearest other cluster. Recompute counts
and centers after each merge, using original cluster indices to break ties.
Repeat until all remaining clusters qualify, allowing a final single cluster. Record
the requested and effective regime counts. A reduced count cannot support
a claim about four distinct learned regimes.

### 5.3 Predictive density and anomaly score

Use Gaussian observation residuals with a fixed diagonal covariance for
each regime. On the normal calibration partition, estimate each expert's
residual mean and centered variance using its past-only soft regime weights.
Add the residual mean to that expert's predictive mean. Shrink each local
variance toward the pooled calibration variance with 32 pseudo-observations.
Floor variances at 1e-6 in standardized observation units. If a regime has
zero effective calibration weight, use pooled mean and variance.

For clarity, let r_tk be each expert's observation residual before bias
correction, and pi_tk its past-only soft weight. The pooled residual mean is
sum_tk(pi_tk*r_tk)/N and the pooled centered variance is
sum_tk(pi_tk*(r_tk-pooled_mean)^2)/N, computed elementwise over N calibration
targets. Local means and centered variances use the corresponding expert's
weights. Its calibration weight mass is n_k=sum_t(pi_tk), and the final
variance is (n_k*local_variance + 32*pooled_variance)/(n_k+32), before flooring.
These definitions use probability-weighted expert residuals throughout.

The anomaly score is

    s_t = -log sum_k pi_k(W_t) N(x_t; mu_k(W_t), diag(sigma_k^2)).

Compute the mixture in log space using float64 residual arithmetic.
Covariances remain fixed at test time. Reconstruction error is a training
diagnostic only. It does not enter this score. Temporal rolling maxima,
state-speed evidence, and transition penalties are excluded.

## 6. Controls and bounded evaluation

The central factorial comparison is K in {1,4} by lambda in {0,1}, with
all other settings fixed. Include a channel-independent representation
control for K=1, lambda=1: a shared univariate convolution/GRU encoder is
applied to each channel independently; local ridge predictions use only
that channel's context, and the observation density factorizes by channel.
Its parameter count differs and must be reported. This control measures
the combined effect of channel coupling in representation and prediction.

Include a window-ridge control using the same L observations directly,
the same fitting/calibration split, a global predictor, alpha=1, and the
same Gaussian residual calibration. Also include the compact neural
forecast head from the lambda=0 fit as a scoring control, with its own
normal calibration covariance and no local ridge stage.

Retain existing DynSphere-E1/E4, the previous neural predictor, and the
declared MTAD-GAT implementation as reference methods. Re-score their
cached predictions on the common 100-row exclusion. Disclose differences
in available normal training rows and model fitting budgets. Comparisons
attributing gains to architecture require matched training-split controls;
the previous neural predictor must be rerun with the same 80/20 split.

The bounded pilot uses seed 37 and 15 predetermined cases: one generated
GutenTAG case per each of its six mechanisms, three Exathlon cases, three
GHL cases, and BATADAL, HAI-20.07, and PowerSystem. Select the Exathlon/GHL
cases by sorting SHA-256(case_id), taking the first three. Select the
GutenTAG case with the smallest declared generation seed per mechanism.
Freeze and save exact case identifiers before fitting. Required source
files must pass the existing frozen-hash checks, including the valid local
GHL copy. Any unavailable principal dataset is reported as pending.

Report AUPRC, AUROC, official VUS-PR and VUS-ROC, per-source and per-mechanism
results, fit/score time, peak memory, regime occupancy, covariance ranges,
and training losses. Treat the pilot as exploratory: these sources have
already informed model development. A held-out confirmation campaign needs
new source-native cases or a separately frozen generator design. Additional
seeds alone do not provide independent confirmation of the design choices.

No broad campaign is launched automatically. The pilot informs an explicit
decision to stop or proceed; all configurations and unsuccessful results
remain recorded. A continuation would freeze the model and evaluate seeds
37, 38, and 39 on all declared cases without selecting favorable sources.

## 7. Software isolation and verification

Create a separate prototype package outside the root-level detector modules
and existing official baseline adapters. Its runner uses a dedicated output
root and hashes its own source files plus the case manifest, hyperparameters,
seed, package versions, and input arrays. Preserve the existing extension
source fingerprint and VUS evaluator sources while VUS is running.

The estimator provides fit, decision_function, and decision_components.
Components include the validity mask, predictive means, regime weights,
effective regime count, and per-regime log density. Diagnostic arrays are
opt-in to avoid retaining large arrays during routine scoring. Inference
uses bounded batches and must not materialize every window at once.

Required tests cover future-observation invariance of earlier predictions
and scores, invariance of predictions to x_t, entity isolation, train/test
boundaries, fitting-only scaling, label-free fitting, deterministic replay,
the one-regime density identity, normalized mixture weights, tiny-cluster
merging, finite calibrated variance, constant channels, invalid times,
short sequences, batch-size invariance, and serialization/reload agreement.
Existing detector and evaluation tests must remain passing.

Builder choices include package layout, batching implementation, numerical
linear-algebra routines, and test fixtures, within these stated contracts.

## 8. Approval boundary

The user approved this written design on October 23, 2026. The approved
scope includes the isolated prototype, tests, and bounded pilot. Paper
changes, HPC submissions, and a full campaign remain separate actions.
