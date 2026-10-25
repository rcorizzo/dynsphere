# Matched DynSphere-E1 component study

This protocol is fixed before inspecting the component results. It uses all
66 cases and detector seeds 37, 38, and 39 from extension campaign 754b33fc172b.
The reference configuration is the existing one-regime Euclidean DynSphere.
The study evaluates a mechanism question and does not select a new best
configuration for the principal comparison.

Six leave-one-component-out configurations remove position, latent dynamics,
raw dynamics, state-speed deviation, velocity deviation, or regime distance.
Transition penalties are identically zero in a one-regime model, so their
removal is not a separate experiment. Four propagation configurations use
radii 0, 1, 4, and 16; the reference radius is 8. There are 11 score
configurations per case/seed, or 2,178 planned metric records.

All components are calibrated from the same normal training data. In E1,
changing score weights cannot change the single selected concept. Therefore,
the leave-one-out scores can be computed from one fitted model's component
arrays. A regression test compares these scores to explicitly refitted models
with the corresponding weight set to zero. This shortcut is restricted to E1.

Each reconstructed reference score must agree with its frozen cached score.
The common first test row is excluded from all metrics. Detector seeds are
averaged within case before source-level means. Every source, including GHL,
is retained. Saved scores and provenance permit subsequent VUS evaluation.
The manuscript-readiness flag remains false pending comparison review.
