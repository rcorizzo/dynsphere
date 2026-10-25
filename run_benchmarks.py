#!/usr/bin/env python3
"""Train/test runner for PyOD, neural baselines, and the pendulum detector."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from manifold_dynamics_detector import ManifoldDynamicsDetector
from boundary_change_point_detector import BoundaryChangePointDetector
from semi_markov_change_point_detector import SemiMarkovChangePointDetector
from pendulum_detector import MultiConceptPendulumDetector
from predictive_generative_detector import PredictiveGenerativeDetector
from segment_predictive_detector import SegmentPredictiveDetector
from unified_generative_detector import UnifiedGenerativeDetector


def load_pyod_models(contamination: float, random_state: int) -> dict[str, object]:
    try:
        from pyod.models.abod import ABOD
        from pyod.models.copod import COPOD
        from pyod.models.hbos import HBOS
        from pyod.models.iforest import IForest
        from pyod.models.knn import KNN
        from pyod.models.lof import LOF
        from pyod.models.ocsvm import OCSVM
        from pyod.models.pca import PCA
    except ImportError as exc:
        raise SystemExit(
            "PyOD is not installed. Install requirements first, for example:\n"
            "  python3 -m pip install -r requirements.txt"
        ) from exc

    return {
        "LOF": LOF(n_neighbors=35, contamination=contamination),
        "KNN": KNN(n_neighbors=35, contamination=contamination),
        "IForest": IForest(n_estimators=250, contamination=contamination, random_state=random_state),
        "ABOD": ABOD(n_neighbors=35, contamination=contamination),
        "OCSVM": OCSVM(contamination=contamination),
        "HBOS": HBOS(contamination=contamination),
        "COPOD": COPOD(contamination=contamination),
        "PCA": PCA(contamination=contamination, random_state=random_state),
    }


def metric_dict(
    y_true: np.ndarray,
    scores: np.ndarray,
    include_vus: bool = False,
    entity_ids: np.ndarray | None = None,
    vus_window: int | None = None,
) -> dict[str, float | int | str | None]:
    positives = int(y_true.sum())
    result: dict[str, float | int | None] = {
        "n": int(len(y_true)),
        "positives": positives,
        "auroc": None,
        "auprc": None,
    }
    if positives == 0 or positives == len(y_true):
        return result
    result["auroc"] = float(roc_auc_score(y_true, scores))
    result["auprc"] = float(average_precision_score(y_true, scores))
    from range_metrics import best_plain_f1, best_point_adjusted_f1, practical_range_metrics

    result.update(best_plain_f1(y_true, scores))
    result.update(best_point_adjusted_f1(y_true, scores))
    result.update(practical_range_metrics(y_true, scores))
    if include_vus:
        from range_metrics import entity_macro_vus

        result.update(entity_macro_vus(y_true, scores, entity_ids, vus_window))
    return result


def evaluate_by_type(
    test: pd.DataFrame,
    scores: np.ndarray,
    include_vus: bool = False,
    vus_window: int | None = None,
) -> dict[str, dict[str, float | int | str | None]]:
    entity_ids = test["entity_id"].to_numpy() if "entity_id" in test.columns else None
    out = {
        "overall": metric_dict(
            test["is_anomaly"].to_numpy(),
            scores,
            include_vus=include_vus,
            entity_ids=entity_ids,
            vus_window=vus_window,
        )
    }
    normal = test["anomaly_type"].eq("none").to_numpy()
    for anomaly_type in sorted(x for x in test["anomaly_type"].unique() if x != "none"):
        mask = normal | test["anomaly_type"].eq(anomaly_type).to_numpy()
        y_type = test.loc[mask, "is_anomaly"].to_numpy()
        entity_type = entity_ids[mask] if entity_ids is not None else None
        out[anomaly_type] = metric_dict(
            y_type,
            scores[mask],
            include_vus=include_vus,
            entity_ids=entity_type,
            vus_window=vus_window,
        )
    return out


def print_table(results: dict[str, dict[str, dict[str, float | int | None]]]) -> None:
    print("\nOverall metrics")
    print(f"{'method':<18} {'AUROC':>8} {'AUPRC':>8} {'n':>8} {'pos':>6}")
    for name, result in results.items():
        overall = result["overall"]
        auroc = overall["auroc"]
        auprc = overall["auprc"]
        print(
            f"{name:<18} "
            f"{auroc if auroc is not None else float('nan'):>8.4f} "
            f"{auprc if auprc is not None else float('nan'):>8.4f} "
            f"{overall['n']:>8} {overall['positives']:>6}"
        )

    print("\nAUPRC by anomaly type")
    anomaly_types = [k for k in next(iter(results.values())).keys() if k != "overall"]
    print(f"{'method':<18} " + " ".join(f"{name[:18]:>18}" for name in anomaly_types))
    for method, result in results.items():
        values = []
        for anomaly_type in anomaly_types:
            auprc = result[anomaly_type]["auprc"]
            values.append(f"{auprc if auprc is not None else float('nan'):>18.4f}")
        print(f"{method:<18} " + " ".join(values))


def fit_score_pyod(
    name: str,
    model: object,
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> np.ndarray:
    del name
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    model.fit(x_train_scaled)
    return np.asarray(model.decision_function(x_test_scaled), dtype=float)


def run_neural_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    random_state: int,
    timings: dict[str, float],
    include_vus: bool = False,
    vus_window: int | None = None,
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    from neural_baselines import (
        DynSphereAEDetector,
        DynSpherePlusPlusDetector,
        ForecastingMLPDetector,
        MultiSphereDeepSVDDDetector,
        SequenceAutoencoderDetector,
        StaticAutoencoderDetector,
        WindowKNNDetector,
    )

    x_train = train[feature_cols].to_numpy()
    x_test = test[feature_cols].to_numpy()
    out = {}

    neural_jobs = [
        ("StaticAE", StaticAutoencoderDetector(random_state=random_state)),
        ("SequenceAE", SequenceAutoencoderDetector(random_state=random_state)),
        ("WindowKNN", WindowKNNDetector()),
        ("ForecastMLP", ForecastingMLPDetector(random_state=random_state)),
        ("DeepSVDD-KMeans", MultiSphereDeepSVDDDetector(random_state=random_state)),
        ("DeepSVDD-OracleRegime", MultiSphereDeepSVDDDetector(random_state=random_state)),
        ("DynSphereAE", DynSphereAEDetector(random_state=random_state)),
        ("DynSphere++", DynSpherePlusPlusDetector(random_state=random_state)),
    ]

    for name, model in neural_jobs:
        print(f"Running {name}...", flush=True)
        start = time.perf_counter()
        if name in {"SequenceAE", "WindowKNN", "ForecastMLP"}:
            model.fit(x_train, train["entity_id"].to_numpy())
            scores = model.decision_function(x_test, test["entity_id"].to_numpy())
        elif name in {"DynSphereAE", "DynSphere++"}:
            model.fit(x_train, train["entity_id"].to_numpy(), train["time"].to_numpy())
            scores = model.decision_function(x_test, test["entity_id"].to_numpy(), test["time"].to_numpy())
        elif name == "DeepSVDD-OracleRegime":
            model.fit(x_train, train["regime_id"].to_numpy())
            scores = model.decision_function(x_test)
        else:
            model.fit(x_train)
            scores = model.decision_function(x_test)
        timings[name] = time.perf_counter() - start
        out[name] = evaluate_by_type(test, scores, include_vus, vus_window)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/pendulum_multiconcept_train.csv")
    parser.add_argument("--test", default="data/pendulum_multiconcept_test.csv")
    parser.add_argument("--out", default="results/benchmark_results.json")
    parser.add_argument("--feature-prefix", default="x_")
    parser.add_argument("--contamination", type=float, default=0.04)
    parser.add_argument("--random-state", type=int, default=37)
    parser.add_argument("--max-train-rows", type=int, default=5000)
    parser.add_argument("--skip-pyod", action="store_true")
    parser.add_argument("--include-neural", action="store_true")
    parser.add_argument(
        "--include-manifold",
        action="store_true",
        help="Run the matched conditional-dynamics models on spherical and Euclidean latent spaces.",
    )
    parser.add_argument(
        "--skip-legacy-dynsphere",
        action="store_true",
        help="Skip DynSphere and its legacy score-component ablations.",
    )
    parser.add_argument(
        "--include-local-manifold",
        action="store_true",
        help="Add nonlinear neighbor-conditioned manifold fields to the matched comparison.",
    )
    parser.add_argument("--manifold-latent-dim", type=int, default=5)
    parser.add_argument("--manifold-modes", type=int, default=4)
    parser.add_argument("--manifold-ema-alpha", type=float, default=1.0)
    parser.add_argument("--manifold-calibration-fraction", type=float, default=0.2)
    parser.add_argument("--manifold-local-neighbors", type=int, default=64)
    parser.add_argument(
        "--include-unified",
        action="store_true",
        help="Run the jointly trained observation--dynamics likelihood and its factor ablations.",
    )
    parser.add_argument("--unified-latent-dim", type=int, default=6)
    parser.add_argument("--unified-hidden-dim", type=int, default=48)
    parser.add_argument("--unified-modes", type=int, default=4)
    parser.add_argument("--unified-epochs", type=int, default=15)
    parser.add_argument("--unified-sequence-length", type=int, default=128)
    parser.add_argument(
        "--include-predictive",
        action="store_true",
        help="Run the causal mode-conditioned next-observation likelihood.",
    )
    parser.add_argument("--predictive-latent-dim", type=int, default=6)
    parser.add_argument("--predictive-hidden-dim", type=int, default=48)
    parser.add_argument("--predictive-modes", type=int, default=4)
    parser.add_argument("--predictive-epochs", type=int, default=15)
    parser.add_argument("--predictive-sequence-length", type=int, default=128)
    parser.add_argument(
        "--include-identifiable-predictive",
        action="store_true",
        help="Run self-supervised mode-identifiability variants of the predictive likelihood.",
    )
    parser.add_argument(
        "--include-segment-predictive",
        action="store_true",
        help="Run the multi-horizon segment-continuity likelihood ratio.",
    )
    parser.add_argument(
        "--segment-horizons",
        nargs="+",
        type=int,
        default=[2, 4, 8, 16, 32],
    )
    parser.add_argument("--segment-rollout-epochs", type=int, default=0)
    parser.add_argument("--segment-probabilistic-rollout", action="store_true")
    parser.add_argument(
        "--include-boundary-change-point",
        action="store_true",
        help="Run the explicit continuation-versus-reset boundary likelihood ratio.",
    )
    parser.add_argument("--boundary-reset-components", type=int, default=4)
    parser.add_argument(
        "--include-semi-markov-change-point",
        action="store_true",
        help="Marginalize boundary evidence under a fixed semi-Markov duration prior.",
    )
    parser.add_argument("--semi-markov-max-duration", type=int, default=32)
    parser.add_argument("--semi-markov-mean-duration", type=float, default=16.0)
    parser.add_argument(
        "--semi-markov-duration-grid",
        nargs="+",
        type=float,
        default=None,
        help="Report a predeclared duration-prior sensitivity grid from one fitted model.",
    )
    parser.add_argument("--semi-markov-boundary-hazard", type=float, default=0.01)
    parser.add_argument("--include-vus", action="store_true")
    parser.add_argument("--vus-window", type=int, default=None)
    args = parser.parse_args()
    if args.include_local_manifold:
        args.include_manifold = True

    train = pd.read_csv(args.train).sort_values(["entity_id", "time"]).reset_index(drop=True)
    test = pd.read_csv(args.test).sort_values(["entity_id", "time"]).reset_index(drop=True)
    feature_cols = [c for c in train.columns if c.startswith(args.feature_prefix)]
    if not feature_cols:
        raise SystemExit(f"No feature columns found with prefix {args.feature_prefix!r}")

    train_fit = train
    if args.max_train_rows and len(train) > args.max_train_rows:
        train_fit = train.sample(n=args.max_train_rows, random_state=args.random_state).sort_values(["entity_id", "time"])

    x_train = train_fit[feature_cols].to_numpy()
    x_test = test[feature_cols].to_numpy()
    y_test = test["is_anomaly"].to_numpy()
    if y_test.sum() == 0:
        raise SystemExit("The test file has no positive labels.")

    results: dict[str, dict[str, dict[str, float | int | None]]] = {}
    timings: dict[str, float] = {}

    if not args.skip_pyod:
        for name, model in load_pyod_models(args.contamination, args.random_state).items():
            start = time.perf_counter()
            print(f"Running {name}...", flush=True)
            scores = fit_score_pyod(name, model, x_train, x_test)
            timings[name] = time.perf_counter() - start
            results[name] = evaluate_by_type(test, scores, args.include_vus, args.vus_window)

    if args.include_neural:
        results.update(
            run_neural_baselines(
                train,
                test,
                feature_cols,
                args.random_state,
                timings,
                args.include_vus,
                args.vus_window,
            )
        )

    x_train_full = train[feature_cols].to_numpy()

    if args.include_predictive:
        predictive_models = {
            "PredictiveGenerative-Euclidean": {
                "geometry": "euclidean",
                "n_modes": args.predictive_modes,
            },
            "PredictiveGenerative-Sphere": {
                "geometry": "sphere",
                "n_modes": args.predictive_modes,
            },
            "PredictiveGenerative-EuclideanSingleMode": {
                "geometry": "euclidean",
                "n_modes": 1,
            },
        }
        for name, configuration in predictive_models.items():
            print(f"Running {name}...", flush=True)
            start = time.perf_counter()
            model = PredictiveGenerativeDetector(
                latent_dim=args.predictive_latent_dim,
                hidden_dim=args.predictive_hidden_dim,
                epochs=args.predictive_epochs,
                sequence_length=args.predictive_sequence_length,
                random_state=args.random_state,
                **configuration,
            )
            model.fit(
                x_train_full,
                train["entity_id"].to_numpy(),
                train["time"].to_numpy(),
            )
            scores = model.decision_function(
                x_test,
                test["entity_id"].to_numpy(),
                test["time"].to_numpy(),
            )
            timings[name] = time.perf_counter() - start
            results[name] = evaluate_by_type(
                test,
                scores,
                args.include_vus,
                args.vus_window,
            )

    if args.include_identifiable_predictive:
        identifiable_common = {
            "geometry": "euclidean",
            "mode_specific_decoders": True,
            "contrastive_weight": 0.5,
            "contrastive_margin": 1.0,
            "diversity_weight": 0.05,
            "diversity_margin": 0.5,
            "balance_weight": 0.01,
            "persistence_weight": 0.01,
        }
        identifiable_models = {
            "IdentifiablePredictive-Euclidean": args.predictive_modes,
            "IdentifiablePredictive-EuclideanSingleMode": 1,
        }
        for name, modes in identifiable_models.items():
            print(f"Running {name}...", flush=True)
            start = time.perf_counter()
            model = PredictiveGenerativeDetector(
                latent_dim=args.predictive_latent_dim,
                hidden_dim=args.predictive_hidden_dim,
                n_modes=modes,
                epochs=args.predictive_epochs,
                sequence_length=args.predictive_sequence_length,
                random_state=args.random_state,
                **identifiable_common,
            )
            model.fit(
                x_train_full,
                train["entity_id"].to_numpy(),
                train["time"].to_numpy(),
            )
            scores = model.decision_function(
                x_test,
                test["entity_id"].to_numpy(),
                test["time"].to_numpy(),
            )
            timings[name] = time.perf_counter() - start
            results[name] = evaluate_by_type(
                test,
                scores,
                args.include_vus,
                args.vus_window,
            )

    if args.include_segment_predictive:
        if args.segment_probabilistic_rollout:
            name = "SegmentContinuity-StateSpace-Euclidean"
        elif args.segment_rollout_epochs > 0:
            name = "SegmentContinuity-Multistep-Euclidean"
        else:
            name = "SegmentContinuity-Euclidean"
        print(f"Running {name}...", flush=True)
        start = time.perf_counter()
        model = SegmentPredictiveDetector(
            horizons=tuple(args.segment_horizons),
            geometry="euclidean",
            latent_dim=args.predictive_latent_dim,
            hidden_dim=args.predictive_hidden_dim,
            epochs=args.predictive_epochs,
            rollout_epochs=args.segment_rollout_epochs,
            probabilistic_rollout=args.segment_probabilistic_rollout,
            sequence_length=args.predictive_sequence_length,
            random_state=args.random_state,
        )
        model.fit(
            x_train_full,
            train["entity_id"].to_numpy(),
            train["time"].to_numpy(),
        )
        scores = model.decision_function(
            x_test,
            test["entity_id"].to_numpy(),
            test["time"].to_numpy(),
        )
        timings[name] = time.perf_counter() - start
        results[name] = evaluate_by_type(
            test,
            scores,
            args.include_vus,
            args.vus_window,
        )

    if args.include_boundary_change_point:
        name = "BoundaryChangePoint-Euclidean"
        print(f"Running {name}...", flush=True)
        start = time.perf_counter()
        model = BoundaryChangePointDetector(
            horizons=(1, *tuple(args.segment_horizons)),
            latent_dim=args.predictive_latent_dim,
            hidden_dim=args.predictive_hidden_dim,
            epochs=args.predictive_epochs,
            sequence_length=args.predictive_sequence_length,
            rollout_epochs=max(1, args.segment_rollout_epochs),
            reset_components=args.boundary_reset_components,
            random_state=args.random_state,
        )
        model.fit(
            x_train_full,
            train["entity_id"].to_numpy(),
            train["time"].to_numpy(),
        )
        scores = model.decision_function(
            x_test,
            test["entity_id"].to_numpy(),
            test["time"].to_numpy(),
        )
        timings[name] = time.perf_counter() - start
        results[name] = evaluate_by_type(
            test,
            scores,
            args.include_vus,
            args.vus_window,
        )

    if args.include_semi_markov_change_point:
        base_name = "SemiMarkovChangePoint-Euclidean"
        print(f"Running {base_name}...", flush=True)
        start = time.perf_counter()
        duration_grid = (
            args.semi_markov_duration_grid
            if args.semi_markov_duration_grid
            else [args.semi_markov_mean_duration]
        )
        if any(duration <= 1.0 for duration in duration_grid):
            raise ValueError("All semi-Markov mean durations must exceed one.")
        model = SemiMarkovChangePointDetector(
            max_duration=args.semi_markov_max_duration,
            mean_duration=duration_grid[0],
            boundary_hazard=args.semi_markov_boundary_hazard,
            latent_dim=args.predictive_latent_dim,
            hidden_dim=args.predictive_hidden_dim,
            epochs=args.predictive_epochs,
            sequence_length=args.predictive_sequence_length,
            rollout_epochs=max(1, args.segment_rollout_epochs),
            reset_components=args.boundary_reset_components,
            random_state=args.random_state,
        )
        model.fit(
            x_train_full,
            train["entity_id"].to_numpy(),
            train["time"].to_numpy(),
        )
        elapsed = time.perf_counter() - start
        for duration in duration_grid:
            model.mean_duration = duration
            model._fit_log_age_prior()
            scores = model.decision_function(
                x_test,
                test["entity_id"].to_numpy(),
                test["time"].to_numpy(),
            )
            name = (
                f"{base_name}-D{duration:g}"
                if len(duration_grid) > 1
                else base_name
            )
            timings[name] = elapsed if duration == duration_grid[0] else 0.0
            results[name] = evaluate_by_type(
                test,
                scores,
                args.include_vus,
                args.vus_window,
            )

    if args.include_unified:
        train_entities = train["entity_id"].to_numpy()
        train_times = train["time"].to_numpy()
        test_entities = test["entity_id"].to_numpy()
        test_times = test["time"].to_numpy()
        for geometry in ("euclidean", "sphere"):
            print(f"Running UnifiedGenerative-{geometry.title()}...", flush=True)
            start = time.perf_counter()
            model = UnifiedGenerativeDetector(
                geometry=geometry,
                latent_dim=args.unified_latent_dim,
                hidden_dim=args.unified_hidden_dim,
                n_modes=args.unified_modes,
                epochs=args.unified_epochs,
                sequence_length=args.unified_sequence_length,
                random_state=args.random_state,
            )
            model.fit(x_train_full, train_entities, train_times)
            elapsed = time.perf_counter() - start
            for factor in ("joint", "observation", "dynamics"):
                name = f"UnifiedGenerative-{geometry.title()}-{factor.title()}"
                scores = model.decision_function(
                    x_test,
                    test_entities,
                    test_times,
                    score_factor=factor,
                )
                timings[name] = elapsed if factor == "joint" else 0.0
                results[name] = evaluate_by_type(
                    test,
                    scores,
                    args.include_vus,
                    args.vus_window,
                )

    if args.include_manifold:
        manifold_models = {
            "ManifoldDynamics-Sphere": {"geometry": "sphere", "include_radius": True},
            "ManifoldDynamics-SphereNoRadius": {"geometry": "sphere", "include_radius": False},
            "ManifoldDynamics-Euclidean": {"geometry": "euclidean", "include_radius": False},
            "ManifoldDynamics-SphereSingleMode": {
                "geometry": "sphere",
                "include_radius": True,
                "n_modes": 1,
            },
        }
        if args.include_local_manifold:
            manifold_models.update(
                {
                    "ManifoldDynamics-SphereLocal": {
                        "geometry": "sphere",
                        "include_radius": True,
                        "field_model": "local",
                    },
                    "ManifoldDynamics-SphereLocalNoRadius": {
                        "geometry": "sphere",
                        "include_radius": False,
                        "field_model": "local",
                    },
                    "ManifoldDynamics-EuclideanLocal": {
                        "geometry": "euclidean",
                        "include_radius": False,
                        "field_model": "local",
                    },
                    "ManifoldDynamics-SphereLocalSingleMode": {
                        "geometry": "sphere",
                        "include_radius": True,
                        "field_model": "local",
                        "n_modes": 1,
                    },
                }
            )
        for name, configuration in manifold_models.items():
            print(f"Running {name}...", flush=True)
            start = time.perf_counter()
            model_configuration = {
                key: value for key, value in configuration.items() if key != "n_modes"
            }
            model = ManifoldDynamicsDetector(
                latent_dim=args.manifold_latent_dim,
                n_modes=int(configuration.get("n_modes", args.manifold_modes)),
                ema_alpha=args.manifold_ema_alpha,
                calibration_fraction=args.manifold_calibration_fraction,
                local_neighbors=args.manifold_local_neighbors,
                random_state=args.random_state,
                **model_configuration,
            )
            model.fit(x_train_full, train["entity_id"].to_numpy(), train["time"].to_numpy())
            scores = model.decision_function(x_test, test["entity_id"].to_numpy(), test["time"].to_numpy())
            timings[name] = time.perf_counter() - start
            results[name] = evaluate_by_type(test, scores, args.include_vus, args.vus_window)

    if not args.skip_legacy_dynsphere:
        print("Running DynSphere...", flush=True)
        start = time.perf_counter()
        pendulum = MultiConceptPendulumDetector(
            n_concepts=4,
            latent_dim=5,
            random_state=args.random_state,
        )
        # Sequence models require complete entity trajectories and therefore use
        # all normal training rows rather than the PyOD subsample.
        pendulum.fit(x_train_full, train["entity_id"].to_numpy(), train["time"].to_numpy())
        pendulum_scores = pendulum.decision_function(
            x_test,
            test["entity_id"].to_numpy(),
            test["time"].to_numpy(),
        )
        timings["DynSphere"] = time.perf_counter() - start
        results["DynSphere"] = evaluate_by_type(test, pendulum_scores, args.include_vus, args.vus_window)

        ablations = {
            "DynSphere-NoTransition": {"transition_weight": 0.0},
            "DynSphere-NoEnergy": {"energy_weight": 0.0},
            "DynSphere-NoRawDynamics": {"raw_dynamic_weight": 0.0},
            "DynSphere-NoConcepts": {"n_concepts": 1, "transition_weight": 0.0},
        }
        for name, kwargs in ablations.items():
            print(f"Running {name}...", flush=True)
            start = time.perf_counter()
            model = MultiConceptPendulumDetector(latent_dim=5, random_state=args.random_state, **kwargs)
            model.fit(x_train_full, train["entity_id"].to_numpy(), train["time"].to_numpy())
            scores = model.decision_function(x_test, test["entity_id"].to_numpy(), test["time"].to_numpy())
            timings[name] = time.perf_counter() - start
            results[name] = evaluate_by_type(test, scores, args.include_vus, args.vus_window)

    print_table(results)

    out = {
        "train": str(Path(args.train)),
        "test": str(Path(args.test)),
        "feature_columns": feature_cols,
        "contamination": args.contamination,
        "max_train_rows_for_pyod": args.max_train_rows,
        "include_neural": args.include_neural,
        "include_manifold": args.include_manifold,
        "include_local_manifold": args.include_local_manifold,
        "include_unified": args.include_unified,
        "include_predictive": args.include_predictive,
        "include_identifiable_predictive": args.include_identifiable_predictive,
        "include_segment_predictive": args.include_segment_predictive,
        "segment_horizons": args.segment_horizons,
        "segment_rollout_epochs": args.segment_rollout_epochs,
        "segment_probabilistic_rollout": args.segment_probabilistic_rollout,
        "include_boundary_change_point": args.include_boundary_change_point,
        "boundary_reset_components": args.boundary_reset_components,
        "include_semi_markov_change_point": args.include_semi_markov_change_point,
        "semi_markov_max_duration": args.semi_markov_max_duration,
        "semi_markov_mean_duration": args.semi_markov_mean_duration,
        "semi_markov_duration_grid": args.semi_markov_duration_grid,
        "semi_markov_boundary_hazard": args.semi_markov_boundary_hazard,
        "skip_legacy_dynsphere": args.skip_legacy_dynsphere,
        "manifold_configuration": {
            "latent_dim": args.manifold_latent_dim,
            "n_modes": args.manifold_modes,
            "ema_alpha": args.manifold_ema_alpha,
            "calibration_fraction": args.manifold_calibration_fraction,
            "local_neighbors": args.manifold_local_neighbors,
        },
        "unified_configuration": {
            "latent_dim": args.unified_latent_dim,
            "hidden_dim": args.unified_hidden_dim,
            "n_modes": args.unified_modes,
            "epochs": args.unified_epochs,
            "sequence_length": args.unified_sequence_length,
        },
        "predictive_configuration": {
            "latent_dim": args.predictive_latent_dim,
            "hidden_dim": args.predictive_hidden_dim,
            "n_modes": args.predictive_modes,
            "epochs": args.predictive_epochs,
            "sequence_length": args.predictive_sequence_length,
        },
        "include_vus": args.include_vus,
        "vus_window": args.vus_window,
        "timings_seconds": timings,
        "results": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
