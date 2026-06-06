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

from pendulum_detector import MultiConceptPendulumDetector


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
        EnhancedPendulumEnsembleDetector,
        ForecastingMLPDetector,
        HybridPendulumSequenceDetector,
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
        ("PendulumSeqAE", HybridPendulumSequenceDetector(random_state=random_state)),
        ("EnhancedPendulum", EnhancedPendulumEnsembleDetector(random_state=random_state)),
    ]

    for name, model in neural_jobs:
        print(f"Running {name}...", flush=True)
        start = time.perf_counter()
        if name in {"SequenceAE", "WindowKNN", "ForecastMLP"}:
            model.fit(x_train, train["entity_id"].to_numpy())
            scores = model.decision_function(x_test, test["entity_id"].to_numpy())
        elif name in {"PendulumSeqAE", "EnhancedPendulum"}:
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
    parser.add_argument("--include-vus", action="store_true")
    parser.add_argument("--vus-window", type=int, default=None)
    args = parser.parse_args()

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

    print("Running MultiConceptPendulum...", flush=True)
    start = time.perf_counter()
    pendulum = MultiConceptPendulumDetector(
        n_concepts=4,
        latent_dim=5,
        random_state=args.random_state,
    )
    # The pendulum method needs complete sequences, so train on all normal rows.
    x_train_full = train[feature_cols].to_numpy()
    pendulum.fit(x_train_full, train["entity_id"].to_numpy(), train["time"].to_numpy())
    pendulum_scores = pendulum.decision_function(x_test, test["entity_id"].to_numpy(), test["time"].to_numpy())
    timings["MultiConceptPendulum"] = time.perf_counter() - start
    results["MultiConceptPendulum"] = evaluate_by_type(test, pendulum_scores, args.include_vus, args.vus_window)

    ablations = {
        "Pendulum-NoTransition": {"transition_weight": 0.0},
        "Pendulum-NoEnergy": {"energy_weight": 0.0},
        "Pendulum-NoRawDynamics": {"raw_dynamic_weight": 0.0},
        "Pendulum-NoConcepts": {"n_concepts": 1, "transition_weight": 0.0},
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
