#!/usr/bin/env python3
"""Multi-seed robustness experiments for Strategy A conditional dynamics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_pendulum_dataset import generate_split
from manifold_dynamics_detector import ManifoldDynamicsDetector
from range_metrics import practical_range_metrics


SCENARIOS: dict[str, dict[str, float]] = {
    "baseline": {
        "observation_noise_std": 0.018,
        "sampling_interval": 1.0,
        "sampling_jitter": 0.0,
    },
    "noise_2x": {
        "observation_noise_std": 0.036,
        "sampling_interval": 1.0,
        "sampling_jitter": 0.0,
    },
    "noise_4x": {
        "observation_noise_std": 0.072,
        "sampling_interval": 1.0,
        "sampling_jitter": 0.0,
    },
    "interval_2x": {
        "observation_noise_std": 0.018,
        "sampling_interval": 2.0,
        "sampling_jitter": 0.0,
    },
    "irregular_40pct": {
        "observation_noise_std": 0.018,
        "sampling_interval": 1.0,
        "sampling_jitter": 0.4,
    },
}

MODELS: dict[str, dict[str, object]] = {
    "SphereRidge": {"geometry": "sphere", "include_radius": True, "field_model": "ridge"},
    "EuclideanRidge": {"geometry": "euclidean", "include_radius": False, "field_model": "ridge"},
    "SphereLocal": {"geometry": "sphere", "include_radius": True, "field_model": "local"},
    "EuclideanLocal": {"geometry": "euclidean", "include_radius": False, "field_model": "local"},
}


def rows_to_arrays(
    rows: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_columns = sorted(column for column in rows[0] if column.startswith("x_"))
    x = np.asarray([[float(row[column]) for column in feature_columns] for row in rows])
    labels = np.asarray([int(row["is_anomaly"]) for row in rows])
    entities = np.asarray([int(row["entity_id"]) for row in rows])
    times = np.asarray([float(row["time"]) for row in rows])
    return x, labels, entities, times


def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }
    metrics.update(practical_range_metrics(labels, scores))
    return metrics


def aggregate(records: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    keys = ["auroc", "auprc", "pauc_fpr_05", "event_recall_top_1", "event_precision_top_1"]
    for scenario in sorted({str(record["scenario"]) for record in records}):
        for model in sorted({str(record["model"]) for record in records}):
            subset = [
                record for record in records
                if record["scenario"] == scenario and record["model"] == model
            ]
            if not subset:
                continue
            row: dict[str, object] = {
                "scenario": scenario,
                "model": model,
                "seeds": len(subset),
            }
            for key in keys:
                values = np.asarray([float(record[key]) for record in subset if record[key] is not None])
                row[f"{key}_mean"] = float(values.mean()) if len(values) else None
                row[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row["runtime_seconds_mean"] = float(
                np.mean([float(record["runtime_seconds"]) for record in subset])
            )
            output.append(row)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/strategy_a_robustness.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[37, 38, 39])
    parser.add_argument("--scenarios", nargs="+", choices=sorted(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=list(MODELS))
    parser.add_argument("--train-entities", type=int, default=8)
    parser.add_argument("--test-entities", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--anomaly-fraction", type=float, default=0.05)
    parser.add_argument("--latent-dim", type=int, default=5)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument("--local-neighbors", type=int, default=64)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    args = parser.parse_args()

    records: list[dict[str, object]] = []
    total = len(args.seeds) * len(args.scenarios) * len(args.models)
    job = 0
    for seed in args.seeds:
        for scenario_name in args.scenarios:
            scenario = SCENARIOS[scenario_name]
            train_rows = generate_split(
                args.train_entities,
                args.steps,
                seed,
                0.0,
                include_anomalies=False,
                **scenario,
            )
            test_rows = generate_split(
                args.test_entities,
                args.steps,
                seed + 1000,
                args.anomaly_fraction,
                include_anomalies=True,
                distribution_preserving=True,
                **scenario,
            )
            x_train, _, train_entities, train_times = rows_to_arrays(train_rows)
            x_test, labels, test_entities, test_times = rows_to_arrays(test_rows)

            for model_name in args.models:
                job += 1
                print(
                    f"[{job}/{total}] seed={seed} scenario={scenario_name} model={model_name}",
                    flush=True,
                )
                started = time.perf_counter()
                detector = ManifoldDynamicsDetector(
                    latent_dim=args.latent_dim,
                    n_modes=args.modes,
                    local_neighbors=args.local_neighbors,
                    calibration_fraction=args.calibration_fraction,
                    random_state=seed,
                    **MODELS[model_name],
                )
                detector.fit(x_train, train_entities, train_times)
                scores = detector.decision_function(x_test, test_entities, test_times)
                record: dict[str, object] = {
                    "seed": seed,
                    "scenario": scenario_name,
                    "model": model_name,
                    "rows": len(labels),
                    "positives": int(labels.sum()),
                    "runtime_seconds": time.perf_counter() - started,
                }
                record.update(evaluate(labels, scores))
                records.append(record)

    summary = aggregate(records)
    payload = {
        "configuration": {
            "seeds": args.seeds,
            "scenarios": {name: SCENARIOS[name] for name in args.scenarios},
            "models": {name: MODELS[name] for name in args.models},
            "train_entities": args.train_entities,
            "test_entities": args.test_entities,
            "steps": args.steps,
            "anomaly_fraction": args.anomaly_fraction,
            "latent_dim": args.latent_dim,
            "modes": args.modes,
            "local_neighbors": args.local_neighbors,
            "calibration_fraction": args.calibration_fraction,
        },
        "records": records,
        "summary": summary,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(out_path.with_suffix(".csv"), summary)
    print(f"Wrote {out_path} and {out_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
