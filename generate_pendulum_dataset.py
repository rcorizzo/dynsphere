#!/usr/bin/env python3
"""Generate a multivariate anomaly-detection benchmark inspired by spherical pendula.

The dataset is intentionally hard for static density/distance baselines:
- several normal sub-concepts coexist at the same timestamp across entities;
- many anomalies are dynamically invalid but lie in high-density normal regions;
- observations are nonlinear noisy projections of latent spherical states.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Regime:
    name: str
    theta0: float
    theta_amp: float
    omega_theta: float
    phi_rate: float
    phi_amp: float
    phase: float
    noise: float


REGIMES = [
    Regime("small_libration", 0.55, 0.12, 0.070, 0.018, 0.05, 0.2, 0.010),
    Regime("wide_libration", 0.92, 0.22, 0.045, 0.027, 0.09, 1.3, 0.012),
    Regime("precessing_high_energy", 1.23, 0.10, 0.038, 0.052, 0.13, 2.2, 0.013),
    Regime("near_resonant", 0.76, 0.18, 0.083, -0.032, 0.16, 3.0, 0.011),
]

ANOMALY_TYPES = [
    "none",
    "sensor_spike",
    "latent_teleport",
    "phase_lag",
    "velocity_inversion",
    "energy_burst",
    "cross_concept_transition",
]

DISTRIBUTION_PRESERVING_ANOMALY_TYPES = [
    "segment_reversal",
    "cyclic_phase_displacement",
    "matched_splice",
    "invalid_mode_transition",
]


def wrap_pi(x: np.ndarray | float) -> np.ndarray | float:
    return (x + math.pi) % (2.0 * math.pi) - math.pi


def latent_state(t: np.ndarray, regime_id: int, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    r = REGIMES[regime_id]
    eps = rng.normal(0.0, r.noise, size=t.shape)
    theta = r.theta0 + r.theta_amp * np.sin(r.omega_theta * t + r.phase)
    theta += 0.035 * np.sin(0.013 * t + 0.7 * r.phase) + eps
    theta = np.clip(theta, 0.08, math.pi - 0.08)

    phi = r.phi_rate * t + r.phi_amp * np.sin(0.5 * r.omega_theta * t + r.phase)
    phi += 0.012 * np.sin(0.009 * t + regime_id)
    phi = wrap_pi(phi)

    theta_dot = np.gradient(theta, t)
    phi_unwrapped = np.unwrap(phi)
    phi_dot = np.gradient(phi_unwrapped, t)

    theta_ddot = np.gradient(theta_dot, t)
    phi_ddot = np.gradient(phi_dot, t)
    return theta, phi, theta_dot, phi_dot, theta_ddot, phi_ddot


def spherical_cartesian(theta: np.ndarray, phi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx = np.sin(theta) * np.cos(phi)
    sy = np.sin(theta) * np.sin(phi)
    sz = np.cos(theta)
    return sx, sy, sz


def energy(theta: np.ndarray, theta_dot: np.ndarray, phi_dot: np.ndarray) -> np.ndarray:
    kinetic = 0.5 * (theta_dot**2 + (np.sin(theta) ** 2) * phi_dot**2)
    potential = 1.0 - np.cos(theta)
    return kinetic + potential


def observation_features(
    theta: np.ndarray,
    phi: np.ndarray,
    theta_dot: np.ndarray,
    phi_dot: np.ndarray,
    theta_ddot: np.ndarray,
    phi_ddot: np.ndarray,
    entity_bias: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = 0.018,
) -> np.ndarray:
    sx, sy, sz = spherical_cartesian(theta, phi)
    e = energy(theta, theta_dot, phi_dot)

    raw = np.column_stack(
        [
            sx,
            sy,
            sz,
            theta_dot,
            phi_dot,
            theta_ddot,
            phi_ddot,
            np.sin(2.0 * theta),
            np.cos(2.0 * phi),
            np.sin(theta + phi),
            e,
            theta_dot * phi_dot,
        ]
    )

    nonlinear = np.column_stack(
        [
            raw[:, 0] + 0.22 * raw[:, 7],
            raw[:, 1] - 0.18 * raw[:, 9],
            raw[:, 2] + 0.12 * raw[:, 8],
            1.7 * raw[:, 3] + 0.10 * raw[:, 0],
            1.5 * raw[:, 4] + 0.08 * raw[:, 1],
            2.4 * raw[:, 5],
            2.2 * raw[:, 6],
            raw[:, 7],
            raw[:, 8],
            raw[:, 9],
            raw[:, 10],
            3.0 * raw[:, 11],
        ]
    )
    noise = rng.normal(0.0, noise_std, size=nonlinear.shape)
    return nonlinear + entity_bias + noise


def smooth_regime_schedule(
    n_steps: int,
    entity_id: int,
    n_regimes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    schedule = np.empty(n_steps, dtype=int)
    current = entity_id % n_regimes
    t = 0
    while t < n_steps:
        duration = int(rng.integers(260, 620))
        schedule[t : min(n_steps, t + duration)] = current
        if rng.random() < 0.72:
            current = int(rng.integers(0, n_regimes))
        else:
            current = (current + 1) % n_regimes
        t += duration
    return schedule


def blend_segments(values: list[np.ndarray], schedule: np.ndarray, width: int = 35) -> np.ndarray:
    out = np.choose(schedule, values)
    change_points = np.flatnonzero(schedule[1:] != schedule[:-1]) + 1
    for cp in change_points:
        lo = max(0, cp - width)
        hi = min(len(schedule), cp + width)
        if hi <= lo:
            continue
        left_regime = schedule[cp - 1]
        right_regime = schedule[cp]
        alpha = np.linspace(0.0, 1.0, hi - lo)
        smooth = 0.5 - 0.5 * np.cos(math.pi * alpha)
        out[lo:hi] = (1.0 - smooth) * values[left_regime][lo:hi] + smooth * values[right_regime][lo:hi]
    return out


def inject_anomalies(
    theta: np.ndarray,
    phi: np.ndarray,
    theta_dot: np.ndarray,
    phi_dot: np.ndarray,
    theta_ddot: np.ndarray,
    phi_ddot: np.ndarray,
    schedule: np.ndarray,
    rng: np.random.Generator,
    anomaly_fraction: float,
    allowed_anomaly_types: list[int] | None = None,
    anomaly_type_weights: list[float] | None = None,
    times: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    n = len(theta)
    if times is None:
        times = np.arange(n, dtype=float)
    labels = np.zeros(n, dtype=int)
    anomaly_names = np.array(["none"] * n, dtype=object)
    target = max(1, int(n * anomaly_fraction))
    occupied = np.zeros(n, dtype=bool)

    def free_start(length: int) -> int:
        for _ in range(500):
            start = int(rng.integers(80, n - length - 80))
            if not occupied[start - 8 : start + length + 8].any():
                return start
        return int(rng.integers(80, n - length - 80))

    choices = allowed_anomaly_types or list(range(1, len(ANOMALY_TYPES)))
    choice_weights = None
    if anomaly_type_weights is not None:
        choice_weights = np.asarray(anomaly_type_weights, dtype=float)
        if len(choice_weights) != len(choices):
            raise ValueError("anomaly_type_weights must match allowed_anomaly_types.")
        choice_weights = choice_weights / choice_weights.sum()
    while labels.sum() < target:
        anomaly_type = int(rng.choice(choices, p=choice_weights))
        length = int(rng.integers(6, 34))
        start = free_start(length)
        end = start + length
        occupied[start:end] = True
        labels[start:end] = 1
        anomaly_names[start:end] = ANOMALY_TYPES[anomaly_type]

        if anomaly_type == 1:
            # Point anomaly: obvious sensor corruption, included as an easy sanity check.
            phi[start:end] = wrap_pi(phi[start:end] + rng.normal(0.55, 0.06, size=length))
            theta[start:end] = np.clip(theta[start:end] + rng.normal(0.28, 0.04, size=length), 0.08, math.pi - 0.08)
        elif anomaly_type == 2:
            # Jump to another normal location: endpoints are normal-looking, transition is not.
            source = int(rng.integers(80, n - length - 80))
            theta[start:end] = theta[source : source + length]
            phi[start:end] = phi[source : source + length]
            theta_dot[start:end] = theta_dot[source : source + length]
            phi_dot[start:end] = phi_dot[source : source + length]
            theta_ddot[start:end] = theta_ddot[source : source + length]
            phi_ddot[start:end] = phi_ddot[source : source + length]
        elif anomaly_type == 3:
            # Phase-lagged normal state from the same concept.
            lag = int(rng.integers(90, 190))
            source = np.clip(np.arange(start, end) - lag, 0, n - 1)
            theta[start:end] = theta[source]
            phi[start:end] = phi[source]
            theta_dot[start:end] = theta_dot[source]
            phi_dot[start:end] = phi_dot[source]
            theta_ddot[start:end] = theta_ddot[source]
            phi_ddot[start:end] = phi_ddot[source]
        elif anomaly_type == 4:
            # Same position but impossible velocity sign.
            theta_dot[start:end] *= -1.0
            phi_dot[start:end] *= -1.0
            theta_ddot[start:end] *= -1.0
            phi_ddot[start:end] *= -1.0
        elif anomaly_type == 5:
            # High kinetic energy while staying near a common angular region.
            theta_dot[start:end] += rng.normal(0.16, 0.025, size=length) * rng.choice([-1.0, 1.0])
            phi_dot[start:end] += rng.normal(0.12, 0.025, size=length) * rng.choice([-1.0, 1.0])
            window = slice(max(0, start - 1), min(n, end + 1))
            theta_ddot[start:end] = np.gradient(theta_dot[window], times[window])[1 : length + 1]
            phi_ddot[start:end] = np.gradient(phi_dot[window], times[window])[1 : length + 1]
        elif anomaly_type == 6:
            # Segment copied from a different normal concept, hard for static density methods.
            candidates = np.flatnonzero(schedule != schedule[start])
            if len(candidates) > length:
                source = int(rng.choice(candidates[:-length]))
                theta[start:end] = theta[source : source + length]
                phi[start:end] = phi[source : source + length]
                theta_dot[start:end] = theta_dot[source : source + length]
                phi_dot[start:end] = phi_dot[source : source + length]
                theta_ddot[start:end] = theta_ddot[source : source + length]
                phi_ddot[start:end] = phi_ddot[source : source + length]

    return theta, phi, theta_dot, phi_dot, theta_ddot, phi_ddot, labels, anomaly_names


def distribution_preserving_index_map(
    features: np.ndarray,
    schedule: np.ndarray,
    rng: np.random.Generator,
    anomaly_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct trajectory anomalies using only rows from the clean sequence.

    The returned index map can be applied to both observed and latent arrays.
    Consequently, every anomalous observation is exactly a clean observation;
    only its temporal placement or ordering is abnormal.  A pointwise detector
    therefore cannot exploit marginal support violations introduced by the
    generator.
    """

    n = len(features)
    if n != len(schedule):
        raise ValueError("features and schedule must have equal length.")
    if n < 24:
        raise ValueError("At least 24 observations are required for distribution-preserving anomalies.")
    if not 0.0 <= anomaly_fraction < 1.0:
        raise ValueError("anomaly_fraction must be in [0, 1).")

    index_map = np.arange(n)
    labels = np.zeros(n, dtype=int)
    anomaly_names = np.array(["none"] * n, dtype=object)
    occupied = np.zeros(n, dtype=bool)
    target = max(1, int(n * anomaly_fraction))
    max_length = max(6, min(32, n // 8))
    margin = min(40, max(4, n // 10))

    def available_start(length: int) -> int | None:
        valid = []
        for start in range(margin, n - length - margin + 1):
            lo = max(0, start - 5)
            hi = min(n, start + length + 5)
            if not occupied[lo:hi].any():
                valid.append(start)
        if not valid:
            return None
        return int(rng.choice(valid))

    def candidate_starts(start: int, length: int, same_regime: bool) -> np.ndarray:
        target_regime = schedule[start]
        candidates: list[int] = []
        for source in range(1, n - length):
            if max(source, start) < min(source + length, start + length):
                continue
            source_regimes = schedule[source : source + length]
            if same_regime:
                valid_regime = np.all(source_regimes == target_regime)
            else:
                valid_regime = np.all(source_regimes != target_regime)
            if valid_regime:
                candidates.append(source)
        return np.asarray(candidates, dtype=int)

    attempts = 0
    while labels.sum() < target and attempts < 10 * n:
        attempts += 1
        remaining = target - int(labels.sum())
        length = int(rng.integers(6, max_length + 1))
        if remaining < length:
            # A two-row segment is the minimum meaningful temporal
            # transformation.  This keeps the realized anomaly fraction close
            # to its target without reintroducing the one-row cyclic-shift bug.
            length = max(2, remaining)
        start = available_start(length)
        if start is None:
            break
        end = start + length
        mechanism = str(rng.choice(DISTRIBUTION_PRESERVING_ANOMALY_TYPES))
        source_indices: np.ndarray | None = None

        if mechanism == "segment_reversal":
            source_indices = np.arange(start, end)[::-1]
        elif mechanism == "cyclic_phase_displacement":
            shift = int(rng.integers(1, length))
            source_indices = np.roll(np.arange(start, end), shift)
        elif mechanism in {"matched_splice", "invalid_mode_transition"}:
            candidates = candidate_starts(
                start,
                length,
                same_regime=mechanism == "matched_splice",
            )
            if len(candidates):
                # Select a boundary-matched source.  This makes the first
                # anomalous sample locally plausible while the subsequent
                # evolution or mode transition remains dynamically invalid.
                boundary = features[start - 1]
                distances = np.linalg.norm(features[candidates] - boundary, axis=1)
                shortlist = np.argsort(distances)[: min(8, len(candidates))]
                source = int(candidates[int(rng.choice(shortlist))])
                source_indices = np.arange(source, source + length)

        if source_indices is None:
            continue
        index_map[start:end] = source_indices
        occupied[start:end] = True
        labels[start:end] = 1
        anomaly_names[start:end] = mechanism

    if labels.sum() == 0:
        raise RuntimeError("Unable to place any distribution-preserving anomaly segments.")
    return index_map, labels, anomaly_names


def generate_split(
    n_entities: int,
    n_steps: int,
    seed: int,
    anomaly_fraction: float,
    include_anomalies: bool,
    allowed_anomaly_types: list[int] | None = None,
    anomaly_type_weights: list[float] | None = None,
    distribution_preserving: bool = False,
    observation_noise_std: float = 0.018,
    sampling_interval: float = 1.0,
    sampling_jitter: float = 0.0,
) -> list[dict[str, object]]:
    if observation_noise_std < 0.0:
        raise ValueError("observation_noise_std must be non-negative.")
    if sampling_interval <= 0.0:
        raise ValueError("sampling_interval must be positive.")
    if not 0.0 <= sampling_jitter < 1.0:
        raise ValueError("sampling_jitter must be in [0, 1).")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    if sampling_jitter > 0.0:
        # Timestamp perturbations use a separate deterministic stream so that
        # schedules, observation noise, and anomaly mechanisms remain paired
        # with the regularly sampled condition for the same seed.
        time_rng = np.random.default_rng(seed + 10_000_003)
        increments = sampling_interval * (
            1.0 + time_rng.uniform(-sampling_jitter, sampling_jitter, size=n_steps - 1)
        )
        time = np.concatenate([[0.0], np.cumsum(increments)])
    else:
        time = np.arange(n_steps, dtype=float) * sampling_interval

    latent_by_regime = [latent_state(time, rid, rng) for rid in range(len(REGIMES))]

    for entity_id in range(n_entities):
        schedule = smooth_regime_schedule(n_steps, entity_id, len(REGIMES), rng)
        entity_bias = rng.normal(0.0, 0.025, size=(1, 12))

        theta = blend_segments([x[0] for x in latent_by_regime], schedule)
        phi_unwrapped = blend_segments([np.unwrap(x[1]) for x in latent_by_regime], schedule)
        phi = wrap_pi(phi_unwrapped)
        theta_dot = np.gradient(theta, time)
        phi_dot = np.gradient(phi_unwrapped, time)
        theta_ddot = np.gradient(theta_dot, time)
        phi_ddot = np.gradient(phi_dot, time)

        if include_anomalies and not distribution_preserving:
            anomaly_pack = inject_anomalies(
                theta,
                phi,
                theta_dot,
                phi_dot,
                theta_ddot,
                phi_ddot,
                schedule,
                rng,
                anomaly_fraction,
                allowed_anomaly_types,
                anomaly_type_weights,
                times=time,
            )
            theta, phi, theta_dot, phi_dot, theta_ddot, phi_ddot, labels, anomaly_names = anomaly_pack
        else:
            labels = np.zeros(n_steps, dtype=int)
            anomaly_names = np.array(["none"] * n_steps, dtype=object)

        features = observation_features(
            theta,
            phi,
            theta_dot,
            phi_dot,
            theta_ddot,
            phi_ddot,
            entity_bias,
            rng,
            noise_std=observation_noise_std,
        )
        if include_anomalies and distribution_preserving:
            index_map, labels, anomaly_names = distribution_preserving_index_map(
                features,
                schedule,
                rng,
                anomaly_fraction,
            )
            theta = theta[index_map]
            phi = phi[index_map]
            theta_dot = theta_dot[index_map]
            phi_dot = phi_dot[index_map]
            theta_ddot = theta_ddot[index_map]
            phi_ddot = phi_ddot[index_map]
            schedule = schedule[index_map]
            features = features[index_map]
        for t in range(n_steps):
            row: dict[str, object] = {
                "time": float(time[t]),
                "entity_id": entity_id,
                "regime_id": int(schedule[t]),
                "regime_name": REGIMES[int(schedule[t])].name,
                "is_anomaly": int(labels[t]),
                "anomaly_type": str(anomaly_names[t]),
                "theta": float(theta[t]),
                "phi": float(phi[t]),
                "theta_dot": float(theta_dot[t]),
                "phi_dot": float(phi_dot[t]),
                "energy": float(energy(theta[t : t + 1], theta_dot[t : t + 1], phi_dot[t : t + 1])[0]),
            }
            for j in range(features.shape[1]):
                row[f"x_{j:02d}"] = float(features[t, j])
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for row in rows:
        name = str(row["anomaly_type"])
        counts[name] = counts.get(name, 0) + 1
    return {
        "rows": len(rows),
        "anomaly_rows": sum(int(row["is_anomaly"]) for row in rows),
        "anomaly_fraction": sum(int(row["is_anomaly"]) for row in rows) / len(rows),
        "anomaly_type_counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--name", default="pendulum_multiconcept")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--train-entities", type=int, default=24)
    parser.add_argument("--test-entities", type=int, default=12)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--test-anomaly-fraction", type=float, default=0.035)
    parser.add_argument("--observation-noise-std", type=float, default=0.018)
    parser.add_argument(
        "--sampling-interval",
        type=float,
        default=1.0,
        help="Mean physical interval between observations.",
    )
    parser.add_argument(
        "--sampling-jitter",
        type=float,
        default=0.0,
        help="Uniform relative timestamp jitter in [0, 1); zero gives regular sampling.",
    )
    parser.add_argument(
        "--anomaly-profile",
        choices=["mixed", "contextual", "heterogeneous", "ensemble_stress", "distribution_preserving"],
        default="mixed",
        help="mixed keeps all anomaly types; contextual removes easy sensor/energy anomalies; heterogeneous emphasizes mixed local, static, and dynamic mechanisms; ensemble_stress emphasizes anomalies where score-fusion components should be complementary; distribution_preserving creates anomalous trajectories exclusively from clean observed rows.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    allowed_anomaly_types = None
    anomaly_type_weights = None
    description = "Multivariate spherical-pendulum-inspired benchmark with coexisting normal sub-concepts and dynamically subtle anomalies."
    if args.anomaly_profile == "contextual":
        allowed_anomaly_types = [
            ANOMALY_TYPES.index("latent_teleport"),
            ANOMALY_TYPES.index("phase_lag"),
            ANOMALY_TYPES.index("velocity_inversion"),
            ANOMALY_TYPES.index("cross_concept_transition"),
        ]
        description = (
            "Contextual-only spherical-pendulum benchmark focused on dynamically invalid but locally plausible anomalies."
        )
    elif args.anomaly_profile == "heterogeneous":
        allowed_anomaly_types = [
            ANOMALY_TYPES.index("sensor_spike"),
            ANOMALY_TYPES.index("latent_teleport"),
            ANOMALY_TYPES.index("phase_lag"),
            ANOMALY_TYPES.index("velocity_inversion"),
            ANOMALY_TYPES.index("energy_burst"),
            ANOMALY_TYPES.index("cross_concept_transition"),
        ]
        anomaly_type_weights = [0.18, 0.15, 0.12, 0.17, 0.28, 0.10]
        description = (
            "Heterogeneous spherical-pendulum stress benchmark with coexisting pointwise, local-neighborhood, energy, and trajectory-context anomalies."
        )
    elif args.anomaly_profile == "ensemble_stress":
        allowed_anomaly_types = [
            ANOMALY_TYPES.index("sensor_spike"),
            ANOMALY_TYPES.index("phase_lag"),
            ANOMALY_TYPES.index("velocity_inversion"),
            ANOMALY_TYPES.index("energy_burst"),
        ]
        anomaly_type_weights = [0.10, 0.15, 0.10, 0.65]
        description = (
            "Synthetic ensemble-stress benchmark dominated by short energy/local perturbations and a minority of trajectory-context anomalies."
        )
    elif args.anomaly_profile == "distribution_preserving":
        description = (
            "Distribution-preserving trajectory benchmark in which anomalous segments contain only clean observed rows, reordered or spliced into dynamically invalid contexts."
        )

    train = generate_split(
        args.train_entities,
        args.steps,
        args.seed,
        0.0,
        include_anomalies=False,
        observation_noise_std=args.observation_noise_std,
        sampling_interval=args.sampling_interval,
        sampling_jitter=args.sampling_jitter,
    )
    test = generate_split(
        args.test_entities,
        args.steps,
        args.seed + 1000,
        args.test_anomaly_fraction,
        include_anomalies=True,
        allowed_anomaly_types=allowed_anomaly_types,
        anomaly_type_weights=anomaly_type_weights,
        distribution_preserving=args.anomaly_profile == "distribution_preserving",
        observation_noise_std=args.observation_noise_std,
        sampling_interval=args.sampling_interval,
        sampling_jitter=args.sampling_jitter,
    )

    train_path = out_dir / f"{args.name}_train.csv"
    test_path = out_dir / f"{args.name}_test.csv"
    meta_path = out_dir / f"{args.name}_metadata.json"

    write_csv(train_path, train)
    write_csv(test_path, test)

    metadata = {
        "description": description,
        "seed": args.seed,
        "observation_noise_std": args.observation_noise_std,
        "sampling_interval": args.sampling_interval,
        "sampling_jitter": args.sampling_jitter,
        "anomaly_profile": args.anomaly_profile,
        "feature_columns": [f"x_{j:02d}" for j in range(12)],
        "label_column": "is_anomaly",
        "anomaly_type_column": "anomaly_type",
        "normal_regimes": [r.__dict__ for r in REGIMES],
        "anomaly_types": ANOMALY_TYPES + DISTRIBUTION_PRESERVING_ANOMALY_TYPES,
        "allowed_anomaly_types": DISTRIBUTION_PRESERVING_ANOMALY_TYPES
        if args.anomaly_profile == "distribution_preserving"
        else (
            [ANOMALY_TYPES[i] for i in allowed_anomaly_types]
            if allowed_anomaly_types is not None
            else ANOMALY_TYPES[1:]
        ),
        "anomaly_type_weights": anomaly_type_weights,
        "train": summarize(train),
        "test": summarize(test),
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"train": str(train_path), "test": str(test_path), "metadata": str(meta_path)}, indent=2))


if __name__ == "__main__":
    main()
