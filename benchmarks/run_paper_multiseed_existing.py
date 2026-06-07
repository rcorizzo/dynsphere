#!/usr/bin/env python3
"""Run the existing paper benchmark methods across multiple seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

METHODS = [
    "iforest",
    "hbos",
    "copod",
    "lof",
    "knn",
    "forecast_mlp",
    "window_knn",
    "static_ae",
    "deep_svdd_kmeans",
    "dynsphere",
    "dynsphere_ae",
    "dynsphere_pp",
]

REAL_WORLD_SPECS = [
    {
        "dataset": "BATADAL",
        "train": "benchmarks/real_world_datasets/data/batadal/BATADAL_dataset03.csv",
        "test": "benchmarks/real_world_datasets/data/batadal/BATADAL_dataset04.csv",
        "train_label": "ATT_FLAG",
        "test_label": "ATT_FLAG",
    },
    {
        "dataset": "HAI-20.07",
        "train": "benchmarks/real_world_datasets/data/hai_20_07/train1.csv.gz",
        "test": "benchmarks/real_world_datasets/data/hai_20_07/test1.csv.gz",
        "train_label": "attack",
        "test_label": "attack",
    },
    {
        "dataset": "PowerSystem",
        "train": "benchmarks/real_world_datasets/data/power_system/binary/combined_binary.csv",
        "test": "benchmarks/real_world_datasets/data/power_system/binary/combined_binary.csv",
        "train_label": "marker",
        "test_label": "marker",
    },
]


def run(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[37, 38, 39])
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper_multiseed_existing"))
    parser.add_argument("--datasets", nargs="+", default=["contextual", "tsb", "real_world"], choices=["contextual", "tsb", "real_world"])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--include-vus", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    for seed in args.seeds:
        if "contextual" in args.datasets:
            out = args.out_dir / f"contextual_seed{seed}.json"
            if not (args.skip_existing and out.exists()):
                cmd = [
                    py,
                    "run_benchmarks.py",
                    "--train",
                    "data/contextual/pendulum_contextual_train.csv",
                    "--test",
                    "data/contextual/pendulum_contextual_test.csv",
                    "--out",
                    str(out),
                    "--random-state",
                    str(seed),
                    "--include-neural",
                ]
                if args.include_vus:
                    cmd.append("--include-vus")
                run(cmd, REPO_ROOT, args.dry_run)

        if "tsb" in args.datasets:
            out_dir = args.out_dir / f"tsb_seed{seed}"
            csv_path = out_dir / "tsb_ad_m_results.csv"
            if not (args.skip_existing and csv_path.exists()):
                cmd = [
                    py,
                    "benchmarks/tsb_ad_m/run_tsb_ad_m.py",
                    "--methods",
                    *args.methods,
                    "--include-sources",
                    "MSL",
                    "SMAP",
                    "SMD",
                    "--limit",
                    "0",
                    "--fit-mode",
                    "all",
                    "--random-state",
                    str(seed),
                    "--out-dir",
                    str(out_dir),
                    "--resume",
                ]
                if args.include_vus:
                    cmd.append("--include-vus")
                run(cmd, REPO_ROOT, args.dry_run)

        if "real_world" in args.datasets:
            real_dir = args.out_dir / "real_world"
            real_dir.mkdir(parents=True, exist_ok=True)
            for spec in REAL_WORLD_SPECS:
                out = real_dir / f"{spec['dataset']}_seed{seed}.csv"
                if args.skip_existing and out.exists():
                    continue
                cmd = [
                    py,
                    "benchmarks/real_world_datasets/run_real_world_datasets.py",
                    "--dataset",
                    spec["dataset"],
                    "--train",
                    spec["train"],
                    "--test",
                    spec["test"],
                    "--train-label",
                    spec["train_label"],
                    "--test-label",
                    spec["test_label"],
                    "--max-train-rows",
                    "20000",
                    "--max-test-rows",
                    "30000",
                    "--methods",
                    *args.methods,
                    "--random-state",
                    str(seed),
                    "--out",
                    str(out),
                ]
                if args.include_vus:
                    cmd.append("--include-vus")
                run(cmd, REPO_ROOT, args.dry_run)


if __name__ == "__main__":
    main()
