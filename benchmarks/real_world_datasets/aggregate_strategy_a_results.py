#!/usr/bin/env python3
"""Aggregate multi-seed Strategy A real-world result files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/strategy_a_real"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    paths = sorted(args.input_dir.glob("*_seed*.csv"))
    if not paths:
        raise SystemExit(f"No seed CSV files found under {args.input_dir}.")
    frames = []
    for path in paths:
        match = re.search(r"_seed(\d+)\.csv$", path.name)
        if match is None:
            continue
        frame = pd.read_csv(path)
        frame["seed"] = int(match.group(1))
        frame["source_file"] = str(path)
        frames.append(frame)
    raw = pd.concat(frames, ignore_index=True)
    ok = raw[raw["status"] == "ok"].copy()
    metrics = [
        column
        for column in [
            "auroc",
            "auprc",
            "plain_f1",
            "pa_f1",
            "pauc_fpr_05",
            "event_recall_top_1",
            "event_precision_top_1",
            "seconds",
        ]
        if column in ok
    ]
    summary = ok.groupby(["dataset", "method_id", "method"])[metrics].agg(["mean", "std"])
    summary = summary.reset_index()
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    out_path = args.out or args.input_dir / "strategy_a_real_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    payload = {
        "input_files": [str(path) for path in paths],
        "successful_runs": int(len(ok)),
        "failed_runs": int(len(raw) - len(ok)),
        "summary": summary.where(pd.notnull(summary), None).to_dict(orient="records"),
    }
    out_path.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"Wrote {out_path} and {out_path.with_suffix('.json')}")


if __name__ == "__main__":
    main()
