#!/usr/bin/env python3
"""Aggregate Strategy A TSB-AD-M results by source family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def source_from_dataset(dataset: str) -> str:
    for source in ("MSL", "SMAP", "SMD"):
        if f"_{source}_" in dataset:
            return source
    return "Other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raw = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    ok = raw[raw["status"] == "ok"].copy()
    ok["source"] = ok["dataset"].astype(str).map(source_from_dataset)
    # A later input can provide a clean-prefix replacement for a failed file.
    ok = ok.drop_duplicates(subset=["file", "method_id"], keep="last")
    metrics = ["auroc", "auprc", "pa_f1", "pauc_fpr_05", "event_recall_top_1"]
    metrics = [metric for metric in metrics if metric in ok]
    summary = ok.groupby(["source", "method_id", "method"])[metrics].agg(["count", "mean", "std"])
    summary = summary.reset_index()
    summary.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    payload = {
        "inputs": [str(path) for path in args.inputs],
        "summary": summary.where(pd.notnull(summary), None).to_dict(orient="records"),
    }
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(f"Wrote {args.out} and {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
