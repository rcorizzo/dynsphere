"""Deterministic adapter fixture used to test process isolation."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="success")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    received = {
        "input_dir": str(args.input_dir),
        "files": sorted(path.name for path in args.input_dir.iterdir()),
        "seed": args.seed,
        "configuration": json.loads(args.config_json.read_text(encoding="utf-8")),
        "environment": {
            name: os.environ.get(name, "")
            for name in ("DGLBACKEND", "DGLDEFAULTDIR", "MPLCONFIGDIR", "XDG_CACHE_HOME")
        },
    }
    (args.output_dir / "received.json").write_text(
        json.dumps(received, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if args.mode == "timeout":
        time.sleep(2.0)
    if args.mode == "nonzero":
        raise SystemExit(7)

    test = np.load(args.input_dir / "test.npy", allow_pickle=False)
    scores = np.arange(len(test), dtype=float) + args.seed
    if args.mode == "truncated":
        scores = scores[:-1]
    elif args.mode == "nan":
        scores[0] = np.nan
    if args.mode != "missing":
        np.save(args.output_dir / "scores.npy", scores, allow_pickle=False)

    adapter_record = {"method_id": "fake"}
    if args.mode != "partial":
        adapter_record["status"] = "complete"
    (args.output_dir / "run.json").write_text(
        json.dumps(adapter_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
