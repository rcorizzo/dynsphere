"""Aggregate only the terminal records declared by one campaign manifest."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from benchmarks.official_baselines.aggregate import aggregate_runs
from benchmarks.official_baselines.run_expanded_evaluation import DEFAULT_RESULT_ROOT


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=20261011)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        campaign = json.loads(args.campaign_manifest.read_text(encoding="utf-8"))
        campaign_id = str(campaign["campaign_id"])
        output_dir = args.output_dir or (
            args.result_root / "aggregate" / campaign_id[:12]
        )
        artifacts = aggregate_runs(
            args.result_root,
            output_dir,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_replicates=args.bootstrap_replicates,
            campaign_manifest=args.campaign_manifest,
        )
        payload = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(artifacts).items()
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"Campaign aggregation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
