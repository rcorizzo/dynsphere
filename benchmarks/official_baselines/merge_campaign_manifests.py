"""Merge component campaign manifests with explicit later-source precedence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from benchmarks.official_baselines.campaign import (
    merge_campaign_documents,
    write_campaign_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("manifests", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        documents = [
            json.loads(path.read_text(encoding="utf-8")) for path in args.manifests
        ]
        merged = merge_campaign_documents(documents)
        path = write_campaign_manifest(args.output, merged)
        print(
            json.dumps(
                {
                    "campaign_id": merged["campaign_id"],
                    "expected_job_count": merged["expected_job_count"],
                    "manifest": str(path),
                    "source_campaign_ids": merged["source_campaign_ids"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(f"Campaign merge failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
