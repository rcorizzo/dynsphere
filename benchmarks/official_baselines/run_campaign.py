"""Run a named official-baseline campaign locally or as one scheduler shard."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

from benchmarks.official_baselines.campaign import (
    PROFILES,
    campaign_document,
    partition_jobs,
    resolve_profile,
    write_campaign_manifest,
)
from benchmarks.official_baselines.run_expanded_evaluation import (
    DEFAULT_MANIFEST,
    DEFAULT_REGISTRY,
    DEFAULT_RESULT_ROOT,
    DEFAULT_TSB_ROOT,
    RunConfig,
    build_matrix,
    execute_job,
    matrix_summary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--tsb-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tsb-data-root", type=Path, default=DEFAULT_TSB_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--campaign-manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        profile = resolve_profile(args.profile)
        config = RunConfig(
            methods=tuple(args.methods) if args.methods else profile.methods,
            datasets=tuple(args.datasets) if args.datasets else profile.datasets,
            seeds=tuple(args.seeds) if args.seeds else profile.seeds,
            tsb_manifest=args.tsb_manifest,
            tsb_data_root=args.tsb_data_root,
            result_root=args.result_root,
            registry_path=args.registry,
            timeout_seconds=args.timeout_seconds or profile.timeout_seconds,
            resume=args.resume,
            dry_run=args.dry_run,
        )
        matrix = build_matrix(config)
        selected = partition_jobs(matrix, args.shard_index, args.shard_count)
        document = campaign_document(args.profile, matrix, args.shard_count)
        manifest_path = args.campaign_manifest or (
            args.result_root / "campaigns" / f"{args.profile}.json"
        )
        write_campaign_manifest(manifest_path, document)
        summary = {
            "campaign_id": document["campaign_id"],
            "campaign_manifest": str(manifest_path),
            "profile": args.profile,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "selected_job_count": len(selected),
            "full_matrix": matrix_summary(matrix),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        if args.dry_run:
            return 0
        outcomes = Counter(str(execute_job(job)["status"]) for job in selected)
        print(json.dumps({"outcomes": dict(sorted(outcomes.items()))}, indent=2, sort_keys=True))
        return 0 if not {"failed", "timeout", "incompatible"}.intersection(outcomes) else 1
    except Exception as exc:
        print(f"Campaign validation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
