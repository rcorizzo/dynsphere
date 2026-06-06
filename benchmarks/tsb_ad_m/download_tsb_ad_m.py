#!/usr/bin/env python3
"""Download and extract TSB-AD-M into this benchmark folder."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_URL = "https://www.thedatum.org/datasets/TSB-AD-M.zip"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


def download_with_urllib(url: str, zip_path: Path) -> None:
    request = Request(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/zip,application/octet-stream,*/*",
            "Referer": "https://github.com/TheDatumOrg/TSB-AD",
        },
    )
    tmp_path = zip_path.with_suffix(".zip.part")
    with urlopen(request, timeout=120) as response, tmp_path.open("wb") as f:
        shutil.copyfileobj(response, f)
    tmp_path.replace(zip_path)


def download_with_curl(url: str, zip_path: Path) -> None:
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is not available")
    tmp_path = zip_path.with_suffix(".zip.part")
    subprocess.run(
        [
            curl,
            "-L",
            "--fail",
            "--retry",
            "3",
            "--connect-timeout",
            "30",
            "-A",
            DEFAULT_USER_AGENT,
            "-e",
            "https://github.com/TheDatumOrg/TSB-AD",
            "-o",
            str(tmp_path),
            url,
        ],
        check=True,
    )
    tmp_path.replace(zip_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out-dir", default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--method", choices=["auto", "urllib", "curl"], default="auto")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    extract_dir = out_dir / "TSB-AD-M"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "TSB-AD-M.zip"

    if zip_path.exists() and not args.force:
        print(f"Using existing {zip_path}")
    else:
        print(f"Downloading {args.url} -> {zip_path}")
        try:
            if args.method == "curl":
                download_with_curl(args.url, zip_path)
            else:
                download_with_urllib(args.url, zip_path)
        except (HTTPError, URLError, TimeoutError, RuntimeError, subprocess.CalledProcessError) as exc:
            if args.method == "auto":
                print(f"urllib download failed: {exc}", file=sys.stderr)
                print("Retrying with curl...", file=sys.stderr)
                try:
                    download_with_curl(args.url, zip_path)
                except (RuntimeError, subprocess.CalledProcessError) as curl_exc:
                    raise SystemExit(
                        "Download failed. The official TSB-AD host sometimes blocks scripted downloads.\n"
                        "Try one of these commands from the repository root:\n"
                        "  python3 benchmarks/tsb_ad_m/download_tsb_ad_m.py --method curl --force\n"
                        "  curl -L -A 'Mozilla/5.0' -e https://github.com/TheDatumOrg/TSB-AD "
                        "-o benchmarks/tsb_ad_m/data/raw/TSB-AD-M.zip "
                        "https://www.thedatum.org/datasets/TSB-AD-M.zip\n"
                        "Then rerun:\n"
                        "  python3 benchmarks/tsb_ad_m/download_tsb_ad_m.py\n"
                        f"Last curl error: {curl_exc}"
                    ) from curl_exc
            else:
                raise SystemExit(
                    "Download failed. The official TSB-AD host sometimes blocks scripted downloads.\n"
                    "Try:\n"
                    "  python3 benchmarks/tsb_ad_m/download_tsb_ad_m.py --method curl --force\n"
                    f"Original error: {exc}"
                ) from exc

    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {zip_path} -> {extract_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    csv_count = len(list(extract_dir.rglob("*.csv")))
    print(f"Done. Found {csv_count} CSV files under {extract_dir}")


if __name__ == "__main__":
    main()
