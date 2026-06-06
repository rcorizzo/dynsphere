#!/usr/bin/env python3
"""Download real-world datasets used by the DynSphere experiments.

The script only downloads public source files into the expected local folder
layout. The repository intentionally does not redistribute these third-party
raw files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parent

URLS = {
    "batadal": [
        (
            "https://www.batadal.net/data/BATADAL_dataset03.csv",
            ROOT / "data" / "batadal" / "BATADAL_dataset03.csv",
        ),
        (
            "https://www.batadal.net/data/BATADAL_dataset04.csv",
            ROOT / "data" / "batadal" / "BATADAL_dataset04.csv",
        ),
    ],
    "hai_20_07": [
        (
            "https://raw.githubusercontent.com/icsdataset/hai/master/hai-20.07/train1.csv.gz",
            ROOT / "data" / "hai_20_07" / "train1.csv.gz",
        ),
        (
            "https://raw.githubusercontent.com/icsdataset/hai/master/hai-20.07/test1.csv.gz",
            ROOT / "data" / "hai_20_07" / "test1.csv.gz",
        ),
    ],
    "power_system": [
        (
            "https://ece.uah.edu/~thm0009/icsdatasets/PowerSystem_Dataset.zip",
            ROOT / "data" / "power_system" / "PowerSystem_Dataset.zip",
        ),
        (
            "https://ece.uah.edu/~thm0009/icsdatasets/PowerSystem_Dataset_README.pdf",
            ROOT / "data" / "power_system" / "PowerSystem_Dataset_README.pdf",
        ),
    ],
}


def download_one(url: str, out_path: Path, overwrite: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        print(f"exists: {out_path}")
        return
    print(f"download: {url} -> {out_path}")
    urlretrieve(url, out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=sorted(URLS) + ["all"],
        default="all",
        help="Dataset source to download.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    selected = URLS.keys() if args.dataset == "all" else [args.dataset]
    for name in selected:
        print(f"== {name} ==")
        for url, out_path in URLS[name]:
            download_one(url, out_path, args.overwrite)

    if args.dataset in {"power_system", "all"}:
        print(
            "\nPowerSystem requires extracting the downloaded archive and preparing "
            "`data/power_system/binary/combined_binary.csv` before running the "
            "paper command. The raw archive is intentionally ignored by git."
        )


if __name__ == "__main__":
    main()
