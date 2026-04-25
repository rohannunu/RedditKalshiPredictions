from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _project_root() -> Path:
    # export_hourly_candidates_only.py -> pipeline -> src -> project root
    return Path(__file__).resolve().parents[2]


def _parse_args() -> argparse.Namespace:
    root = _project_root()
    parser = argparse.ArgumentParser(
        description=(
            "Copy only hourly candidate sentiment files (mamdani, cuomo, sliwa) "
            "from outputs/by_subreddit/<sub>/tables into sentiment_hourly_subreddit/<sub>/."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=root / "outputs" / "by_subreddit",
        help="Root folder containing per-subreddit outputs with /tables subfolders.",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=root / "sentiment_hourly_subreddit",
        help="Destination root for exported candidate-hourly files.",
    )
    parser.add_argument(
        "--clean-dest",
        action="store_true",
        help="Delete destination root before exporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_root = args.source_root.resolve()
    dest_root = args.dest_root.resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")

    if args.clean_dest and dest_root.exists():
        shutil.rmtree(dest_root)

    dest_root.mkdir(parents=True, exist_ok=True)

    # Canonical filenames
    candidate_files = [
        "sentiment_hourly_mamdani.csv",
        "sentiment_hourly_cuomo.csv",
        "sentiment_hourly_sliwa.csv",
    ]

    subdirs = sorted(
        d for d in source_root.iterdir()
        if d.is_dir() and (d / "tables").exists()
    )

    if not subdirs:
        raise FileNotFoundError(f"No subreddit folders with /tables found under: {source_root}")

    copied_total = 0
    missing_total = 0

    for subdir in subdirs:
        subreddit = subdir.name
        tables_dir = subdir / "tables"
        out_dir = dest_root / subreddit
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"=== {subreddit} ===")
        for filename in candidate_files:
            src = tables_dir / filename
            dst = out_dir / filename
            if src.exists():
                shutil.copy2(src, dst)
                copied_total += 1
                print(f"copied:  {src} -> {dst}")
            else:
                missing_total += 1
                print(f"missing: {src}")

    print("\nDone.")
    print(f"Destination root: {dest_root}")
    print(f"Files copied: {copied_total}")
    print(f"Missing files: {missing_total}")


if __name__ == "__main__":
    main()
