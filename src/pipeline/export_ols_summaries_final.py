from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _strip_rank_warning(inp: Path, out: Path) -> None:
    df = pd.read_csv(inp)
    if "rank_warning" in df.columns:
        df = df.drop(columns=["rank_warning"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def _parse_args() -> argparse.Namespace:
    root = _project_root()
    p = argparse.ArgumentParser(
        description=(
            "Drop rank_warning from all_ols_summaries.csv and write OLS_summaries_final.csv. "
            "By default, processes every full_subreddit_analysis/*/all_ols_summaries.csv."
        )
    )
    p.add_argument(
        "--scan-root",
        type=Path,
        default=root / "full_subreddit_analysis",
        help="Root folder to search for all_ols_summaries.csv (default: project full_subreddit_analysis).",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Single input file; if set, only this file is processed (--output optional).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Single output path (only with --input; default: <input_dir>/OLS_summaries_final.csv).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.input is not None:
        inp = args.input.resolve()
        if not inp.exists():
            raise FileNotFoundError(f"Input not found: {inp}")
        out = args.output.resolve() if args.output else inp.parent / "OLS_summaries_final.csv"
        _strip_rank_warning(inp, out)
        df = pd.read_csv(out)
        print(f"Wrote {out} ({len(df)} rows, {len(df.columns)} columns)")
        return

    scan_root = args.scan_root.resolve()
    if not scan_root.is_dir():
        raise FileNotFoundError(f"scan-root not found: {scan_root}")

    paths = sorted(scan_root.glob("**/all_ols_summaries.csv"))
    if not paths:
        print(f"No all_ols_summaries.csv under {scan_root}")
        return

    for inp in paths:
        out = inp.parent / "OLS_summaries_final.csv"
        _strip_rank_warning(inp, out)
        df = pd.read_csv(out)
        print(f"Wrote {out} ({len(df)} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
