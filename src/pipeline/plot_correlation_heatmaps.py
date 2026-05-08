from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Matplotlib may try writing cache under ~/.matplotlib which isn't always writable.
_DEFAULT_MPLCONFIGDIR = _project_root() / "outputs" / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(_DEFAULT_MPLCONFIGDIR))
_DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class CorrSpec:
    key: str
    title: str
    filename_suffix: str


SPECS: list[CorrSpec] = [
    CorrSpec(key="pearson_corr", title="Pearson correlation", filename_suffix="pearson"),
    CorrSpec(key="spearman_corr", title="Spearman correlation", filename_suffix="spearman"),
]


def _load_corr_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"x", "y", "candidate", "pearson_corr", "spearman_corr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def _pivot(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return (
        df.pivot_table(index="x", columns="y", values=value_col, aggfunc="mean")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )


def _heatmap(
    mat: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    annotate: bool,
    vmin: float,
    vmax: float,
) -> None:
    if mat.empty:
        return

    fig_w = max(6.0, 1.6 * mat.shape[1])
    fig_h = max(4.0, 0.9 * mat.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)

    im = ax.imshow(mat.to_numpy(), cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(mat.shape[1]), labels=list(mat.columns), rotation=35, ha="right")
    ax.set_yticks(range(mat.shape[0]), labels=list(mat.index))
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, shrink=0.95)
    cbar.set_label("Correlation")

    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.iat[i, j]
                if pd.isna(val):
                    continue
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color="black")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    root = _project_root()
    p = argparse.ArgumentParser(
        description=(
            "Create correlation heatmaps (Pearson + Spearman) from "
            "full_subreddit_analysis/**/correlations/*_correlations.csv."
        )
    )
    p.add_argument(
        "--root",
        type=Path,
        default=root / "full_subreddit_analysis",
        help="Root folder containing per-subreddit analysis folders.",
    )
    p.add_argument(
        "--out-subdir",
        default="heatmaps",
        help="Subfolder under each /correlations directory to write images into.",
    )
    p.add_argument("--annotate", action="store_true", help="Overlay numeric values in each cell.")
    p.add_argument("--limit", type=int, default=0, help="If >0, only process up to this many CSVs.")
    p.add_argument("--vmin", type=float, default=-1.0, help="Minimum colormap value.")
    p.add_argument("--vmax", type=float, default=1.0, help="Maximum colormap value.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()

    csvs = sorted(root.glob("**/correlations/*_correlations.csv"))
    if args.limit and args.limit > 0:
        csvs = csvs[: args.limit]

    processed = 0
    for csv_path in csvs:
        df = _load_corr_csv(csv_path)
        candidate = str(df["candidate"].iloc[0])
        out_dir = csv_path.parent / args.out_subdir

        for spec in SPECS:
            mat = _pivot(df, spec.key)
            title = f"{candidate}: {spec.title}"
            out_path = out_dir / f"{csv_path.stem}_{spec.filename_suffix}_heatmap.png"
            _heatmap(
                mat,
                title=title,
                out_path=out_path,
                annotate=args.annotate,
                vmin=float(args.vmin),
                vmax=float(args.vmax),
            )

        processed += 1

    print(f"Processed {processed} correlation CSV(s).")


if __name__ == "__main__":
    main()

