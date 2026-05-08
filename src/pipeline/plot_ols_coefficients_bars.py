from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Ensure matplotlib is usable in sandbox/non-writable $HOME contexts.
_MPLCONFIGDIR = _project_root() / "outputs" / ".mplconfig"
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


# lag terms look like: mean_compound_lag1, log_comments_change_lag4, etc.
LAG_RE = re.compile(r"^(?P<base>.+)_lag(?P<lag>[1-9]\d*)$")


@dataclass(frozen=True)
class TermPoint:
    base: str
    lag: int
    coef: float


def _parse_args() -> argparse.Namespace:
    root = _project_root()
    p = argparse.ArgumentParser(
        description=(
            "For each folder in full_subreddit_analysis/, read all_ols_coefficients.csv and "
            "produce grouped bar charts of lagged coefficients (lag1-4) for each candidate."
        )
    )
    p.add_argument(
        "--root",
        type=Path,
        default=root / "full_subreddit_analysis",
        help="Root folder containing analysis subfolders.",
    )
    p.add_argument(
        "--coefficients-name",
        default="all_ols_coefficients.csv",
        help="Filename to read from each analysis folder.",
    )
    p.add_argument(
        "--summaries-name",
        default="OLS_summaries_final.csv",
        help="Filename used to discover which dependent variables (y) to plot.",
    )
    p.add_argument(
        "--lags",
        default="1,2,3,4",
        help="Comma-separated lag integers to plot (default 1,2,3,4).",
    )
    p.add_argument(
        "--outdir-name",
        default="figures",
        help="Subfolder (inside each analysis folder) to write plots.",
    )
    p.add_argument(
        "--annotate",
        dest="annotate",
        action="store_true",
        default=True,
        help="Write coefficient values on top of bars (default on).",
    )
    p.add_argument(
        "--no-annotate",
        dest="annotate",
        action="store_false",
        help="Disable writing coefficient values on top of bars.",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="PNG DPI.",
    )
    p.add_argument(
        "--figsize",
        default="18,12",
        help="Figure size as 'width,height' in inches (default 18,12).",
    )
    p.add_argument(
        "--legend-outside/--legend-inside",
        dest="legend_outside",
        default=True,
        help="Place the lag legend outside the axes (default outside).",
    )
    p.add_argument(
        "--annotate-min-abs",
        type=float,
        default=0.0,
        help="Skip annotating coefficients with abs(value) < this threshold (default 0 = annotate all present terms).",
    )
    return p.parse_args()


def _extract_term_points(df: pd.DataFrame, lags: set[int]) -> list[TermPoint]:
    points: list[TermPoint] = []
    for _, row in df.iterrows():
        term = str(row["term"])
        if term == "const":
            continue
        m = LAG_RE.match(term)
        if not m:
            continue
        lag = int(m.group("lag"))
        if lag not in lags:
            continue
        base = m.group("base")
        try:
            coef = float(row["coef"])
        except Exception:
            continue
        points.append(TermPoint(base=base, lag=lag, coef=coef))
    return points


def _nice_base_order(y_name: str, bases: list[str]) -> list[str]:
    # Prefer a consistent order; fall back to alphabetical for any extras.
    preferred = [
        y_name,
        "mean_compound",
        "sentiment_change",
        "n_comments",
        "log_comments_change",
        "log_volume_change",
        "oi_change",
        "price_change",
    ]
    seen = set(bases)
    ordered: list[str] = []
    for b in preferred:
        if b in seen and b not in ordered:
            ordered.append(b)
    for b in sorted(bases):
        if b not in ordered:
            ordered.append(b)
    return ordered


def _plot_candidate_panel(
    ax: plt.Axes,
    *,
    points: list[TermPoint],
    candidate: str,
    y_name: str,
    lags_sorted: list[int],
    annotate: bool,
    annotate_min_abs: float,
    legend_outside: bool,
    show_legend: bool,
) -> None:
    if not points:
        ax.set_axis_off()
        ax.set_title(f"{candidate}: no lagged terms found")
        return

    # Layout: group by variable (base) on x-axis, with 4 lag bars side-by-side per variable.
    bases = sorted({p.base for p in points})
    bases = _nice_base_order(y_name, bases)
    xs = list(range(len(bases)))
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.8)

    # Colors per lag (so bars are comparable across variables)
    lag_colors = {lag: plt.get_cmap("tab10")(i % 10) for i, lag in enumerate(lags_sorted)}

    # Quick lookup
    val = {(p.base, p.lag): p.coef for p in points}

    n_lags = max(len(lags_sorted), 1)
    group_width = 0.82
    bar_w = group_width / n_lags

    # Small label offset relative to data magnitude to avoid huge gaps.
    all_vals = [abs(float(p.coef)) for p in points if p is not None]
    scale = max(all_vals) if all_vals else 0.0
    y_offset = max(0.001 * scale, 1e-9)

    for j, lag in enumerate(lags_sorted):
        offsets = [x - group_width / 2 + (j + 0.5) * bar_w for x in xs]
        heights = [float(val.get((base, lag), 0.0)) for base in bases]
        bars = ax.bar(
            offsets,
            heights,
            width=bar_w * 0.92,
            color=lag_colors[lag],
            label=f"lag{lag}",
        )

        if annotate:
            for i, (rect, h) in enumerate(zip(bars, heights)):
                if (bases[i], lag) not in val:
                    continue
                if float(annotate_min_abs) > 0 and abs(h) < float(annotate_min_abs):
                    continue
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    h + (y_offset if h >= 0 else -y_offset),
                    f"{h:.3g}",
                    ha="center",
                    va="bottom" if h >= 0 else "top",
                    fontsize=7,
                    rotation=90,
                )

    ax.set_xticks(xs, labels=bases, rotation=25, ha="right")
    ax.set_xlabel("Variable")
    ax.set_ylabel("Coefficient")
    ax.set_title(f"{candidate} OLS Coefficients")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)

    if show_legend:
        # Legend shows lags.
        if legend_outside:
            ax.legend(
                ncol=min(4, len(lags_sorted)),
                fontsize=8,
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
            )
        else:
            ax.legend(ncol=min(4, len(lags_sorted)), fontsize=8, frameon=False, loc="upper right")


def _plot_folder(folder: Path, *, coeff_name: str, summ_name: str, lags_sorted: list[int], annotate: bool, dpi: int, outdir_name: str) -> list[Path]:
    coeff_path = folder / coeff_name
    if not coeff_path.exists():
        return []

    df = pd.read_csv(coeff_path)
    required = {"candidate", "y", "term", "coef"}
    if not required.issubset(set(df.columns)):
        return []

    # Determine which dependent variables to plot.
    ys: list[str]
    summ_path = folder / summ_name
    if summ_path.exists():
        summ = pd.read_csv(summ_path)
        ys = sorted({str(y) for y in summ.get("y", pd.Series([], dtype="object")).dropna().unique()})
        if not ys:
            ys = sorted({str(y) for y in df["y"].dropna().unique()})
    else:
        ys = sorted({str(y) for y in df["y"].dropna().unique()})

    candidates = ["sliwa", "cuomo", "mamdani"]
    out_paths: list[Path] = []
    lags_set = set(lags_sorted)

    out_dir = folder / outdir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    figsize = (18, 12)
    try:
        w_s, h_s = str(getattr(_plot_folder, "_figsize_arg", "18,12")).split(",", 1)
        figsize = (float(w_s.strip()), float(h_s.strip()))
    except Exception:
        figsize = (18, 12)

    for y_name in ys:
        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=figsize,
            dpi=dpi,
            sharex=True,
            constrained_layout=True,
        )
        fig.suptitle(f"{folder.name}: OLS lagged coefficients for y={y_name}", fontsize=14)

        for ax, cand in zip(axes, candidates):
            sub = df[(df["candidate"] == cand) & (df["y"] == y_name)].copy()
            pts = _extract_term_points(sub, lags=lags_set)
            _plot_candidate_panel(
                ax,
                points=pts,
                candidate=cand,
                y_name=y_name,
                lags_sorted=lags_sorted,
                annotate=annotate,
                annotate_min_abs=float(getattr(_plot_folder, "_annotate_min_abs_arg", 0.0)),
                legend_outside=bool(getattr(_plot_folder, "_legend_outside_arg", True)),
                show_legend=False,
            )

        axes[-1].set_xlabel("Variable")
        # Single, figure-level legend (lags) in the top-right margin.
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper right",
                bbox_to_anchor=(0.99, 0.98),
                ncol=min(4, len(labels)),
                frameon=False,
                fontsize=9,
            )
        # Leave room on the right/top for the figure legend.
        fig.subplots_adjust(right=0.86, top=0.92)
        out_path = out_dir / f"ols_coefficients_{y_name}.png"
        fig.savefig(out_path)
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()
    lags_sorted = [int(x.strip()) for x in str(args.lags).split(",") if x.strip()]

    folders = sorted([p for p in root.iterdir() if p.is_dir()])
    produced = 0
    for folder in folders:
        # Pass a few args into _plot_folder without changing its call signature too much.
        _plot_folder._figsize_arg = str(args.figsize)  # type: ignore[attr-defined]
        _plot_folder._legend_outside_arg = bool(args.legend_outside)  # type: ignore[attr-defined]
        _plot_folder._annotate_min_abs_arg = float(args.annotate_min_abs)  # type: ignore[attr-defined]
        outs = _plot_folder(
            folder,
            coeff_name=args.coefficients_name,
            summ_name=args.summaries_name,
            lags_sorted=lags_sorted,
            annotate=bool(args.annotate),
            dpi=int(args.dpi),
            outdir_name=str(args.outdir_name),
        )
        produced += len(outs)

    print(f"Generated {produced} plot(s).")


if __name__ == "__main__":
    main()

