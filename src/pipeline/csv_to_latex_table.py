from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _latex_escape(text: str) -> str:
    # Minimal, robust escaping for table cell text.
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def _format_number(x: float, digits: int) -> str:
    if pd.isna(x):
        return ""
    v = float(x)
    if not math.isfinite(v):
        return ""
    if abs(v - round(v)) < 1e-12:
        return f"{int(round(v)):,}"
    if abs(v) >= 1000:
        return f"{v:,.{digits}f}"
    return f"{v:.{digits}g}"


def _auto_column_format(df: pd.DataFrame) -> str:
    # l for non-numeric, r for numeric
    parts: list[str] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            parts.append("r")
        else:
            parts.append("l")
    return "".join(parts) if parts else "l"


def _coerce_to_display_df(df: pd.DataFrame, digits: int, na: str) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda v: _format_number(v, digits)).fillna(na)
        else:
            out[col] = out[col].fillna(na).astype(str).map(_latex_escape)
    out.columns = [_latex_escape(str(c)) for c in out.columns]
    return out


def _render_tabular(df: pd.DataFrame, *, col_format: str, booktabs: bool) -> str:
    # df is expected to already be escaped/converted to display strings
    lines: list[str] = []
    lines.append(f"\\begin{{tabular}}{{{col_format}}}")
    if booktabs:
        lines.append("\\toprule")
    else:
        lines.append("\\hline")

    header = " & ".join(str(c) for c in df.columns) + " \\\\"
    lines.append(header)

    if booktabs:
        lines.append("\\midrule")
    else:
        lines.append("\\hline")

    for _, row in df.iterrows():
        vals = [str(v) if v is not None else "" for v in row.tolist()]
        lines.append(" & ".join(vals) + " \\\\")

    if booktabs:
        lines.append("\\bottomrule")
    else:
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    root = _project_root()
    p = argparse.ArgumentParser(
        description="Convert a CSV to a LaTeX table (table* environment)."
    )
    p.add_argument("csv", type=Path, help="Path to input CSV.")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write LaTeX to this .tex file (also prints to stdout).",
    )
    p.add_argument(
        "--env",
        choices=["table", "table*"],
        default="table*",
        help="LaTeX float environment to use.",
    )
    p.add_argument("--caption", default="", help="\\caption{...} text (optional).")
    p.add_argument("--label", default="", help="\\label{...} value (optional).")
    p.add_argument(
        "--digits",
        type=int,
        default=4,
        help="Significant digits / decimal precision used for numeric formatting.",
    )
    p.add_argument(
        "--na",
        default="",
        help="String to use for missing values (default empty).",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="If >0, truncate to this many rows (head).",
    )
    p.add_argument(
        "--max-cols",
        type=int,
        default=0,
        help="If >0, truncate to this many columns (leftmost).",
    )
    p.add_argument(
        "--col-format",
        default="",
        help="Override tabular column format, e.g. 'lrrr'. Default auto.",
    )
    p.add_argument(
        "--booktabs",
        dest="booktabs",
        default=True,
        help="Use booktabs (\\toprule/\\midrule/\\bottomrule).",
    )
    p.add_argument(
        "--no-booktabs",
        dest="booktabs",
        action="store_false",
        help="Use \\hline instead of booktabs rules.",
    )
    p.add_argument(
        "--small",
        dest="small",
        default=True,
        help="Wrap tabular in \\small for denser tables.",
    )
    p.add_argument(
        "--no-small",
        dest="small",
        action="store_false",
        help="Do not add \\small.",
    )
    p.add_argument(
        "--tabcolsep",
        type=float,
        default=4.0,
        help="Set \\tabcolsep (in pt).",
    )
    p.add_argument(
        "--arraystretch",
        type=float,
        default=1.0,
        help="Set \\arraystretch.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    csv_path = args.csv.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()
    if args.max_cols and args.max_cols > 0:
        df = df.iloc[:, : args.max_cols].copy()

    display_df = _coerce_to_display_df(df, digits=int(args.digits), na=str(args.na))
    col_format = args.col_format.strip() or _auto_column_format(display_df)

    tabular = _render_tabular(display_df, col_format=col_format, booktabs=bool(args.booktabs))

    # pandas emits a full tabular environment; wrap it in a float.
    lines: list[str] = []
    lines.append(f"\\begin{{{args.env}}}[t]")
    lines.append("\\centering")
    if args.small:
        lines.append("\\small")
    if args.tabcolsep != 4.0:
        lines.append(f"\\setlength{{\\tabcolsep}}{{{args.tabcolsep:g}pt}}")
    if args.arraystretch != 1.0:
        lines.append(f"\\renewcommand{{\\arraystretch}}{{{args.arraystretch:g}}}")

    if args.booktabs:
        lines.append("% requires \\usepackage{booktabs}")
    lines.append(tabular.strip())

    if args.caption:
        lines.append(f"\\caption{{{_latex_escape(args.caption)}}}")
    if args.label:
        lines.append(f"\\label{{{args.label}}}")
    lines.append(f"\\end{{{args.env}}}")
    out = "\n".join(lines) + "\n"

    print(out, end="")
    if args.output is not None:
        out_path = args.output.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()

