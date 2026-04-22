from __future__ import annotations

import re
from typing import Pattern

import pandas as pd


def build_keyword_patterns(
    keywords: list[str],
    word_boundary: bool = True,
    case_insensitive: bool = True,
) -> dict[str, Pattern[str]]:
    flags = re.IGNORECASE if case_insensitive else 0
    patterns: dict[str, Pattern[str]] = {}
    for keyword in keywords:
        escaped = re.escape(keyword)
        expression = rf"\b{escaped}\b" if word_boundary else escaped
        patterns[keyword] = re.compile(expression, flags)
    return patterns


def daily_keyword_counts(
    df: pd.DataFrame,
    text_col: str,
    date_col: str = "date",
    keywords: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if keywords is None:
        keywords = []

    patterns = build_keyword_patterns(keywords, word_boundary=True, case_insensitive=True)

    working = df[[date_col, text_col]].copy()
    working[text_col] = working[text_col].fillna("").astype(str)

    daily_wide = (
        working.groupby(date_col, as_index=False)
        .size()
        .rename(columns={"size": "total_comments"})
    )

    for keyword, pattern in patterns.items():
        keyword_col = f"count_{keyword}"
        keyword_counts = (
            working.assign(_hit=working[text_col].str.contains(pattern, na=False))
            .groupby(date_col, as_index=False)["_hit"]
            .sum()
            .rename(columns={"_hit": keyword_col})
        )
        keyword_counts[keyword_col] = keyword_counts[keyword_col].astype(int)
        daily_wide = daily_wide.merge(keyword_counts, on=date_col, how="left")

    for keyword in keywords:
        col_name = f"count_{keyword}"
        if col_name not in daily_wide.columns:
            daily_wide[col_name] = 0

    count_cols = [f"count_{keyword}" for keyword in keywords]
    if count_cols:
        daily_wide[count_cols] = daily_wide[count_cols].fillna(0).astype(int)

    daily_long = daily_wide[[date_col] + count_cols].melt(
        id_vars=[date_col],
        var_name="keyword",
        value_name="count",
    )
    daily_long["keyword"] = daily_long["keyword"].str.replace("count_", "", regex=False)
    daily_long["count"] = daily_long["count"].astype(int)

    return daily_wide, daily_long


def normalize_per_1000(daily_df: pd.DataFrame) -> pd.DataFrame:
    out = daily_df.copy()
    total = out["total_comments"].replace(0, pd.NA)

    count_cols = [col for col in out.columns if col.startswith("count_")]
    for count_col in count_cols:
        keyword = count_col.replace("count_", "")
        rate_col = f"rate_{keyword}_per_1000"
        out[rate_col] = (out[count_col] / total) * 1000
        out[rate_col] = out[rate_col].fillna(0.0)

    return out
