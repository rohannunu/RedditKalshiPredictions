from __future__ import annotations

import re
from typing import Any

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.pipeline.time_filter import add_date_parts, parse_timestamp_to_tz


def _is_valid_text(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype(str).str.strip()
    lowered = normalized.str.lower()
    invalid_tokens = {"", "[deleted]", "[removed]"}
    return ~lowered.isin(invalid_tokens)


def _word_count(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.findall(r"\b\w+\b").str.len().astype(int)


def _label_sentiment(compound: pd.Series) -> pd.Series:
    labels = pd.Series("neutral", index=compound.index, dtype="object")
    labels.loc[compound >= 0.05] = "positive"
    labels.loc[compound <= -0.05] = "negative"
    return labels


def score_comments_vader(
    df: pd.DataFrame,
    text_col: str,
    timestamp_col: str,
    timezone: str = "America/New_York",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = df.copy()

    valid_text_mask = _is_valid_text(working[text_col])
    dropped_invalid_text_count = int((~valid_text_mask).sum())
    working = working.loc[valid_text_mask].copy()

    parsed_dt, timestamp_type = parse_timestamp_to_tz(working[timestamp_col], timezone=timezone)
    working["parsed_datetime"] = parsed_dt

    invalid_dt_mask = working["parsed_datetime"].isna()
    dropped_invalid_datetime_count = int(invalid_dt_mask.sum())
    working = working.loc[~invalid_dt_mask].copy()

    working = add_date_parts(working, datetime_col="parsed_datetime")
    working["word_count"] = _word_count(working[text_col])

    analyzer = SentimentIntensityAnalyzer()
    scores = working[text_col].astype(str).map(analyzer.polarity_scores)
    score_df = pd.DataFrame(scores.tolist(), index=working.index)

    working["neg"] = score_df["neg"].astype(float)
    working["neu"] = score_df["neu"].astype(float)
    working["pos"] = score_df["pos"].astype(float)
    working["compound"] = score_df["compound"].astype(float)
    working["sentiment_label"] = _label_sentiment(working["compound"])

    output_cols = [
        text_col,
        timestamp_col,
        "parsed_datetime",
        "date",
        "year",
        "word_count",
        "neg",
        "neu",
        "pos",
        "compound",
        "sentiment_label",
    ]

    meta = {
        "timestamp_type": timestamp_type,
        "dropped_invalid_text_count": dropped_invalid_text_count,
        "dropped_invalid_datetime_count": dropped_invalid_datetime_count,
        "scored_row_count": int(len(working)),
    }

    return working[output_cols].copy(), meta


def aggregate_sentiment(
    scored_df: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    grouped = scored_df.groupby(group_col, as_index=False)

    agg = grouped.agg(
        n_comments=("compound", "size"),
        mean_compound=("compound", "mean"),
        median_compound=("compound", "median"),
        mean_pos=("pos", "mean"),
        mean_neu=("neu", "mean"),
        mean_neg=("neg", "mean"),
        total_words=("word_count", "sum"),
        mean_words=("word_count", "mean"),
    )

    label_frame = grouped["sentiment_label"].value_counts(normalize=True)
    if "proportion" in label_frame.columns:
        label_frame = label_frame.rename(columns={"proportion": "pct"})
    else:
        value_col = [c for c in label_frame.columns if c not in {group_col, "sentiment_label"}][0]
        label_frame = label_frame.rename(columns={value_col: "pct"})
    label_frame["pct"] = label_frame["pct"] * 100
    pct_wide = (
        label_frame.pivot(index=group_col, columns="sentiment_label", values="pct")
        .fillna(0)
        .rename(columns={
            "positive": "pct_positive",
            "negative": "pct_negative",
            "neutral": "pct_neutral",
        })
        .reset_index()
    )

    for pct_col in ["pct_positive", "pct_negative", "pct_neutral"]:
        if pct_col not in pct_wide.columns:
            pct_wide[pct_col] = 0.0

    out = agg.merge(
        pct_wide[[group_col, "pct_positive", "pct_negative", "pct_neutral"]],
        on=group_col,
        how="left",
    )

    ordered_cols = [
        group_col,
        "n_comments",
        "mean_compound",
        "median_compound",
        "mean_pos",
        "mean_neu",
        "mean_neg",
        "pct_positive",
        "pct_negative",
        "pct_neutral",
        "total_words",
        "mean_words",
    ]

    return out[ordered_cols].copy()
