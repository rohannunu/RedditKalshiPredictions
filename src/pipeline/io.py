from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


def load_csv(csv_path: str | Path) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _find_candidates(columns: list[str], patterns: list[str]) -> list[str]:
    found: list[str] = []
    for col in columns:
        lowered = col.lower()
        if any(re.search(pattern, lowered) for pattern in patterns):
            found.append(col)
    return found


def detect_schema(df: pd.DataFrame) -> dict[str, Any]:
    columns = list(df.columns)

    text_candidates = _find_candidates(columns, [r"body", r"comment", r"text", r"content", r"message"])
    ts_candidates = _find_candidates(columns, [r"created", r"timestamp", r"time", r"date", r"utc"])

    if not text_candidates:
        object_cols = [c for c in columns if pd.api.types.is_object_dtype(df[c])]
        if object_cols:
            ranked = sorted(
                object_cols,
                key=lambda c: float(df[c].fillna("").astype(str).str.len().mean()),
                reverse=True,
            )
            text_candidates = ranked[:3]

    preferred_text = ["body", "comment", "text", "content", "message"]
    text_col = None
    for preference in preferred_text:
        text_col = next((c for c in text_candidates if preference in c.lower()), None)
        if text_col:
            break

    preferred_ts = ["created_utc_comment", "created_utc", "created", "timestamp", "time", "date"]
    timestamp_col = None
    for preference in preferred_ts:
        timestamp_col = next((c for c in ts_candidates if preference in c.lower()), None)
        if timestamp_col:
            break

    return {
        "text_col": text_col,
        "timestamp_col": timestamp_col,
        "text_candidates": text_candidates,
        "timestamp_candidates": ts_candidates,
    }
