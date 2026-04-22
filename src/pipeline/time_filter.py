from __future__ import annotations

from typing import Literal

import pandas as pd


TimestampType = Literal["iso_or_string", "unix_seconds", "unix_milliseconds", "unknown"]


def _detect_timestamp_type(series: pd.Series) -> TimestampType:
    non_null = series.dropna()
    if non_null.empty:
        return "unknown"

    numeric = pd.to_numeric(non_null, errors="coerce")
    numeric_ratio = float(numeric.notna().mean())
    if numeric_ratio <= 0.95:
        return "iso_or_string"

    median_value = float(numeric.median())
    if median_value > 1e12:
        return "unix_milliseconds"
    if median_value > 1e9:
        return "unix_seconds"
    return "unknown"


def parse_timestamp_to_tz(
    series: pd.Series,
    timezone: str = "America/New_York",
) -> tuple[pd.Series, TimestampType]:
    ts_type = _detect_timestamp_type(series)

    if ts_type == "unix_milliseconds":
        parsed_utc = pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit="ms", utc=True, errors="coerce")
        return parsed_utc.dt.tz_convert(timezone), ts_type

    if ts_type == "unix_seconds":
        parsed_utc = pd.to_datetime(pd.to_numeric(series, errors="coerce"), unit="s", utc=True, errors="coerce")
        return parsed_utc.dt.tz_convert(timezone), ts_type

    parsed_utc = pd.to_datetime(series, utc=True, errors="coerce")
    non_null = series.dropna().astype(str).head(50)
    has_tz_hint = bool(non_null.str.contains(r"Z$|[+-]\d{2}:?\d{2}$", regex=True, na=False).any())

    if has_tz_hint:
        return parsed_utc.dt.tz_convert(timezone), "iso_or_string"

    parsed_naive = pd.to_datetime(series, errors="coerce")
    localized = parsed_naive.dt.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")
    return localized, "iso_or_string"


def add_date_parts(df: pd.DataFrame, datetime_col: str = "parsed_datetime") -> pd.DataFrame:
    out = df.copy()
    out["date"] = out[datetime_col].dt.strftime("%Y-%m-%d")
    out["year"] = out[datetime_col].dt.year
    out["month"] = out[datetime_col].dt.month
    out["day"] = out[datetime_col].dt.day
    return out


def filter_date_range(
    df: pd.DataFrame,
    datetime_col: str,
    start: str,
    end: str,
    timezone: str = "America/New_York",
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz=timezone)
    end_ts = pd.Timestamp(end, tz=timezone)
    return df[(df[datetime_col] >= start_ts) & (df[datetime_col] <= end_ts)].copy()
