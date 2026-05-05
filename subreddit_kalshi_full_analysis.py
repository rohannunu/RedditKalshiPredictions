# subreddit_kalshi_full_analysis.py
#
# For each subreddit folder inside sentiment_hourly_subreddit/:
#   - merges candidate-specific sentiment with candidate-specific Kalshi data
#   - runs correlations for every Reddit input vs every Kalshi output
#   - runs lagged OLS for every Reddit input -> Kalshi output pair
#   - runs Granger causality for every Reddit input -> Kalshi output pair
#   - saves all outputs inside that subreddit's own folder
#
# Outputs:
#   full_subreddit_analysis/
#       all/
#           correlations/
#           ols/
#           granger/
#       AskALiberal/
#           ...
#
# Install:
#   pip install pandas numpy matplotlib statsmodels
#
# Run:
#   python3 subreddit_kalshi_full_analysis.py

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# ============================================================
# CONFIG
# ============================================================

KALSHI_FILES = {
    "mamdani": "mamdani_data.csv",
    "cuomo": "cuomo_data.csv",
    "sliwa": "sliwa_data.csv",
}

SENTIMENT_ROOT = Path("sentiment_hourly_subreddit")
OUTPUT_ROOT = Path("full_subreddit_analysis")

MAX_GRANGER_LAG = 12
MAX_OLS_LAG = 12
MIN_ROWS_FOR_TEST = 100

SIGNIFICANCE_LEVEL = 0.05
FILL_MISSING_SENTIMENT_WITH_ZERO = True
FFILL_KALSHI_LEVELS = True

REDDIT_INPUTS = [
    "mean_compound",
    "sentiment_change",
    "n_comments",
    "log_comments_change",
]

KALSHI_OUTPUTS = [
    "price_change",
    "log_volume_change",
    "oi_change",
]

LABEL_MAP = {
    "price_change": "Price Change",
    "log_volume_change": "Trading Activity",
    "oi_change": "Open Interest Change",
    "mean_compound": "Sentiment Tone",
    "sentiment_change": "Change in Sentiment",
    "n_comments": "Comment Volume",
    "log_comments_change": "Change in Comment Volume",
}


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def pretty_label(name: str) -> str:
    return LABEL_MAP.get(name, name)


def safe_log_diff(series: pd.Series) -> pd.Series:
    return np.log1p(series).diff()


def adf_test(series, name):
    s = pd.Series(series).dropna()
    if len(s) < 20 or s.nunique() <= 1:
        return {
            "variable": name,
            "n": len(s),
            "adf_stat": np.nan,
            "p_value": np.nan,
            "stationary_5pct": False,
        }

    result = adfuller(s, autolag="AIC")
    return {
        "variable": name,
        "n": len(s),
        "adf_stat": result[0],
        "p_value": result[1],
        "stationary_5pct": result[1] < 0.05,
    }


def get_sentiment_files_for_subreddit(subreddit_dir: Path):
    return {
        "mamdani": subreddit_dir / "sentiment_hourly_mamdani.csv",
        "cuomo": subreddit_dir / "sentiment_hourly_cuomo.csv",
        "sliwa": subreddit_dir / "sentiment_hourly_sliwa.csv",
    }


def load_and_prepare_candidate(candidate_name: str, sentiment_file: Path, kalshi_file: str) -> pd.DataFrame:
    kalshi = pd.read_csv(kalshi_file)
    sent = pd.read_csv(sentiment_file)

    # Parse timestamps
    kalshi["timestamp"] = pd.to_datetime(kalshi["date"]).dt.floor("h")

    possible_time_cols = ["timestamp", "date", "datetime", "hour"]
    sent_time_col = None
    for col in possible_time_cols:
        if col in sent.columns:
            sent_time_col = col
            break

    if sent_time_col is None:
        raise ValueError(
            f"Could not find a timestamp column in {sentiment_file}. Tried: {possible_time_cols}"
        )

    sent["timestamp"] = pd.to_datetime(sent[sent_time_col]).dt.floor("h")

    required_kalshi_cols = ["timestamp", "yes_bid.close_dollars", "volume", "open_interest"]
    missing_kalshi_cols = [c for c in required_kalshi_cols if c not in kalshi.columns]
    if missing_kalshi_cols:
        raise ValueError(f"Missing required Kalshi columns in {kalshi_file}: {missing_kalshi_cols}")

    required_sent_cols = ["timestamp", "mean_compound", "n_comments"]
    missing_sent_cols = [c for c in required_sent_cols if c not in sent.columns]
    if missing_sent_cols:
        raise ValueError(f"Missing required sentiment columns in {sentiment_file}: {missing_sent_cols}")

    kalshi = kalshi[required_kalshi_cols].copy()
    sent = sent[required_sent_cols].copy()

    # Aggregate duplicate hours
    kalshi = (
        kalshi.sort_values("timestamp")
        .groupby("timestamp", as_index=False)
        .agg({
            "yes_bid.close_dollars": "last",
            "volume": "sum",
            "open_interest": "last",
        })
    )

    sent = (
        sent.sort_values("timestamp")
        .groupby("timestamp", as_index=False)
        .agg({
            "mean_compound": "mean",
            "n_comments": "sum",
        })
    )

    # Overlap only
    start_time = max(kalshi["timestamp"].min(), sent["timestamp"].min())
    end_time = min(kalshi["timestamp"].max(), sent["timestamp"].max())

    if pd.isna(start_time) or pd.isna(end_time) or start_time > end_time:
        raise ValueError(f"No overlapping timestamps for {candidate_name}")

    full_times = pd.DataFrame({
        "timestamp": pd.date_range(start_time, end_time, freq="h")
    })

    df = (
        full_times
        .merge(kalshi, on="timestamp", how="left")
        .merge(sent, on="timestamp", how="left")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if FFILL_KALSHI_LEVELS:
        df["yes_bid.close_dollars"] = df["yes_bid.close_dollars"].ffill()
        df["open_interest"] = df["open_interest"].ffill()

    df["volume"] = df["volume"].fillna(0)

    if FILL_MISSING_SENTIMENT_WITH_ZERO:
        df["mean_compound"] = df["mean_compound"].fillna(0)
        df["n_comments"] = df["n_comments"].fillna(0)
    else:
        df = df.dropna(subset=["mean_compound", "n_comments"])

    df = df.dropna(subset=["yes_bid.close_dollars"]).copy()

    # Derived variables
    df["price_change"] = df["yes_bid.close_dollars"].diff()
    df["log_volume_change"] = safe_log_diff(df["volume"])
    df["oi_change"] = df["open_interest"].diff()
    df["sentiment_change"] = df["mean_compound"].diff()
    df["log_comments_change"] = safe_log_diff(df["n_comments"])

    df["candidate"] = candidate_name
    return df


# ============================================================
# CORRELATIONS
# ============================================================

def run_correlations(df: pd.DataFrame, subreddit: str, candidate: str) -> pd.DataFrame:
    rows = []
    for x in REDDIT_INPUTS:
        for y in KALSHI_OUTPUTS:
            temp = df[[x, y]].dropna().copy()
            n = len(temp)

            if n < 5 or temp[x].nunique() <= 1 or temp[y].nunique() <= 1:
                pearson = np.nan
                spearman = np.nan
            else:
                pearson = temp[x].corr(temp[y], method="pearson")
                spearman = temp[x].corr(temp[y], method="spearman")

            rows.append({
                "subreddit": subreddit,
                "candidate": candidate,
                "x": x,
                "y": y,
                "n": n,
                "pearson_corr": pearson,
                "spearman_corr": spearman,
            })
    return pd.DataFrame(rows)


# ============================================================
# OLS
# ============================================================

def build_lagged_regression_data(df, y_col, x_col, max_lag):
    temp = df[["timestamp", y_col, x_col]].copy()

    for lag in range(1, max_lag + 1):
        temp[f"{y_col}_lag{lag}"] = temp[y_col].shift(lag)
        temp[f"{x_col}_lag{lag}"] = temp[x_col].shift(lag)

    temp = temp.dropna().copy()
    return temp


def fit_ols_with_lags(df, subreddit, candidate, y_col, x_col, max_lag):
    temp = build_lagged_regression_data(df, y_col, x_col, max_lag)

    if len(temp) < MIN_ROWS_FOR_TEST:
        return None, {
            "subreddit": subreddit,
            "candidate": candidate,
            "y": y_col,
            "x": x_col,
            "n": len(temp),
            "error": f"Not enough rows (< {MIN_ROWS_FOR_TEST})"
        }

    y = temp[y_col]
    x_cols = (
        [f"{y_col}_lag{i}" for i in range(1, max_lag + 1)] +
        [f"{x_col}_lag{i}" for i in range(1, max_lag + 1)]
    )
    X = sm.add_constant(temp[x_cols])

    model = sm.OLS(y, X).fit()

    restriction = " = 0, ".join([f"{x_col}_lag{i}" for i in range(1, max_lag + 1)]) + " = 0"
    ftest = model.f_test(restriction)

    coef_rows = []
    for term in model.params.index:
        coef_rows.append({
            "subreddit": subreddit,
            "candidate": candidate,
            "y": y_col,
            "x": x_col,
            "term": term,
            "coef": model.params[term],
            "stderr": model.bse[term],
            "tstat": model.tvalues[term],
            "pvalue": model.pvalues[term],
        })

    summary_row = {
        "subreddit": subreddit,
        "candidate": candidate,
        "y": y_col,
        "x": x_col,
        "n": int(model.nobs),
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "aic": model.aic,
        "bic": model.bic,
        "joint_f_stat": float(np.asarray(ftest.fvalue).squeeze()),
        "joint_f_pvalue": float(np.asarray(ftest.pvalue).squeeze()),
    }

    return model, pd.DataFrame(coef_rows), summary_row, temp


# ============================================================
# GRANGER
# ============================================================

def run_granger_pair(df, subreddit, candidate, y_col, x_col, maxlag=12):
    temp = df[[y_col, x_col]].dropna().copy()

    if len(temp) < MIN_ROWS_FOR_TEST:
        return [], {
            "subreddit": subreddit,
            "candidate": candidate,
            "y": y_col,
            "x": x_col,
            "n": len(temp),
            "error": f"Not enough rows (< {MIN_ROWS_FOR_TEST})"
        }

    if temp[y_col].nunique() <= 1 or temp[x_col].nunique() <= 1:
        return [], {
            "subreddit": subreddit,
            "candidate": candidate,
            "y": y_col,
            "x": x_col,
            "n": len(temp),
            "error": "One series has no variation"
        }

    try:
        results = grangercausalitytests(temp[[y_col, x_col]], maxlag=maxlag, verbose=False)

        rows = []
        for lag in range(1, maxlag + 1):
            test_dict = results[lag][0]
            ssr_f_stat = test_dict["ssr_ftest"][0]
            p_ssr = test_dict["ssr_ftest"][1]
            params_f_stat = test_dict["params_ftest"][0]
            p_params = test_dict["params_ftest"][1]

            rows.append({
                "subreddit": subreddit,
                "candidate": candidate,
                "y": y_col,
                "x": x_col,
                "n": len(temp),
                "lag": lag,
                "ssr_f_stat": ssr_f_stat,
                "p_ssr_ftest": p_ssr,
                "params_f_stat": params_f_stat,
                "p_params_ftest": p_params,
                "significant_5pct": (p_ssr < SIGNIFICANCE_LEVEL),
            })

        return rows, None

    except Exception as e:
        return [], {
            "subreddit": subreddit,
            "candidate": candidate,
            "y": y_col,
            "x": x_col,
            "n": len(temp),
            "error": str(e)
        }


def make_granger_heatmap(granger_df: pd.DataFrame, outpath: Path, subreddit: str, candidate: str):
    if granger_df.empty:
        return

    plot_df = granger_df.copy()
    plot_df["pair"] = plot_df["x"].map(pretty_label) + " → " + plot_df["y"].map(pretty_label)

    pivot = plot_df.pivot(index="pair", columns="lag", values="p_ssr_ftest")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("p-value")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title(f"{subreddit} | {candidate.capitalize()} | Granger p-value heatmap")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Relationship")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN PER SUBREDDIT
# ============================================================

def process_subreddit(subreddit_dir: Path):
    subreddit = subreddit_dir.name
    print(f"\n{'#'*100}")
    print(f"PROCESSING SUBREDDIT: {subreddit}")
    print(f"{'#'*100}")

    sentiment_files = get_sentiment_files_for_subreddit(subreddit_dir)

    subreddit_out = OUTPUT_ROOT / subreddit
    corr_dir = subreddit_out / "correlations"
    ols_dir = subreddit_out / "ols"
    granger_dir = subreddit_out / "granger"
    debug_dir = subreddit_out / "debug"

    for d in [subreddit_out, corr_dir, ols_dir, granger_dir, debug_dir]:
        ensure_dir(d)

    corr_all = []
    ols_summary_all = []
    ols_coef_all = []
    granger_all = []
    adf_all = []
    errors = []

    for candidate, sentiment_file in sentiment_files.items():
        kalshi_file = KALSHI_FILES[candidate]

        if not sentiment_file.exists():
            errors.append({
                "subreddit": subreddit,
                "candidate": candidate,
                "error": f"Missing sentiment file: {sentiment_file}"
            })
            continue

        try:
            df = load_and_prepare_candidate(candidate, sentiment_file, kalshi_file)
            df.to_csv(debug_dir / f"{candidate}_merged_transformed_hourly.csv", index=False)

            # ADF
            for var in ["yes_bid.close_dollars", "mean_compound", "n_comments",
                        "price_change", "log_volume_change", "oi_change",
                        "sentiment_change", "log_comments_change"]:
                row = adf_test(df[var], var)
                row["subreddit"] = subreddit
                row["candidate"] = candidate
                adf_all.append(row)

            # Correlations
            corr_df = run_correlations(df, subreddit, candidate)
            corr_df.to_csv(corr_dir / f"{candidate}_correlations.csv", index=False)
            corr_all.append(corr_df)

            # OLS
            candidate_ols_summaries = []
            candidate_ols_coefs = []

            for y_col in KALSHI_OUTPUTS:
                for x_col in REDDIT_INPUTS:
                    result = fit_ols_with_lags(df, subreddit, candidate, y_col, x_col, MAX_OLS_LAG)

                    if result is None:
                        continue

                    if len(result) == 2:
                        errors.append(result[1])
                        continue

                    model, coef_df, summary_row, model_input = result
                    candidate_ols_summaries.append(summary_row)
                    candidate_ols_coefs.append(coef_df)

                    base = f"{candidate}__{x_col}_to_{y_col}"
                    coef_df.to_csv(ols_dir / f"{base}__coefficients.csv", index=False)
                    model_input.to_csv(ols_dir / f"{base}__model_input.csv", index=False)
                    with open(ols_dir / f"{base}__summary.txt", "w") as f:
                        f.write(model.summary().as_text())

            if candidate_ols_summaries:
                cand_ols_sum_df = pd.DataFrame(candidate_ols_summaries)
                cand_ols_sum_df.to_csv(ols_dir / f"{candidate}_ols_summary.csv", index=False)
                ols_summary_all.append(cand_ols_sum_df)

            if candidate_ols_coefs:
                cand_ols_coef_df = pd.concat(candidate_ols_coefs, ignore_index=True)
                cand_ols_coef_df.to_csv(ols_dir / f"{candidate}_ols_all_coefficients.csv", index=False)
                ols_coef_all.append(cand_ols_coef_df)

            # Granger
            candidate_granger_rows = []
            for y_col in KALSHI_OUTPUTS:
                for x_col in REDDIT_INPUTS:
                    rows, err = run_granger_pair(df, subreddit, candidate, y_col, x_col, maxlag=MAX_GRANGER_LAG)
                    if err is not None:
                        errors.append(err)
                    if rows:
                        candidate_granger_rows.extend(rows)

            if candidate_granger_rows:
                cand_granger_df = pd.DataFrame(candidate_granger_rows)
                cand_granger_df.to_csv(granger_dir / f"{candidate}_granger_results_summary.csv", index=False)

                sig_df = cand_granger_df[cand_granger_df["significant_5pct"]].copy()
                sig_df.to_csv(granger_dir / f"{candidate}_granger_significant_results.csv", index=False)

                best_df = (
                    cand_granger_df
                    .groupby(["subreddit", "candidate", "y", "x"], as_index=False)["p_ssr_ftest"]
                    .min()
                    .rename(columns={"p_ssr_ftest": "best_p_value"})
                )
                best_df.to_csv(granger_dir / f"{candidate}_granger_best_pvalues.csv", index=False)

                make_granger_heatmap(
                    cand_granger_df,
                    granger_dir / f"{candidate}_granger_heatmap.png",
                    subreddit=subreddit,
                    candidate=candidate,
                )

                granger_all.append(cand_granger_df)

        except Exception as e:
            errors.append({
                "subreddit": subreddit,
                "candidate": candidate,
                "error": str(e),
            })

    # Save subreddit-wide combined outputs
    if corr_all:
        pd.concat(corr_all, ignore_index=True).to_csv(subreddit_out / "all_correlations.csv", index=False)

    if ols_summary_all:
        pd.concat(ols_summary_all, ignore_index=True).to_csv(subreddit_out / "all_ols_summaries.csv", index=False)

    if ols_coef_all:
        pd.concat(ols_coef_all, ignore_index=True).to_csv(subreddit_out / "all_ols_coefficients.csv", index=False)

    if granger_all:
        pd.concat(granger_all, ignore_index=True).to_csv(subreddit_out / "all_granger_results.csv", index=False)

    if adf_all:
        pd.DataFrame(adf_all).to_csv(subreddit_out / "adf_stationarity_checks.csv", index=False)

    if errors:
        pd.DataFrame(errors).to_csv(subreddit_out / "errors.csv", index=False)

    print(f"Saved outputs for subreddit: {subreddit}")


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(OUTPUT_ROOT)

    if not SENTIMENT_ROOT.exists():
        raise FileNotFoundError(f"Missing sentiment root folder: {SENTIMENT_ROOT}")

    subreddit_dirs = sorted([p for p in SENTIMENT_ROOT.iterdir() if p.is_dir()])

    if not subreddit_dirs:
        raise ValueError(f"No subreddit folders found in {SENTIMENT_ROOT}")

    print("Found subreddit folders:")
    for p in subreddit_dirs:
        print(f" - {p.name}")

    for subreddit_dir in subreddit_dirs:
        process_subreddit(subreddit_dir)

    print("\nDone. Outputs saved in:")
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()