# graph_granger_hourly_results.py
#
# Reads granger_hourly_results_summary.csv and graphs the hourly Granger results
# separately for each candidate, showing only lags 1–12.
#
# Output:
#   granger_hourly_graphs/
#       mamdani_hourly_heatmap.png
#       mamdani_hourly_pvalue_lines.png
#       mamdani_hourly_significant_only.png
#       mamdani_hourly_best_pvalues.png
#       ...
#
# Install:
#   pip install pandas matplotlib
#
# Run:
#   python graph_granger_hourly_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# SETTINGS
# ----------------------------
INPUT_FILE = "granger_hourly_results_summary.csv"
OUTPUT_DIR = "granger_hourly_graphs"
P_THRESHOLD = 0.05
MAX_PLOTTED_LAG = 12

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# LABEL MAPPING
# ----------------------------
LABEL_MAP = {
    "price_change": "Price Change",
    "log_volume_change": "Trading Activity",
    "oi_change": "Open Interest Change",
    "mean_compound": "Sentiment Tone",
    "sentiment_change": "Change in Sentiment",
    "n_comments": "Comment Volume",
    "log_comments_change": "Change in Comment Volume",
}

def pretty_label(name):
    return LABEL_MAP.get(name, name)

def format_pvalue(val):
    if pd.isna(val):
        return ""
    if val < 0.001:
        return "<0.001"
    return f"{val:.3f}"

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(INPUT_FILE)

required_cols = [
    "candidate", "x", "y", "lag",
    "p_ssr_ftest", "p_params_ftest", "significant_5pct"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# Keep only the first 12 hourly lags
df = df[df["lag"] <= MAX_PLOTTED_LAG].copy()

# Build nicer relationship labels
df["pair"] = df["x"].map(pretty_label) + " → " + df["y"].map(pretty_label)
df = df.sort_values(["candidate", "pair", "lag"]).reset_index(drop=True)

candidates = df["candidate"].unique()

# ----------------------------
# HELPERS
# ----------------------------
def save_heatmap(candidate_df, candidate):
    pivot = candidate_df.pivot(index="pair", columns="lag", values="p_ssr_ftest")
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(11, max(5, 0.6 * len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("p-value")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title(f"Hourly Granger p-value heatmap (lags 1–{MAX_PLOTTED_LAG}): {candidate.capitalize()}")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Relationship")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(
                    j, i, format_pvalue(val),
                    ha="center", va="center", fontsize=7
                )

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, f"{candidate}_hourly_heatmap.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def save_pvalue_lines(candidate_df, candidate):
    pairs = sorted(candidate_df["pair"].unique())

    fig, ax = plt.subplots(figsize=(12, 7))
    for pair in pairs:
        temp = candidate_df[candidate_df["pair"] == pair].sort_values("lag")
        ax.plot(temp["lag"], temp["p_ssr_ftest"], marker="o", label=pair)

    ax.axhline(P_THRESHOLD, linestyle="--", linewidth=1, label="0.05 threshold")
    ax.set_title(f"Hourly Granger p-values by lag (1–{MAX_PLOTTED_LAG}): {candidate.capitalize()}")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("p-value")
    ax.set_xticks(sorted(candidate_df["lag"].unique()))
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.tight_layout()

    outpath = os.path.join(OUTPUT_DIR, f"{candidate}_hourly_pvalue_lines.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def save_significant_only(candidate_df, candidate):
    sig_pairs = (
        candidate_df.groupby("pair")["p_ssr_ftest"]
        .min()
        .reset_index()
        .query("p_ssr_ftest < @P_THRESHOLD")["pair"]
        .tolist()
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    if len(sig_pairs) == 0:
        ax.text(
            0.5, 0.5,
            f"No significant relationships at p < {P_THRESHOLD}",
            ha="center", va="center", fontsize=14
        )
        ax.set_title(f"Hourly significant relationships only (1–{MAX_PLOTTED_LAG}): {candidate.capitalize()}")
        ax.axis("off")
    else:
        for pair in sig_pairs:
            temp = candidate_df[candidate_df["pair"] == pair].sort_values("lag")
            ax.plot(temp["lag"], temp["p_ssr_ftest"], marker="o", label=pair)

        ax.axhline(P_THRESHOLD, linestyle="--", linewidth=1, label="0.05 threshold")
        ax.set_title(f"Hourly significant relationships only (1–{MAX_PLOTTED_LAG}): {candidate.capitalize()}")
        ax.set_xlabel("Lag (hours)")
        ax.set_ylabel("p-value")
        ax.set_xticks(sorted(candidate_df["lag"].unique()))
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, f"{candidate}_hourly_significant_only.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def save_best_pvalue_bar(candidate_df, candidate):
    best_df = (
        candidate_df.groupby("pair", as_index=False)["p_ssr_ftest"]
        .min()
        .rename(columns={"p_ssr_ftest": "best_p"})
        .sort_values("best_p", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, max(5, 0.5 * len(best_df))))
    ax.barh(best_df["pair"], best_df["best_p"])
    ax.axvline(P_THRESHOLD, linestyle="--", linewidth=1)
    ax.set_title(f"Best hourly p-value by relationship (lags 1–{MAX_PLOTTED_LAG}): {candidate.capitalize()}")
    ax.set_xlabel("Minimum p-value across plotted lags")
    ax.set_ylabel("Relationship")
    ax.invert_yaxis()

    for i, v in enumerate(best_df["best_p"]):
        label = "<0.001" if v < 0.001 else f"{v:.3f}"
        ax.text(v + 0.005, i, label, va="center", fontsize=8)

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, f"{candidate}_hourly_best_pvalues.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


# ----------------------------
# MAIN
# ----------------------------
print(f"Loaded {len(df)} rows from {INPUT_FILE}")
print(f"Candidates: {list(candidates)}")
print(f"Plotting only lags 1–{MAX_PLOTTED_LAG}")

for candidate in candidates:
    candidate_df = df[df["candidate"] == candidate].copy()

    print(f"\nMaking hourly graphs for: {candidate}")
    save_heatmap(candidate_df, candidate)
    save_pvalue_lines(candidate_df, candidate)
    save_significant_only(candidate_df, candidate)
    save_best_pvalue_bar(candidate_df, candidate)

print(f"\nDone. Graphs saved in: {OUTPUT_DIR}")