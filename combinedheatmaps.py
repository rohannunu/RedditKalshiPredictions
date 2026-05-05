# combined_granger_heatmaps_clean.py
#
# Creates one poster-ready combined figure with:
#   Cuomo | Mamdani | Sliwa
# from granger_hourly_results_summary.csv
#
# This version REMOVES the numbers inside the heatmap cells.
#
# Run:
#   python3 combined_granger_heatmaps_clean.py
#
# Install if needed:
#   pip install pandas matplotlib numpy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
INPUT_FILE = "granger_hourly_results_summary.csv"
OUTPUT_FILE = "combined_hourly_granger_heatmaps_clean.png"
MAX_LAG = 12

# Candidate order on the poster
CANDIDATE_ORDER = ["cuomo", "mamdani", "sliwa"]

# Cleaner labels
LABEL_MAP = {
    "price_change": "Price Change",
    "log_volume_change": "Trading Activity",
    "oi_change": "Open Interest Change",
    "mean_compound": "Sentiment Tone",
    "sentiment_change": "Change in Sentiment",
    "n_comments": "Comment Volume",
    "log_comments_change": "Change in Comment Volume",
}

# Row order you want to show
PAIR_ORDER = [
    "Change in Comment Volume → Price Change",
    "Change in Sentiment → Price Change",
    "Comment Volume → Open Interest Change",
    "Comment Volume → Price Change",
    "Comment Volume → Trading Activity",
    "Sentiment Tone → Open Interest Change",
    "Sentiment Tone → Price Change",
    "Sentiment Tone → Trading Activity",
]

def pretty_label(name):
    return LABEL_MAP.get(name, name)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE)

required_cols = ["candidate", "x", "y", "lag", "p_ssr_ftest"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Keep only lags 1-12
df = df[df["lag"] <= MAX_LAG].copy()

# Clean relationship labels
df["pair"] = df["x"].map(pretty_label) + " → " + df["y"].map(pretty_label)

# Keep only the relationships we want
df = df[df["pair"].isin(PAIR_ORDER)].copy()

# --------------------------------------------------
# BUILD FIGURE
# --------------------------------------------------
fig, axes = plt.subplots(
    1, 3,
    figsize=(16, 5.4),
    gridspec_kw={"width_ratios": [1, 1, 1]},
    constrained_layout=True
)

vmin = 0.0
vmax = 1.0
im = None

for idx, candidate in enumerate(CANDIDATE_ORDER):
    ax = axes[idx]

    sub = df[df["candidate"].str.lower() == candidate].copy()

    pivot = sub.pivot(index="pair", columns="lag", values="p_ssr_ftest")

    # Force same row/column order for all panels
    pivot = pivot.reindex(index=PAIR_ORDER, columns=list(range(1, MAX_LAG + 1)))

    data = pivot.values

    im = ax.imshow(
        data,
        aspect="auto",
        vmin=vmin,
        vmax=vmax
    )

    # Panel title
    ax.set_title(candidate.capitalize(), fontsize=15, pad=10)

    # X-axis
    ax.set_xticks(np.arange(MAX_LAG))
    ax.set_xticklabels(range(1, MAX_LAG + 1), fontsize=10)

    # Y-axis
    ax.set_yticks(np.arange(len(PAIR_ORDER)))
    if idx == 0:
        ax.set_yticklabels(PAIR_ORDER, fontsize=10)
        ax.set_ylabel("Relationship", fontsize=12)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)

# Shared x-label
fig.supxlabel("Lag (hours)", fontsize=12)

# Shared colorbar
cbar = fig.colorbar(
    im,
    ax=axes,
    shrink=0.95,
    pad=0.02
)
cbar.set_label("p-value (darker = smaller)", fontsize=12)

# Optional overall title
fig.suptitle("Hourly Granger p-value Heatmaps (Lags 1–12)", fontsize=16, y=1.03)

# Save
plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved combined figure to: {OUTPUT_FILE}")