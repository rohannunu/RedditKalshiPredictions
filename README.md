# CS598 Reddit Sentiment Pipeline

This project analyzes Reddit comments with two main workflows:

- October 2025 keyword frequency analysis
- VADER sentiment analysis with topic/candidate/relevance slices and plotting

Input data is expected at:

- `filtered_reddit_data.csv`

## Quickstart (3 Commands)

From the project root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install pandas matplotlib vaderSentiment
PYTHONPATH=. .venv/bin/python src/pipeline/run_vader_sentiment.py
```

Then generate an example plot:

```bash
PYTHONPATH=. .venv/bin/python src/pipeline/plot_vader_timeseries.py --all_candidates --add_scatter
```

## 1. Environment Setup

From the project root, create and activate a virtual environment, then install dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas matplotlib vaderSentiment
```

All pipeline commands below should be run from the project root with `PYTHONPATH=.`.

## 2. Run the Full Pipeline

### Step A: October 2025 keyword pipeline

```bash
PYTHONPATH=. .venv/bin/python src/pipeline/run_oct2025_keywords.py
```

Main outputs:

- `outputs/tables/oct2025_keyword_daily_wide.csv`
- `outputs/tables/oct2025_keyword_daily_long.csv`
- `outputs/figures/oct2025_keywords_counts.png`
- `outputs/figures/oct2025_keywords_counts_debate_days.png`
- `outputs/figures/oct2025_keywords_rates_per_1000.png`

### Step B: VADER sentiment + topic/candidate/relevance tables

```bash
PYTHONPATH=. .venv/bin/python src/pipeline/run_vader_sentiment.py
```

Core outputs include:

- Base sentiment: `comment_vader_scored.csv`, `sentiment_daily.csv`, `sentiment_yearly.csv`
- Topic flags: `comment_topic_flags.csv`, `comment_topic_high_recall.csv`, `comment_topic_higher_precision.csv`
- Variant daily: `sentiment_daily_candidate_only.csv`, `sentiment_daily_election_only.csv`, `sentiment_daily_high_recall.csv`, `sentiment_daily_higher_precision.csv`
- Candidate daily: `sentiment_daily_cuomo.csv`, `sentiment_daily_mamdani.csv`, `sentiment_daily_sliwa.csv`, `sentiment_daily_by_candidate.csv`
- Relevance daily: `sentiment_daily_explicit_comment_relevant.csv`, `sentiment_daily_thread_relevant_only.csv`, `sentiment_daily_merged_relevant.csv`, `sentiment_daily_by_relevance_type.csv`

All tables are written to:

- `outputs/tables/`

## 3. Generate Plots

Plot script:

- `src/pipeline/plot_vader_timeseries.py`

### A) Topic filter variants

```bash
# single
PYTHONPATH=. .venv/bin/python src/pipeline/plot_vader_timeseries.py --variant high_recall

# all variants + comparison
PYTHONPATH=. .venv/bin/python src/pipeline/plot_vader_timeseries.py --all_variants
```

### B) Candidate plots

```bash
# single candidate
PYTHONPATH=. .venv/bin/python src/pipeline/plot_vader_timeseries.py --candidate cuomo

# all candidates + comparison
PYTHONPATH=. .venv/bin/python src/pipeline/plot_vader_timeseries.py --all_candidates

# add scatter outputs
PYTHONPATH=. .venv/bin/python src/pipeline/plot_vader_timeseries.py --all_candidates --add_scatter
```

### C) Relevance plots

```bash
# single relevance type
PYTHONPATH=. .venv/bin/python src/pipeline/plot_vader_timeseries.py --relevance merged_relevant

# all relevance types + comparison
PYTHONPATH=. .venv/bin/python src/pipeline/plot_vader_timeseries.py --all_relevance
```

### D) Candidate + relevance plots

```bash
# one relevance bucket for all candidates
PYTHONPATH=. .venv/bin/python src/pipeline/plot_vader_timeseries.py --candidate_relevance merged_relevant

# all relevance buckets for all candidates
PYTHONPATH=. .venv/bin/python src/pipeline/plot_vader_timeseries.py --all_candidate_relevance
```

## 4. Date-Windowed Plots

You can constrain any plotting mode with inclusive date bounds:

- `--start_date YYYY-MM-DD`
- `--end_date YYYY-MM-DD`

Example:

```bash
PYTHONPATH=. .venv/bin/python src/pipeline/plot_vader_timeseries.py \
  --all_candidates --add_scatter \
  --start_date 2025-06-01 --end_date 2025-11-30
```

Windowed plots are saved with a suffix to avoid overwriting full-range files, e.g.:

- `..._2025-06-01_to_2025-11-30.png`

## 5. Notes

- Timezone handling in the pipeline uses `America/New_York`.
- If a chosen date window has no rows, the plotting script prints a skip message and does not write a figure for that series.
- If `start_date > end_date`, plotting exits with a clear error.
