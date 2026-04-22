from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.pipeline.io import detect_schema, load_csv
from src.pipeline.keywords import daily_keyword_counts, normalize_per_1000
from src.pipeline.time_filter import add_date_parts, filter_date_range, parse_timestamp_to_tz


def main() -> None:
    workspace = Path('/Users/ajoji/Desktop/2025-2026/CS598_CSS')
    csv_path = workspace / 'filtered_reddit_data.csv'

    keywords = ['debate', 'mamdani', 'sliwa', 'cuomo', 'election', 'policy', 'economy', 'healthcare', 'education', 'climate', 'scandal']

    df = load_csv(csv_path)
    schema = detect_schema(df)

    text_col = schema['text_col']
    timestamp_col = schema['timestamp_col']

    if not text_col or not timestamp_col:
        raise ValueError(f'Could not confidently detect required schema fields: {schema}')

    parsed_dt, ts_type = parse_timestamp_to_tz(df[timestamp_col], timezone='America/New_York')
    df['parsed_datetime'] = parsed_dt
    df = add_date_parts(df, datetime_col='parsed_datetime')

    oct_df = filter_date_range(
        df,
        datetime_col='parsed_datetime',
        start='2025-10-01 00:00:00',
        end='2025-10-31 23:59:59.999999',
        timezone='America/New_York',
    )

    daily_wide, daily_long = daily_keyword_counts(
        oct_df,
        text_col=text_col,
        date_col='date',
        keywords=keywords,
    )

    all_dates = pd.date_range('2025-10-01', '2025-10-31', freq='D').strftime('%Y-%m-%d')
    all_dates_df = pd.DataFrame({'date': all_dates})

    daily_wide = all_dates_df.merge(daily_wide, on='date', how='left')
    daily_wide['total_comments'] = daily_wide['total_comments'].fillna(0).astype(int)
    for keyword in keywords:
        col = f'count_{keyword}'
        daily_wide[col] = daily_wide[col].fillna(0).astype(int)

    daily_wide = normalize_per_1000(daily_wide)

    count_cols = [f'count_{keyword}' for keyword in keywords]
    daily_long = daily_wide[['date'] + count_cols].melt(
        id_vars=['date'],
        var_name='keyword',
        value_name='count',
    )
    daily_long['keyword'] = daily_long['keyword'].str.replace('count_', '', regex=False)
    daily_long['count'] = daily_long['count'].astype(int)

    tables_dir = workspace / 'outputs' / 'tables'
    figures_dir = workspace / 'outputs' / 'figures'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    wide_out = tables_dir / 'oct2025_keyword_daily_wide.csv'
    long_out = tables_dir / 'oct2025_keyword_daily_long.csv'
    daily_wide.to_csv(wide_out, index=False)
    daily_long.to_csv(long_out, index=False)

    plt.figure(figsize=(12, 6))
    for keyword in keywords:
        plt.plot(daily_wide['date'], daily_wide[f'count_{keyword}'], marker='o', linewidth=1.5, label=keyword)
    plt.title('October 2025 Keyword Counts per Day')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / 'oct2025_keywords_counts.png', dpi=150)
    plt.close()

    x_dates = pd.to_datetime(daily_wide['date'])
    plt.figure(figsize=(12, 6))
    for keyword in keywords:
        plt.plot(x_dates, daily_wide[f'count_{keyword}'], marker='o', linewidth=1.5, label=keyword)

    debate_days = [
        pd.Timestamp('2025-10-16'),
        pd.Timestamp('2025-10-22'),
    ]
    for i, debate_day in enumerate(debate_days):
        line_label = 'Debate Day' if i == 0 else None
        plt.axvline(debate_day, color='black', linestyle='--', linewidth=1.5, alpha=0.8, label=line_label)

    max_count = max(1, int(daily_wide[[f'count_{keyword}' for keyword in keywords]].to_numpy().max()))
    for debate_day in debate_days:
        plt.text(
            debate_day,
            max_count * 0.97,
            f'Debate Day\n{debate_day.strftime("%b %d")}',
            rotation=90,
            va='top',
            ha='right',
            fontsize=9,
            color='black',
        )

    plt.title('October 2025 Keyword Counts per Day (Debate Days Marked)')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / 'oct2025_keywords_counts_debate_days.png', dpi=150)
    plt.close()

    plt.figure(figsize=(12, 6))
    for keyword in keywords:
        plt.plot(
            daily_wide['date'],
            daily_wide[f'rate_{keyword}_per_1000'],
            marker='o',
            linewidth=1.5,
            label=keyword,
        )
    plt.title('October 2025 Keyword Rates per 1000 Comments')
    plt.xlabel('Date')
    plt.ylabel('Rate per 1000 comments')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / 'oct2025_keywords_rates_per_1000.png', dpi=150)
    plt.close()

    print('Loaded:', csv_path)
    print('Detected text_col:', text_col)
    print('Detected timestamp_col:', timestamp_col)
    print('Detected timestamp_type:', ts_type)
    print('October rows:', len(oct_df))
    print('Saved:', wide_out)
    print('Saved:', long_out)
    print('Saved:', figures_dir / 'oct2025_keywords_counts.png')
    print('Saved:', figures_dir / 'oct2025_keywords_counts_debate_days.png')
    print('Saved:', figures_dir / 'oct2025_keywords_rates_per_1000.png')


if __name__ == '__main__':
    main()
