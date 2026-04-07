from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from src.pipeline.io import detect_schema, load_csv
from src.pipeline.sentiment import aggregate_sentiment, score_comments_vader
from src.pipeline.time_filter import parse_timestamp_to_tz
from src.pipeline.topic_filter import CANDIDATE_CANONICAL, add_candidate_match_flags, apply_topic_filters


VARIANTS = [
    'candidate_only',
    'election_only',
    'high_recall',
    'higher_precision',
]

RELEVANCE_TYPES = [
    'explicit_comment_relevant',
    'thread_relevant_only',
    'merged_relevant',
]

POST_BODY_ALLOWLIST = ['selftext', 'post_body', 'submission_body']


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run VADER sentiment pipeline with topic-filtered variant outputs.')
    parser.add_argument(
        '--variant',
        choices=['all'] + VARIANTS,
        default='all',
        help='Filter variant to export. Default writes all variants.',
    )
    return parser.parse_args()


def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _variant_mask(df, variant: str):
    mask_map = {
        'candidate_only': df['filter_candidate_only'],
        'election_only': df['filter_election_only'],
        'high_recall': df['filter_high_recall'],
        'higher_precision': df['filter_higher_precision'],
    }
    return mask_map[variant]


def _detect_post_body_col(columns: list[str]) -> str | None:
    normalized = {str(col).lower(): str(col) for col in columns}
    for candidate in POST_BODY_ALLOWLIST:
        if candidate in normalized:
            return normalized[candidate]
    return None


def main() -> None:
    args = _parse_args()

    workspace = Path('/Users/ajoji/Desktop/2025-2026/CS598_CSS')
    csv_path = workspace / 'filtered_reddit_data.csv'

    text_col = 'body'
    timestamp_col = 'created_utc_comment'

    df = load_csv(csv_path)
    detected = detect_schema(df)

    if text_col not in df.columns:
        raise ValueError(f"Configured text column '{text_col}' not found in CSV")
    if timestamp_col not in df.columns:
        raise ValueError(f"Configured timestamp column '{timestamp_col}' not found in CSV")
    if 'title' not in df.columns:
        raise ValueError("Configured post title column 'title' not found in CSV")

    post_body_col = _detect_post_body_col(list(df.columns))

    topic_df, topic_stats = apply_topic_filters(
        df,
        body_col=text_col,
        title_col='title',
        post_body_col=post_body_col,
        word_boundary=True,
    )

    scored_df, meta = score_comments_vader(
        topic_df,
        text_col=text_col,
        timestamp_col=timestamp_col,
        timezone='America/New_York',
    )

    topic_cols = [
        'candidate_match',
        'election_match',
        'filter_candidate_only',
        'filter_election_only',
        'filter_high_recall',
        'filter_higher_precision',
        'matched_candidates',
        'matched_election_terms',
        'matched_in_body',
        'matched_in_title',
        'matched_in_post_body',
        'post_relevant_any',
        'explicit_comment_relevant',
        'thread_relevant_only',
        'merged_relevant',
    ]
    scored_df = scored_df.join(topic_df.loc[scored_df.index, topic_cols])
    scored_df = add_candidate_match_flags(
        scored_df,
        matched_candidates_col='matched_candidates',
        candidates=CANDIDATE_CANONICAL,
    )

    daily_df = aggregate_sentiment(scored_df, group_col='date')
    yearly_df = aggregate_sentiment(scored_df, group_col='year')

    tables_dir = workspace / 'outputs' / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)

    comment_out = tables_dir / 'comment_vader_scored.csv'
    daily_out = tables_dir / 'sentiment_daily.csv'
    yearly_out = tables_dir / 'sentiment_yearly.csv'
    topic_flags_out = tables_dir / 'comment_topic_flags.csv'
    topic_high_recall_out = tables_dir / 'comment_topic_high_recall.csv'
    topic_higher_precision_out = tables_dir / 'comment_topic_higher_precision.csv'
    topic_daily_counts_out = tables_dir / 'topic_match_daily_counts.csv'
    candidate_daily_by_out = tables_dir / 'sentiment_daily_by_candidate.csv'
    candidate_yearly_by_out = tables_dir / 'sentiment_yearly_by_candidate.csv'
    relevance_daily_by_out = tables_dir / 'sentiment_daily_by_relevance_type.csv'

    selected_variants = VARIANTS if args.variant == 'all' else [args.variant]

    unfiltered_pre_hash = {
        'comment_vader_scored.csv': _file_hash(comment_out),
        'sentiment_daily.csv': _file_hash(daily_out),
        'sentiment_yearly.csv': _file_hash(yearly_out),
    }

    scored_df.to_csv(comment_out, index=False)
    daily_df.to_csv(daily_out, index=False)
    yearly_df.to_csv(yearly_out, index=False)

    unfiltered_post_hash = {
        'comment_vader_scored.csv': _file_hash(comment_out),
        'sentiment_daily.csv': _file_hash(daily_out),
        'sentiment_yearly.csv': _file_hash(yearly_out),
    }

    topic_export_cols = [
        'title',
        text_col,
        timestamp_col,
    ] + topic_cols
    if post_body_col is not None and post_body_col not in topic_export_cols:
        topic_export_cols.insert(2, post_body_col)
    topic_df[topic_export_cols].to_csv(topic_flags_out, index=False)
    topic_df.loc[topic_df['filter_high_recall'], topic_export_cols].to_csv(topic_high_recall_out, index=False)
    topic_df.loc[topic_df['filter_higher_precision'], topic_export_cols].to_csv(topic_higher_precision_out, index=False)

    parsed_dt, _ = parse_timestamp_to_tz(topic_df[timestamp_col], timezone='America/New_York')
    topic_daily = topic_df.copy()
    topic_daily['parsed_datetime'] = parsed_dt
    topic_daily = topic_daily.dropna(subset=['parsed_datetime']).copy()
    topic_daily['date'] = topic_daily['parsed_datetime'].dt.strftime('%Y-%m-%d')
    topic_match_daily = (
        topic_daily.groupby('date', as_index=False)
        .agg(
            n_comments=(text_col, 'size'),
            candidate_match=('candidate_match', 'sum'),
            election_match=('election_match', 'sum'),
            filter_high_recall=('filter_high_recall', 'sum'),
            filter_higher_precision=('filter_higher_precision', 'sum'),
        )
    )
    topic_match_daily.to_csv(topic_daily_counts_out, index=False)

    candidate_outputs = []
    candidate_daily_frames = []
    candidate_yearly_frames = []
    multi_candidate_count = int((scored_df['matched_candidate_count'] > 1).sum())
    for candidate in CANDIDATE_CANONICAL:
        candidate_col = f'match_{candidate}'
        candidate_scored = scored_df.loc[scored_df[candidate_col]].copy()
        candidate_daily = aggregate_sentiment(candidate_scored, group_col='date') if len(candidate_scored) else aggregate_sentiment(scored_df.head(0), group_col='date')
        candidate_yearly = aggregate_sentiment(candidate_scored, group_col='year') if len(candidate_scored) else aggregate_sentiment(scored_df.head(0), group_col='year')

        candidate_daily_out = tables_dir / f'sentiment_daily_{candidate}.csv'
        candidate_yearly_out = tables_dir / f'sentiment_yearly_{candidate}.csv'
        candidate_scored_out = tables_dir / f'comment_vader_scored_{candidate}.csv'

        candidate_daily.to_csv(candidate_daily_out, index=False)
        candidate_yearly.to_csv(candidate_yearly_out, index=False)
        candidate_scored.to_csv(candidate_scored_out, index=False)

        candidate_daily_with_name = candidate_daily.copy()
        candidate_daily_with_name['candidate'] = candidate
        candidate_daily_frames.append(candidate_daily_with_name)

        candidate_yearly_with_name = candidate_yearly.copy()
        candidate_yearly_with_name['candidate'] = candidate
        candidate_yearly_frames.append(candidate_yearly_with_name)

        row_count = int(len(candidate_scored))
        daily_sum = int(candidate_daily['n_comments'].sum()) if len(candidate_daily) else 0
        yearly_sum = int(candidate_yearly['n_comments'].sum()) if len(candidate_yearly) else 0
        date_min = candidate_scored['date'].min() if row_count else 'N/A'
        date_max = candidate_scored['date'].max() if row_count else 'N/A'

        candidate_outputs.append({
            'candidate': candidate,
            'row_count': row_count,
            'daily_sum': daily_sum,
            'yearly_sum': yearly_sum,
            'date_min': date_min,
            'date_max': date_max,
            'daily_out': candidate_daily_out,
            'yearly_out': candidate_yearly_out,
            'scored_out': candidate_scored_out,
            'sample': candidate_scored[['body', 'date', 'year', 'compound', 'matched_candidates']].head(5),
        })

    if candidate_daily_frames:
        pd.concat(candidate_daily_frames, ignore_index=True).to_csv(candidate_daily_by_out, index=False)
    else:
        aggregate_sentiment(scored_df.head(0), group_col='date').assign(candidate='').to_csv(candidate_daily_by_out, index=False)

    if candidate_yearly_frames:
        pd.concat(candidate_yearly_frames, ignore_index=True).to_csv(candidate_yearly_by_out, index=False)
    else:
        aggregate_sentiment(scored_df.head(0), group_col='year').assign(candidate='').to_csv(candidate_yearly_by_out, index=False)

    relevance_outputs = []
    relevance_daily_frames = []
    for relevance_type in RELEVANCE_TYPES:
        relevance_mask = scored_df[relevance_type]
        relevance_scored = scored_df.loc[relevance_mask].copy()
        relevance_daily = aggregate_sentiment(relevance_scored, group_col='date') if len(relevance_scored) else aggregate_sentiment(scored_df.head(0), group_col='date')
        relevance_daily_out = tables_dir / f'sentiment_daily_{relevance_type}.csv'
        relevance_daily.to_csv(relevance_daily_out, index=False)

        relevance_daily_with_name = relevance_daily.copy()
        relevance_daily_with_name['relevance_type'] = relevance_type
        relevance_daily_frames.append(relevance_daily_with_name)

        row_count = int(len(relevance_scored))
        daily_sum = int(relevance_daily['n_comments'].sum()) if len(relevance_daily) else 0
        date_min = relevance_scored['date'].min() if row_count else 'N/A'
        date_max = relevance_scored['date'].max() if row_count else 'N/A'
        relevance_outputs.append({
            'relevance_type': relevance_type,
            'row_count': row_count,
            'daily_sum': daily_sum,
            'date_min': date_min,
            'date_max': date_max,
            'daily_out': relevance_daily_out,
        })

    if relevance_daily_frames:
        pd.concat(relevance_daily_frames, ignore_index=True).to_csv(relevance_daily_by_out, index=False)
    else:
        aggregate_sentiment(scored_df.head(0), group_col='date').assign(relevance_type='').to_csv(relevance_daily_by_out, index=False)

    variant_outputs = []
    sample_cols_variant = [
        'body',
        'date',
        'year',
        'sentiment_label',
        'compound',
        'matched_candidates',
        'matched_election_terms',
    ]
    for variant in selected_variants:
        mask = _variant_mask(scored_df, variant)
        filtered_scored = scored_df.loc[mask].copy()
        filtered_daily = aggregate_sentiment(filtered_scored, group_col='date') if len(filtered_scored) else aggregate_sentiment(scored_df.head(0), group_col='date')
        filtered_yearly = aggregate_sentiment(filtered_scored, group_col='year') if len(filtered_scored) else aggregate_sentiment(scored_df.head(0), group_col='year')

        variant_comment_out = tables_dir / f'comment_vader_scored_{variant}.csv'
        variant_daily_out = tables_dir / f'sentiment_daily_{variant}.csv'
        variant_yearly_out = tables_dir / f'sentiment_yearly_{variant}.csv'

        filtered_scored.to_csv(variant_comment_out, index=False)
        filtered_daily.to_csv(variant_daily_out, index=False)
        filtered_yearly.to_csv(variant_yearly_out, index=False)

        row_count = int(len(filtered_scored))
        daily_sum = int(filtered_daily['n_comments'].sum()) if len(filtered_daily) else 0
        yearly_sum = int(filtered_yearly['n_comments'].sum()) if len(filtered_yearly) else 0
        date_min = filtered_scored['date'].min() if row_count else 'N/A'
        date_max = filtered_scored['date'].max() if row_count else 'N/A'
        year_min = int(filtered_scored['year'].min()) if row_count else 'N/A'
        year_max = int(filtered_scored['year'].max()) if row_count else 'N/A'

        variant_outputs.append({
            'variant': variant,
            'row_count': row_count,
            'daily_sum': daily_sum,
            'yearly_sum': yearly_sum,
            'date_min': date_min,
            'date_max': date_max,
            'year_min': year_min,
            'year_max': year_max,
            'comment_out': variant_comment_out,
            'daily_out': variant_daily_out,
            'yearly_out': variant_yearly_out,
            'sample': filtered_scored[sample_cols_variant].head(5),
        })

    print('Detected text/timestamp columns from schema detector:', detected['text_col'], detected['timestamp_col'])
    print('Configured text/timestamp columns used:', text_col, timestamp_col)
    print('Detected post_body_col:', post_body_col)
    print('Variant run mode:', args.variant)
    print('candidate_only / election_only interpreted as inclusive flags.')
    print('\nTOPIC FILTER COUNTS:')
    print('total comments:', topic_stats['total_comments'])
    print('candidate_match:', topic_stats['candidate_match'])
    print('election_match:', topic_stats['election_match'])
    print('filter_high_recall:', topic_stats['filter_high_recall'])
    print('filter_higher_precision:', topic_stats['filter_higher_precision'])

    print('\nPOST-DROP SCORED TOPIC COUNTS:')
    print('scored rows:', int(len(scored_df)))
    print('candidate_match:', int(scored_df['candidate_match'].sum()))
    print('election_match:', int(scored_df['election_match'].sum()))
    print('filter_high_recall:', int(scored_df['filter_high_recall'].sum()))
    print('filter_higher_precision:', int(scored_df['filter_higher_precision'].sum()))
    print('matched_in_body:', int(scored_df['matched_in_body'].sum()))
    print('matched_in_title:', int(scored_df['matched_in_title'].sum()))
    print('matched_in_post_body:', int(scored_df['matched_in_post_body'].sum()))
    print('explicit_comment_relevant:', int(scored_df['explicit_comment_relevant'].sum()))
    print('post_relevant_any:', int(scored_df['post_relevant_any'].sum()))
    print('thread_relevant_only:', int(scored_df['thread_relevant_only'].sum()))
    print('merged_relevant:', int(scored_df['merged_relevant'].sum()))
    for candidate in CANDIDATE_CANONICAL:
        print(f'match_{candidate}:', int(scored_df[f'match_{candidate}'].sum()))
    print('multi-candidate overlap count:', multi_candidate_count)

    consistency_explicit = bool((scored_df['explicit_comment_relevant'] == scored_df['matched_in_body']).all())
    consistency_post_any = bool((scored_df['post_relevant_any'] == (scored_df['matched_in_title'] | scored_df['matched_in_post_body'])).all())
    consistency_thread_implies_merged = bool((~scored_df['thread_relevant_only'] | scored_df['merged_relevant']).all())
    consistency_merged = bool((scored_df['merged_relevant'] == (scored_df['explicit_comment_relevant'] | scored_df['post_relevant_any'])).all())
    print('\nMERGED RELEVANCE CONSISTENCY CHECKS:')
    print('explicit_comment_relevant == matched_in_body:', consistency_explicit)
    print('post_relevant_any == matched_in_title OR matched_in_post_body:', consistency_post_any)
    print('thread_relevant_only implies merged_relevant:', consistency_thread_implies_merged)
    print('merged_relevant == explicit_comment_relevant OR post_relevant_any:', consistency_merged)

    overlap = (
        scored_df.groupby(['explicit_comment_relevant', 'thread_relevant_only'], dropna=False)
        .size()
        .reset_index(name='n_rows')
        .sort_values(['explicit_comment_relevant', 'thread_relevant_only'])
    )
    print('\nOVERLAP SUMMARY (explicit_comment_relevant x thread_relevant_only):')
    print(overlap.to_string(index=False))

    print('Dropped invalid text rows:', meta['dropped_invalid_text_count'])
    print('Parse failure count (dropped invalid parsed datetimes):', meta['dropped_invalid_datetime_count'])
    print('Scored row count:', meta['scored_row_count'])
    print('Date range:', scored_df['date'].min(), 'to', scored_df['date'].max())
    print('Year range:', int(scored_df['year'].min()), 'to', int(scored_df['year'].max()))

    sample_cols = ['title', text_col, 'matched_candidates', 'matched_election_terms']
    print('\nSAMPLE - candidate_only (first 5):')
    print(
        topic_df.loc[topic_df['filter_candidate_only'], sample_cols]
        .head(5)
        .to_string(index=False)
    )

    print('\nSAMPLE - election_only (first 5):')
    print(
        topic_df.loc[topic_df['filter_election_only'], sample_cols]
        .head(5)
        .to_string(index=False)
    )

    print('\nSAMPLE - high_recall (first 5):')
    print(
        topic_df.loc[topic_df['filter_high_recall'], sample_cols]
        .head(5)
        .to_string(index=False)
    )

    print('\nSAMPLE - higher_precision (first 5):')
    print(
        topic_df.loc[topic_df['filter_higher_precision'], sample_cols]
        .head(5)
        .to_string(index=False)
    )

    print('\nTITLE-ONLY MATCH EXAMPLES (first 5):')
    print(
        topic_df.loc[(topic_df['matched_in_title']) & (~topic_df['matched_in_body']), sample_cols]
        .head(5)
        .to_string(index=False)
    )

    merged_sample_cols = ['title', text_col, 'matched_in_body', 'matched_in_title', 'matched_in_post_body', 'explicit_comment_relevant', 'thread_relevant_only', 'merged_relevant']
    if post_body_col is not None and post_body_col not in merged_sample_cols:
        merged_sample_cols.insert(2, post_body_col)

    print('\nSAMPLE - explicit_comment_relevant (first 5):')
    print(
        topic_df.loc[topic_df['explicit_comment_relevant'], merged_sample_cols]
        .head(5)
        .to_string(index=False)
    )

    print('\nSAMPLE - thread_relevant_only (first 5):')
    print(
        topic_df.loc[topic_df['thread_relevant_only'], merged_sample_cols]
        .head(5)
        .to_string(index=False)
    )

    print('\nSAMPLE - matched_in_post_body (first 5):')
    print(
        topic_df.loc[topic_df['matched_in_post_body'], merged_sample_cols]
        .head(5)
        .to_string(index=False)
    )

    print('\nAMBIGUITY CHECKS:')
    cuomo_only = topic_df['matched_candidates'].str.contains(r'(?:^|;)cuomo(?:;|$)', regex=True)
    generic_election = topic_df['matched_election_terms'].str.contains(r'campaign|candidate|poll|rcv', regex=True)
    print('rows mentioning cuomo candidate:', int(cuomo_only.sum()))
    print('rows matching generic election terms (campaign/candidate/poll/rcv):', int(generic_election.sum()))
    print('rows with rcv term:', int(topic_df['matched_election_terms'].str.contains(r'(?:^|;)rcv(?:;|$)', regex=True).sum()))

    print('\nHEAD(5) - comment_vader_scored:')
    print(scored_df.head(5).to_string(index=False))

    print('\nHEAD(5) - sentiment_daily:')
    print(daily_df.head(5).to_string(index=False))

    print('\nHEAD(5) - sentiment_yearly:')
    print(yearly_df.head(5).to_string(index=False))

    print('\nWROTE FILES:')
    print(comment_out)
    print(daily_out)
    print(yearly_out)
    print(topic_flags_out)
    print(topic_high_recall_out)
    print(topic_higher_precision_out)
    print(topic_daily_counts_out)
    print(candidate_daily_by_out)
    print(candidate_yearly_by_out)
    print(relevance_daily_by_out)

    exported_topic_cols = pd.read_csv(topic_flags_out, nrows=0).columns.tolist()
    required_new_topic_cols = [
        'matched_in_post_body',
        'post_relevant_any',
        'explicit_comment_relevant',
        'thread_relevant_only',
        'merged_relevant',
    ]
    missing_new_cols = [col for col in required_new_topic_cols if col not in exported_topic_cols]
    print('\nTOPIC FLAGS EXPORT COLUMN CHECK:')
    print('new columns present:', len(missing_new_cols) == 0)
    print('missing new columns:', missing_new_cols)

    print('\nUNFILTERED FILE CHANGE CHECK (pre vs post hash):')
    for name in ['comment_vader_scored.csv', 'sentiment_daily.csv', 'sentiment_yearly.csv']:
        before = unfiltered_pre_hash[name]
        after = unfiltered_post_hash[name]
        if before is None:
            status = 'new_file_written'
        else:
            status = 'unchanged' if before == after else 'changed'
        print(name, status)

    print('\nVARIANT OUTPUT CHECKS:')
    for item in variant_outputs:
        print(f"[{item['variant']}] rows={item['row_count']} daily_sum={item['daily_sum']} yearly_sum={item['yearly_sum']} checks={{row_vs_daily:{item['row_count']==item['daily_sum']}, row_vs_yearly:{item['row_count']==item['yearly_sum']}}}")
        print(f"[{item['variant']}] date_range={item['date_min']} to {item['date_max']} year_range={item['year_min']} to {item['year_max']}")
        print(f"[{item['variant']}] files={item['comment_out']}, {item['daily_out']}, {item['yearly_out']}")
        print(f"[{item['variant']}] sample rows:")
        print(item['sample'].to_string(index=False))

    print('\nRELEVANCE OUTPUT CHECKS:')
    for item in relevance_outputs:
        print(f"[{item['relevance_type']}] rows={item['row_count']} daily_sum={item['daily_sum']} check={{row_vs_daily:{item['row_count']==item['daily_sum']}}}")
        print(f"[{item['relevance_type']}] date_range={item['date_min']} to {item['date_max']}")
        print(f"[{item['relevance_type']}] file={item['daily_out']}")

    merged_rows = int(scored_df['merged_relevant'].sum())
    high_recall_rows = int(scored_df['filter_high_recall'].sum())
    print('merged_relevant matches scored full merged topic series (filter_high_recall):', merged_rows == high_recall_rows)
    print('merged_relevant_rows:', merged_rows, 'filter_high_recall_rows:', high_recall_rows)

    print('\nCANDIDATE OUTPUT CHECKS:')
    for item in candidate_outputs:
        print(f"[{item['candidate']}] rows={item['row_count']} daily_sum={item['daily_sum']} yearly_sum={item['yearly_sum']} checks={{row_vs_daily:{item['row_count']==item['daily_sum']}, row_vs_yearly:{item['row_count']==item['yearly_sum']}}}")
        print(f"[{item['candidate']}] date_range={item['date_min']} to {item['date_max']}")
        print(f"[{item['candidate']}] files={item['daily_out']}, {item['yearly_out']}, {item['scored_out']}")
        print(f"[{item['candidate']}] sample rows:")
        print(item['sample'].to_string(index=False))


if __name__ == '__main__':
    main()
