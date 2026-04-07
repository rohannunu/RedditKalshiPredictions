from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd
from src.pipeline.sentiment import aggregate_sentiment

matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


VARIANT_TO_DAILY_FILE = {
    'candidate_only': 'sentiment_daily_candidate_only.csv',
    'election_only': 'sentiment_daily_election_only.csv',
    'high_recall': 'sentiment_daily_high_recall.csv',
    'higher_precision': 'sentiment_daily_higher_precision.csv',
}

CANDIDATE_TO_DAILY_FILE = {
    'cuomo': 'sentiment_daily_cuomo.csv',
    'mamdani': 'sentiment_daily_mamdani.csv',
    'sliwa': 'sentiment_daily_sliwa.csv',
}

CANDIDATE_COLORS = {
    'mamdani': 'tab:blue',
    'cuomo': 'tab:green',
    'sliwa': 'tab:red',
}

RELEVANCE_TO_DAILY_FILE = {
    'explicit_comment_relevant': 'sentiment_daily_explicit_comment_relevant.csv',
    'thread_relevant_only': 'sentiment_daily_thread_relevant_only.csv',
    'merged_relevant': 'sentiment_daily_merged_relevant.csv',
}

CANDIDATE_RELEVANCE_TYPES = list(RELEVANCE_TO_DAILY_FILE.keys())


def _apply_readable_date_axis(ax: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def _candidate_color(candidate: str) -> str:
    return CANDIDATE_COLORS.get(candidate, 'tab:blue')


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot year-long VADER time series for filtered variants.')
    parser.add_argument(
        '--variant',
        choices=list(VARIANT_TO_DAILY_FILE.keys()),
        help='Single filter variant to plot.',
    )
    parser.add_argument(
        '--all_variants',
        action='store_true',
        help='Plot all filter variants and generate a comparison chart.',
    )
    parser.add_argument(
        '--candidate',
        choices=list(CANDIDATE_TO_DAILY_FILE.keys()),
        help='Single candidate series to plot.',
    )
    parser.add_argument(
        '--all_candidates',
        action='store_true',
        help='Plot all candidate series and generate candidate comparison chart.',
    )
    parser.add_argument(
        '--relevance',
        choices=list(RELEVANCE_TO_DAILY_FILE.keys()),
        help='Single relevance-type series to plot.',
    )
    parser.add_argument(
        '--all_relevance',
        action='store_true',
        help='Plot all relevance-type series and generate relevance comparison chart.',
    )
    parser.add_argument(
        '--candidate_relevance',
        choices=CANDIDATE_RELEVANCE_TYPES,
        help='Generate candidate-specific plots for one relevance type from scored comment-level data.',
    )
    parser.add_argument(
        '--all_candidate_relevance',
        action='store_true',
        help='Generate candidate-specific plots for all relevance types from scored comment-level data.',
    )
    parser.add_argument(
        '--start_date',
        type=str,
        help='Optional inclusive start date (YYYY-MM-DD) for plotting window.',
    )
    parser.add_argument(
        '--end_date',
        type=str,
        help='Optional inclusive end date (YYYY-MM-DD) for plotting window.',
    )
    parser.add_argument(
        '--add_scatter',
        action='store_true',
        help='Also generate scatter-based visualizations (in addition to existing line plots).',
    )
    parser.add_argument(
        '--line_scatter_overlay',
        action='store_true',
        help='When scatter is enabled, also generate line+scatter overlay plots for mean compound.',
    )
    return parser.parse_args()


def _load_daily_table(daily_path: Path) -> pd.DataFrame:
    daily_df = pd.read_csv(daily_path)
    daily_df['date'] = pd.to_datetime(daily_df['date'], errors='coerce')
    daily_df = daily_df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    return daily_df


def _parse_date_window(start_date: str | None, end_date: str | None) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str]:
    start_ts = pd.to_datetime(start_date, errors='coerce') if start_date else None
    end_ts = pd.to_datetime(end_date, errors='coerce') if end_date else None

    if start_date and pd.isna(start_ts):
        raise ValueError(f"Invalid --start_date '{start_date}'. Expected format YYYY-MM-DD.")
    if end_date and pd.isna(end_ts):
        raise ValueError(f"Invalid --end_date '{end_date}'. Expected format YYYY-MM-DD.")
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        raise ValueError(
            f"Invalid date window: start_date ({start_ts.strftime('%Y-%m-%d')}) is after end_date ({end_ts.strftime('%Y-%m-%d')})."
        )

    if start_ts is None and end_ts is None:
        suffix = ''
    else:
        start_label = start_ts.strftime('%Y-%m-%d') if start_ts is not None else 'min'
        end_label = end_ts.strftime('%Y-%m-%d') if end_ts is not None else 'max'
        suffix = f'_{start_label}_to_{end_label}'

    return start_ts, end_ts, suffix


def _apply_date_window(
    daily_df: pd.DataFrame,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> pd.DataFrame:
    filtered = daily_df.copy()
    if start_ts is not None:
        filtered = filtered.loc[filtered['date'] >= start_ts]
    if end_ts is not None:
        filtered = filtered.loc[filtered['date'] <= end_ts]
    return filtered.sort_values('date').reset_index(drop=True)


def _plot_variant(daily_df: pd.DataFrame, figures_dir: Path, variant: str, output_suffix: str = '') -> tuple[Path, Path, Path]:
    compound_fig = figures_dir / f'mean_compound_over_time_{variant}{output_suffix}.png'
    pct_fig = figures_dir / f'sentiment_shares_over_time_{variant}{output_suffix}.png'
    volume_fig = figures_dir / f'comment_volume_over_time_{variant}{output_suffix}.png'

    plt.figure(figsize=(12, 5.5))
    plt.plot(daily_df['date'], daily_df['mean_compound'], linewidth=1.8, color='tab:blue')
    plt.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    plt.title(f'Daily Mean Compound Sentiment ({variant})')
    plt.xlabel('Date')
    plt.ylabel('Mean compound score')
    _apply_readable_date_axis(plt.gca())
    plt.tight_layout()
    plt.savefig(compound_fig, dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5.5))
    plt.plot(daily_df['date'], daily_df['pct_positive'], linewidth=1.6, label='pct_positive')
    plt.plot(daily_df['date'], daily_df['pct_negative'], linewidth=1.6, label='pct_negative')
    plt.plot(daily_df['date'], daily_df['pct_neutral'], linewidth=1.6, label='pct_neutral')
    plt.title(f'Daily Sentiment Shares ({variant})')
    plt.xlabel('Date')
    plt.ylabel('Percent of comments')
    _apply_readable_date_axis(plt.gca())
    plt.legend()
    plt.tight_layout()
    plt.savefig(pct_fig, dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5.5))
    plt.plot(daily_df['date'], daily_df['n_comments'], linewidth=1.8, color='tab:purple')
    plt.title(f'Daily Comment Volume ({variant})')
    plt.xlabel('Date')
    plt.ylabel('Number of comments')
    _apply_readable_date_axis(plt.gca())
    plt.tight_layout()
    plt.savefig(volume_fig, dpi=150)
    plt.close()

    return compound_fig, pct_fig, volume_fig


def _plot_variant_scatter(
    daily_df: pd.DataFrame,
    figures_dir: Path,
    variant: str,
    include_overlay: bool,
    output_suffix: str = '',
) -> tuple[Path, Path, Path | None]:
    compound_scatter_fig = figures_dir / f'mean_compound_scatter_over_time_{variant}{output_suffix}.png'
    pct_scatter_fig = figures_dir / f'sentiment_shares_scatter_over_time_{variant}{output_suffix}.png'
    overlay_fig = figures_dir / f'mean_compound_line_scatter_over_time_{variant}{output_suffix}.png' if include_overlay else None

    marker_size = 16
    marker_alpha = 0.55

    plt.figure(figsize=(12, 5.5))
    plt.scatter(
        daily_df['date'],
        daily_df['mean_compound'],
        s=marker_size,
        alpha=marker_alpha,
        color='tab:blue',
        edgecolors='none',
    )
    plt.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    plt.title(f'Daily Mean Compound Sentiment (Scatter, {variant})')
    plt.xlabel('Date')
    plt.ylabel('Mean compound score')
    _apply_readable_date_axis(plt.gca())
    plt.tight_layout()
    plt.savefig(compound_scatter_fig, dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5.5))
    plt.scatter(daily_df['date'], daily_df['pct_positive'], s=marker_size, alpha=marker_alpha, label='pct_positive')
    plt.scatter(daily_df['date'], daily_df['pct_negative'], s=marker_size, alpha=marker_alpha, label='pct_negative')
    plt.scatter(daily_df['date'], daily_df['pct_neutral'], s=marker_size, alpha=marker_alpha, label='pct_neutral')
    plt.title(f'Daily Sentiment Shares (Scatter, {variant})')
    plt.xlabel('Date')
    plt.ylabel('Percent of comments')
    _apply_readable_date_axis(plt.gca())
    plt.legend()
    plt.tight_layout()
    plt.savefig(pct_scatter_fig, dpi=150)
    plt.close()

    if include_overlay and overlay_fig is not None:
        plt.figure(figsize=(12, 5.5))
        plt.plot(daily_df['date'], daily_df['mean_compound'], linewidth=1.2, color='tab:blue', alpha=0.8, label='line')
        plt.scatter(
            daily_df['date'],
            daily_df['mean_compound'],
            s=marker_size,
            alpha=marker_alpha,
            color='tab:blue',
            edgecolors='none',
            label='daily point',
        )
        plt.axhline(0.0, color='gray', linestyle='--', linewidth=1)
        plt.title(f'Daily Mean Compound Sentiment (Line + Scatter, {variant})')
        plt.xlabel('Date')
        plt.ylabel('Mean compound score')
        _apply_readable_date_axis(plt.gca())
        plt.legend()
        plt.tight_layout()
        plt.savefig(overlay_fig, dpi=150)
        plt.close()

    return compound_scatter_fig, pct_scatter_fig, overlay_fig


def _plot_all_variants_comparison(all_variant_daily: dict[str, pd.DataFrame], figures_dir: Path, output_suffix: str = '') -> Path:
    comparison_fig = figures_dir / f'mean_compound_over_time_all_variants{output_suffix}.png'
    plt.figure(figsize=(12, 5.5))
    for variant, daily_df in all_variant_daily.items():
        plt.plot(daily_df['date'], daily_df['mean_compound'], linewidth=1.7, label=variant)
    plt.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    plt.title('Daily Mean Compound Sentiment (All Variants)')
    plt.xlabel('Date')
    plt.ylabel('Mean compound score')
    _apply_readable_date_axis(plt.gca())
    plt.legend()
    plt.tight_layout()
    plt.savefig(comparison_fig, dpi=150)
    plt.close()
    return comparison_fig


def _plot_all_candidates_comparison(all_candidate_daily: dict[str, pd.DataFrame], figures_dir: Path, output_suffix: str = '') -> Path:
    comparison_fig = figures_dir / f'mean_compound_over_time_candidates_comparison{output_suffix}.png'
    plt.figure(figsize=(12, 5.5))
    for candidate, daily_df in all_candidate_daily.items():
        color = _candidate_color(candidate)
        plt.plot(daily_df['date'], daily_df['mean_compound'], linewidth=1.7, label=candidate, color=color)
    plt.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    plt.title('Daily Mean Compound Sentiment (Candidates Comparison)')
    plt.xlabel('Date')
    plt.ylabel('Mean compound score')
    _apply_readable_date_axis(plt.gca())
    plt.legend()
    plt.tight_layout()
    plt.savefig(comparison_fig, dpi=150)
    plt.close()
    return comparison_fig


def _plot_all_candidates_scatter_comparison(all_candidate_daily: dict[str, pd.DataFrame], figures_dir: Path, output_suffix: str = '') -> Path:
    comparison_fig = figures_dir / f'mean_compound_scatter_over_time_candidates_comparison{output_suffix}.png'
    candidates = list(all_candidate_daily.keys())
    fig, axes = plt.subplots(len(candidates), 1, figsize=(12, 3.6 * len(candidates)), sharex=True)

    if len(candidates) == 1:
        axes = [axes]

    marker_size = 14
    marker_alpha = 0.55

    for ax, candidate in zip(axes, candidates):
        daily_df = all_candidate_daily[candidate]
        color = _candidate_color(candidate)
        ax.scatter(
            daily_df['date'],
            daily_df['mean_compound'],
            s=marker_size,
            alpha=marker_alpha,
            label=candidate,
            color=color,
        )
        ax.plot(daily_df['date'], daily_df['mean_compound'], linewidth=1.0, alpha=0.6, color=color)
        ax.axhline(0.0, color='gray', linestyle='--', linewidth=0.9)
        ax.set_ylabel('Mean compound')
        ax.set_title(candidate)
        _apply_readable_date_axis(ax)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Date')
    fig.suptitle('Daily Mean Compound Sentiment (Candidates Scatter Comparison)', y=0.995)
    fig.tight_layout()
    fig.savefig(comparison_fig, dpi=150)
    plt.close(fig)
    return comparison_fig


def _plot_relevance_mean_compound(daily_df: pd.DataFrame, figures_dir: Path, relevance_type: str, output_suffix: str = '') -> Path:
    compound_fig = figures_dir / f'mean_compound_over_time_{relevance_type}{output_suffix}.png'
    plt.figure(figsize=(12, 5.5))
    plt.plot(daily_df['date'], daily_df['mean_compound'], linewidth=1.8, color='tab:blue')
    plt.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    plt.title(f'Daily Mean Compound Sentiment ({relevance_type})')
    plt.xlabel('Date')
    plt.ylabel('Mean compound score')
    _apply_readable_date_axis(plt.gca())
    plt.tight_layout()
    plt.savefig(compound_fig, dpi=150)
    plt.close()
    return compound_fig


def _plot_all_relevance_comparison(all_relevance_daily: dict[str, pd.DataFrame], figures_dir: Path, output_suffix: str = '') -> Path:
    comparison_fig = figures_dir / f'mean_compound_over_time_relevance_comparison{output_suffix}.png'
    plt.figure(figsize=(12, 5.8))
    for relevance_type, daily_df in all_relevance_daily.items():
        plt.plot(daily_df['date'], daily_df['mean_compound'], linewidth=1.8, label=relevance_type)
    plt.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    plt.title('Daily Mean Compound Sentiment (Relevance Comparison)')
    plt.xlabel('Date')
    plt.ylabel('Mean compound score')
    _apply_readable_date_axis(plt.gca())
    plt.legend()
    plt.tight_layout()
    plt.savefig(comparison_fig, dpi=150)
    plt.close()
    return comparison_fig


def _load_scored_comments(scored_path: Path) -> pd.DataFrame:
    scored_df = pd.read_csv(scored_path)
    scored_df['date'] = pd.to_datetime(scored_df['date'], errors='coerce')
    scored_df = scored_df.dropna(subset=['date']).copy()
    scored_df['date'] = scored_df['date'].dt.strftime('%Y-%m-%d')
    return scored_df


def _plot_mean_line_scatter_overlay(daily_df: pd.DataFrame, figures_dir: Path, stem: str, title: str, output_suffix: str = '') -> Path:
    out_path = figures_dir / f'mean_compound_line_scatter_over_time_{stem}{output_suffix}.png'
    marker_size = 15
    marker_alpha = 0.55
    candidate = stem.split('_', 1)[0] if '_' in stem else stem
    color = _candidate_color(candidate)

    plt.figure(figsize=(12, 5.5))
    plt.plot(daily_df['date'], daily_df['mean_compound'], linewidth=1.4, color=color, alpha=0.85)
    plt.scatter(
        daily_df['date'],
        daily_df['mean_compound'],
        s=marker_size,
        alpha=marker_alpha,
        color=color,
        edgecolors='none',
    )
    plt.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Mean compound score')
    _apply_readable_date_axis(plt.gca())
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _plot_shares_line_scatter_overlay(daily_df: pd.DataFrame, figures_dir: Path, stem: str, title: str, output_suffix: str = '') -> Path:
    out_path = figures_dir / f'sentiment_shares_line_scatter_over_time_{stem}{output_suffix}.png'
    marker_size = 12
    marker_alpha = 0.45

    plt.figure(figsize=(12, 5.5))
    for metric in ['pct_positive', 'pct_negative', 'pct_neutral']:
        plt.plot(daily_df['date'], daily_df[metric], linewidth=1.3, label=metric)
        plt.scatter(
            daily_df['date'],
            daily_df[metric],
            s=marker_size,
            alpha=marker_alpha,
            edgecolors='none',
        )
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Percent of comments')
    _apply_readable_date_axis(plt.gca())
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _plot_candidate_relevance_mean_comparison(
    daily_by_candidate: dict[str, pd.DataFrame],
    figures_dir: Path,
    relevance_type: str,
    output_suffix: str = '',
) -> Path:
    out_path = figures_dir / f'mean_compound_line_scatter_over_time_candidates_{relevance_type}_comparison{output_suffix}.png'
    marker_size = 13
    marker_alpha = 0.5

    plt.figure(figsize=(12, 5.8))
    for candidate, daily_df in daily_by_candidate.items():
        color = _candidate_color(candidate)
        plt.plot(daily_df['date'], daily_df['mean_compound'], linewidth=1.4, alpha=0.9, label=candidate, color=color)
        plt.scatter(
            daily_df['date'],
            daily_df['mean_compound'],
            s=marker_size,
            alpha=marker_alpha,
            color=color,
            edgecolors='none',
        )
    plt.axhline(0.0, color='gray', linestyle='--', linewidth=1)
    plt.title(f'Daily Mean Compound Sentiment by Candidate ({relevance_type})')
    plt.xlabel('Date')
    plt.ylabel('Mean compound score')
    _apply_readable_date_axis(plt.gca())
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def _plot_candidate_relevance_single_share_comparison(
    daily_by_candidate: dict[str, pd.DataFrame],
    figures_dir: Path,
    relevance_type: str,
    metric: str,
    output_suffix: str = '',
) -> Path:
    out_path = figures_dir / f'{metric}_line_scatter_over_time_candidates_{relevance_type}_comparison{output_suffix}.png'
    candidates = list(daily_by_candidate.keys())
    marker_size = 12
    marker_alpha = 0.5

    fig, axes = plt.subplots(len(candidates), 1, figsize=(12, 3.8 * len(candidates)), sharex=True)
    if len(candidates) == 1:
        axes = [axes]

    for ax, candidate in zip(axes, candidates):
        daily_df = daily_by_candidate[candidate]
        ax.plot(daily_df['date'], daily_df[metric], linewidth=1.4, alpha=0.9, color='tab:blue')
        ax.scatter(
            daily_df['date'],
            daily_df[metric],
            s=marker_size,
            alpha=marker_alpha,
            color='tab:blue',
            edgecolors='none',
        )
        ax.set_ylabel('Percent')
        ax.set_title(candidate)
        _apply_readable_date_axis(ax)

    axes[-1].set_xlabel('Date')
    fig.suptitle(f'Daily {metric} by Candidate ({relevance_type})', y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _aggregate_candidate_relevance_daily(
    scored_df: pd.DataFrame,
    candidate: str,
    relevance_type: str,
) -> tuple[pd.DataFrame, int, str, str]:
    candidate_col = f'match_{candidate}'
    mask = scored_df[candidate_col] & scored_df[relevance_type]
    filtered = scored_df.loc[mask].copy()
    if len(filtered):
        daily_df = aggregate_sentiment(filtered, group_col='date')
        daily_df['date'] = pd.to_datetime(daily_df['date'], errors='coerce')
        daily_df = daily_df.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
        date_min = str(filtered['date'].min())
        date_max = str(filtered['date'].max())
    else:
        daily_df = aggregate_sentiment(scored_df.head(0), group_col='date')
        daily_df['date'] = pd.to_datetime(daily_df['date'], errors='coerce')
        date_min = 'N/A'
        date_max = 'N/A'
    return daily_df, int(len(filtered)), date_min, date_max


def main() -> None:
    args = _parse_args()

    workspace = Path('/Users/ajoji/Desktop/2025-2026/CS598_CSS')
    figures_dir = workspace / 'outputs' / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    start_ts, end_ts, output_suffix = _parse_date_window(args.start_date, args.end_date)

    run_candidate_mode = args.candidate is not None or args.all_candidates
    run_relevance_mode = args.relevance is not None or args.all_relevance
    run_candidate_relevance_mode = args.candidate_relevance is not None or args.all_candidate_relevance

    active_modes = sum([bool(run_candidate_mode), bool(run_relevance_mode), bool(run_candidate_relevance_mode)])
    if active_modes > 1:
        raise ValueError('Choose only one plotting mode among candidate, relevance, or candidate+relevance.')

    if run_candidate_relevance_mode:
        workspace = Path('/Users/ajoji/Desktop/2025-2026/CS598_CSS')
        scored_path = workspace / 'outputs' / 'tables' / 'comment_vader_scored.csv'
        scored_df = _load_scored_comments(scored_path)
        required_cols = [
            'date',
            'pos',
            'neu',
            'neg',
            'compound',
            'sentiment_label',
            'word_count',
            'match_cuomo',
            'match_mamdani',
            'match_sliwa',
            'explicit_comment_relevant',
            'thread_relevant_only',
            'merged_relevant',
        ]
        missing = [column for column in required_cols if column not in scored_df.columns]
        if missing:
            raise ValueError(f'Missing required columns in comment_vader_scored.csv: {missing}')

        relevance_types = CANDIDATE_RELEVANCE_TYPES if args.all_candidate_relevance else [args.candidate_relevance or 'merged_relevant']
        candidates = list(CANDIDATE_TO_DAILY_FILE.keys())

        for relevance_type in relevance_types:
            daily_by_candidate: dict[str, pd.DataFrame] = {}
            for candidate in candidates:
                daily_df, row_count, date_min, date_max = _aggregate_candidate_relevance_daily(scored_df, candidate, relevance_type)
                daily_df = _apply_date_window(daily_df, start_ts, end_ts)
                if len(daily_df) == 0:
                    print(
                        f"SKIP: no data after date window for CANDIDATE_RELEVANCE {candidate} {relevance_type} "
                        f"(start_date={args.start_date}, end_date={args.end_date})"
                    )
                    continue
                daily_by_candidate[candidate] = daily_df

                date_min_plot = daily_df['date'].min().date()
                date_max_plot = daily_df['date'].max().date()

                mean_fig = _plot_mean_line_scatter_overlay(
                    daily_df,
                    figures_dir,
                    stem=f'{candidate}_{relevance_type}',
                    title=f'Daily Mean Compound Sentiment ({candidate}, {relevance_type})',
                    output_suffix=output_suffix,
                )
                shares_fig = _plot_shares_line_scatter_overlay(
                    daily_df,
                    figures_dir,
                    stem=f'{candidate}_{relevance_type}',
                    title=f'Daily Sentiment Shares ({candidate}, {relevance_type})',
                    output_suffix=output_suffix,
                )

                print('CANDIDATE_RELEVANCE:', candidate, relevance_type)
                print('ROWS_FOR_AGG:', row_count)
                print('RAW_DATE_RANGE:', date_min, 'to', date_max)
                print('PLOT_DATE_RANGE:', date_min_plot, 'to', date_max_plot)
                print('DAILY_ROWS_FOR_PLOTTING:', len(daily_df))
                print('WROTE FIGURES:')
                print(mean_fig)
                print(shares_fig)
                print('---')

            if len(daily_by_candidate) == 0:
                print(
                    f"SKIP: no candidate data available for comparison after date window for relevance_type={relevance_type} "
                    f"(start_date={args.start_date}, end_date={args.end_date})"
                )
                continue

            comparison_mean_fig = _plot_candidate_relevance_mean_comparison(
                daily_by_candidate,
                figures_dir,
                relevance_type,
                output_suffix=output_suffix,
            )
            share_comparison_figs = [
                _plot_candidate_relevance_single_share_comparison(daily_by_candidate, figures_dir, relevance_type, metric='pct_positive', output_suffix=output_suffix),
                _plot_candidate_relevance_single_share_comparison(daily_by_candidate, figures_dir, relevance_type, metric='pct_negative', output_suffix=output_suffix),
                _plot_candidate_relevance_single_share_comparison(daily_by_candidate, figures_dir, relevance_type, metric='pct_neutral', output_suffix=output_suffix),
            ]
            print('CANDIDATE_RELEVANCE COMPARISON:', relevance_type)
            print('WROTE FIGURES:')
            print(comparison_mean_fig)
            for fig_path in share_comparison_figs:
                print(fig_path)
            print('===')

        return

    if run_candidate_mode:
        if args.all_candidates:
            variants_to_run = list(CANDIDATE_TO_DAILY_FILE.keys())
        else:
            variants_to_run = [args.candidate or 'mamdani']
        file_map = CANDIDATE_TO_DAILY_FILE
        comparison_mode = args.all_candidates
        comparison_label = 'CANDIDATE'
    elif run_relevance_mode:
        if args.all_relevance:
            variants_to_run = list(RELEVANCE_TO_DAILY_FILE.keys())
        else:
            variants_to_run = [args.relevance or 'merged_relevant']
        file_map = RELEVANCE_TO_DAILY_FILE
        comparison_mode = args.all_relevance
        comparison_label = 'RELEVANCE'
    else:
        if args.all_variants:
            variants_to_run = list(VARIANT_TO_DAILY_FILE.keys())
        else:
            variants_to_run = [args.variant or 'high_recall']
        file_map = VARIANT_TO_DAILY_FILE
        comparison_mode = args.all_variants
        comparison_label = 'VARIANT'

    all_variant_daily: dict[str, pd.DataFrame] = {}

    for variant in variants_to_run:
        daily_path = workspace / 'outputs' / 'tables' / file_map[variant]
        daily_df = _load_daily_table(daily_path)
        daily_df = _apply_date_window(daily_df, start_ts, end_ts)
        if len(daily_df) == 0:
            print(
                f"SKIP: no data after date window for {comparison_label} {variant} "
                f"(start_date={args.start_date}, end_date={args.end_date})"
            )
            continue
        all_variant_daily[variant] = daily_df

        date_min = daily_df['date'].min()
        date_max = daily_df['date'].max()

        scatter_outputs: tuple[Path, Path, Path | None] | None = None
        if run_relevance_mode:
            compound_fig = _plot_relevance_mean_compound(daily_df, figures_dir, variant, output_suffix=output_suffix)
            pct_fig = None
            volume_fig = None
        else:
            compound_fig, pct_fig, volume_fig = _plot_variant(daily_df, figures_dir, variant, output_suffix=output_suffix)
            if args.add_scatter:
                scatter_outputs = _plot_variant_scatter(
                    daily_df,
                    figures_dir,
                    variant,
                    include_overlay=args.line_scatter_overlay,
                    output_suffix=output_suffix,
                )

        print(f'{comparison_label}:', variant)
        print('INPUT:', daily_path)
        print('DATE_RANGE:', date_min.date(), 'to', date_max.date())
        print('ROWS_FOR_PLOTTING:', len(daily_df))
        print('HEAD(5):')
        print(daily_df.head(5).to_string(index=False))
        print('WROTE FIGURES:')
        print(compound_fig)
        if pct_fig is not None:
            print(pct_fig)
        if volume_fig is not None:
            print(volume_fig)
        if scatter_outputs is not None:
            print(scatter_outputs[0])
            print(scatter_outputs[1])
            if scatter_outputs[2] is not None:
                print(scatter_outputs[2])
        print('---')

    if comparison_mode:
        if len(all_variant_daily) == 0:
            print(
                f"SKIP: no series available for comparison after date window "
                f"(start_date={args.start_date}, end_date={args.end_date})"
            )
            return
        if run_candidate_mode:
            comparison_fig = _plot_all_candidates_comparison(all_variant_daily, figures_dir, output_suffix=output_suffix)
            print('WROTE COMPARISON FIGURE:')
            print(comparison_fig)
            if args.add_scatter:
                scatter_comparison_fig = _plot_all_candidates_scatter_comparison(all_variant_daily, figures_dir, output_suffix=output_suffix)
                print('WROTE SCATTER COMPARISON FIGURE:')
                print(scatter_comparison_fig)
        elif run_relevance_mode:
            comparison_fig = _plot_all_relevance_comparison(all_variant_daily, figures_dir, output_suffix=output_suffix)
            print('WROTE COMPARISON FIGURE:')
            print(comparison_fig)
        else:
            comparison_fig = _plot_all_variants_comparison(all_variant_daily, figures_dir, output_suffix=output_suffix)
            print('WROTE COMPARISON FIGURE:')
            print(comparison_fig)


if __name__ == '__main__':
    main()
