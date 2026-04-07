from __future__ import annotations

import re
from typing import Any

import pandas as pd

CANDIDATE_TERMS: dict[str, list[str]] = {
    "cuomo": ["Andrew Cuomo", "Cuomo", "andrew"],
    "mamdani": ["Zohran Mamdani", "Mamdani", "Zohran"],
    "sliwa": ["Curtis Sliwa", "Sliwa", "Curtis"],
}

CANDIDATE_CANONICAL = list(CANDIDATE_TERMS.keys())

ELECTION_TERMS: list[str] = [
    "mayor",
    "mayoral",
    "election",
    "vote",
    "voting",
    "ballot",
    "primary",
    "campaign",
    "candidate",
    "poll",
    "polling",
    "debate",
    "ranked choice",
    "ranked-choice",
    "rcv",
]


def _normalize_phrase_for_pattern(term: str) -> str:
    escaped = re.escape(term.strip())
    escaped = escaped.replace(r"\ ", r"[\s\-]+")
    escaped = escaped.replace(r"\-", r"[\s\-]+")
    return escaped


def _compile_term_pattern(term: str, word_boundary: bool = True) -> re.Pattern[str]:
    core = _normalize_phrase_for_pattern(term)
    if word_boundary:
        expression = rf"\b{core}\b"
    else:
        expression = core
    return re.compile(expression, flags=re.IGNORECASE)


def _compile_candidates(candidate_map: dict[str, list[str]], word_boundary: bool = True) -> dict[str, list[re.Pattern[str]]]:
    compiled: dict[str, list[re.Pattern[str]]] = {}
    for canonical, aliases in candidate_map.items():
        compiled[canonical] = [_compile_term_pattern(alias, word_boundary=word_boundary) for alias in aliases]
    return compiled


def _compile_election_terms(election_terms: list[str], word_boundary: bool = True) -> dict[str, re.Pattern[str]]:
    return {term: _compile_term_pattern(term, word_boundary=word_boundary) for term in election_terms}


def _safe_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def _find_candidate_hits(text: str, patterns: dict[str, list[re.Pattern[str]]]) -> list[str]:
    hits: list[str] = []
    for canonical, compiled_list in patterns.items():
        if any(pattern.search(text) for pattern in compiled_list):
            hits.append(canonical)
    return sorted(set(hits))


def _find_election_hits(text: str, patterns: dict[str, re.Pattern[str]]) -> list[str]:
    hits: list[str] = []
    for term, pattern in patterns.items():
        if pattern.search(text):
            hits.append(term)
    return sorted(set(hits))


def apply_topic_filters(
    df: pd.DataFrame,
    body_col: str = "body",
    title_col: str = "title",
    post_body_col: str | None = None,
    word_boundary: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()

    body = _safe_text(out[body_col])
    title = _safe_text(out[title_col])
    if post_body_col is not None and post_body_col in out.columns:
        post_body = _safe_text(out[post_body_col])
    else:
        post_body = pd.Series("", index=out.index, dtype="object")

    candidate_patterns = _compile_candidates(CANDIDATE_TERMS, word_boundary=word_boundary)
    election_patterns = _compile_election_terms(ELECTION_TERMS, word_boundary=word_boundary)

    body_candidate_hits = body.map(lambda txt: _find_candidate_hits(txt, candidate_patterns))
    title_candidate_hits = title.map(lambda txt: _find_candidate_hits(txt, candidate_patterns))
    post_body_candidate_hits = post_body.map(lambda txt: _find_candidate_hits(txt, candidate_patterns))

    body_election_hits = body.map(lambda txt: _find_election_hits(txt, election_patterns))
    title_election_hits = title.map(lambda txt: _find_election_hits(txt, election_patterns))
    post_body_election_hits = post_body.map(lambda txt: _find_election_hits(txt, election_patterns))

    combined_candidate_hits = [sorted(set(bh) | set(th)) for bh, th in zip(body_candidate_hits, title_candidate_hits)]
    combined_election_hits = [sorted(set(bh) | set(th)) for bh, th in zip(body_election_hits, title_election_hits)]

    out["matched_candidates"] = [";".join(items) for items in combined_candidate_hits]
    out["matched_election_terms"] = [";".join(items) for items in combined_election_hits]

    out["candidate_match"] = out["matched_candidates"].ne("")
    out["election_match"] = out["matched_election_terms"].ne("")

    out["matched_in_body"] = [
        bool(bc) or bool(be) for bc, be in zip(body_candidate_hits, body_election_hits)
    ]
    out["matched_in_title"] = [
        bool(tc) or bool(te) for tc, te in zip(title_candidate_hits, title_election_hits)
    ]
    out["matched_in_post_body"] = [
        bool(pc) or bool(pe) for pc, pe in zip(post_body_candidate_hits, post_body_election_hits)
    ]

    out["post_relevant_any"] = out["matched_in_title"] | out["matched_in_post_body"]
    out["explicit_comment_relevant"] = out["matched_in_body"]
    out["thread_relevant_only"] = out["post_relevant_any"] & (~out["explicit_comment_relevant"])
    out["merged_relevant"] = out["explicit_comment_relevant"] | out["post_relevant_any"]

    out["filter_candidate_only"] = out["candidate_match"]
    out["filter_election_only"] = out["election_match"]
    out["filter_high_recall"] = out["candidate_match"] | out["election_match"]
    out["filter_higher_precision"] = out["candidate_match"] & out["election_match"]

    stats = {
        "total_comments": int(len(out)),
        "candidate_match": int(out["candidate_match"].sum()),
        "election_match": int(out["election_match"].sum()),
        "filter_high_recall": int(out["filter_high_recall"].sum()),
        "filter_higher_precision": int(out["filter_higher_precision"].sum()),
    }

    return out, stats


def add_candidate_match_flags(
    df: pd.DataFrame,
    matched_candidates_col: str = 'matched_candidates',
    candidates: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    canonical_candidates = candidates or CANDIDATE_CANONICAL

    matched_series = out[matched_candidates_col].fillna('').astype(str)
    for candidate in canonical_candidates:
        pattern = rf'(?:^|;){re.escape(candidate)}(?:;|$)'
        out[f'match_{candidate}'] = matched_series.str.contains(pattern, regex=True)

    out['matched_candidate_count'] = out[[f'match_{candidate}' for candidate in canonical_candidates]].sum(axis=1)
    return out
