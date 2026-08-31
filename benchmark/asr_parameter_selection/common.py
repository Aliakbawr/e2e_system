"""Shared metrics and selection helpers for ASR parameter sweeps."""

from __future__ import annotations

import csv
from pathlib import Path

from benchmark.asr_llm_error_propagation.asr.run_fleurs_fa_ir_asr import (
    edit_distance,
    normalize_persian_for_evaluation,
)
from benchmark.asr_llm_error_propagation.prepare_downstream_eval import normalize_persian


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def word_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    reference_tokens = normalize_persian_for_evaluation(reference).split()
    hypothesis_tokens = normalize_persian_for_evaluation(hypothesis).split()
    return edit_distance(reference_tokens, hypothesis_tokens), len(reference_tokens)


def answer_preserved(gold: str, hypothesis: str) -> bool:
    normalized_gold = normalize_persian(gold)
    return bool(normalized_gold) and normalized_gold in normalize_persian(hypothesis)


def choose_development_configuration(
    rows: list[dict[str, object]],
    *,
    max_clarification_rate: float,
    max_unnecessary_rate: float,
) -> dict[str, object]:
    """Select without looking at held-out metrics.

    Feasible configurations maximize answer-span preservation, then minimize
    oracle-assisted WER, unnecessary clarifications, and total clarifications.
    If no configuration satisfies the operational limits, violations are
    minimized before applying the same performance ordering.
    """
    development = [row for row in rows if row["split"] == "development"]
    if not development:
        raise ValueError("No development configurations were supplied")

    def violations(row: dict[str, object]) -> tuple[float, float]:
        return (
            max(0.0, float(row["clarification_rate"]) - max_clarification_rate),
            max(0.0, float(row["unnecessary_clarification_rate"]) - max_unnecessary_rate),
        )

    feasible = [row for row in development if violations(row) == (0.0, 0.0)]
    pool = feasible or development
    selected = min(
        pool,
        key=lambda row: (
            0.0 if feasible else sum(violations(row)),
            -float(row["oracle_answer_span_rate"]),
            float(row["oracle_corpus_wer"]),
            float(row["unnecessary_clarification_rate"]),
            float(row["clarification_rate"]),
            float(row["word_confidence_threshold"]),
            float(row["alternative_score_gap"]),
        ),
    )
    return dict(selected)
