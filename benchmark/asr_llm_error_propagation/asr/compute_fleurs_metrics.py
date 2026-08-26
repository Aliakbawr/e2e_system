#!/usr/bin/env python3
"""Compute corpus-level WER/CER and runtime statistics for the frozen ASR CSV."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from benchmark.asr_llm_error_propagation.asr.run_fleurs_fa_ir_asr import normalize_persian_for_evaluation


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR / "fleurs_fa_ir_asr_checkpoint.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "fleurs_fa_ir_asr_metrics.json"


def _alignment_error_rate(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> tuple[float, int, int, int, int]:
    """Return error rate, substitutions, deletions, insertions, and ref length.

    This is a Levenshtein alignment with deterministic backtracing. Ties prefer
    a diagonal operation, then deletion, then insertion. The rate denominator
    is the number of reference tokens, as in Kaldi/Icefall corpus scoring.
    """
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    costs = [[0] * columns for _ in range(rows)]
    operations = [[""] * columns for _ in range(rows)]

    for row in range(1, rows):
        costs[row][0] = row
        operations[row][0] = "D"
    for column in range(1, columns):
        costs[0][column] = column
        operations[0][column] = "I"

    for row in range(1, rows):
        for column in range(1, columns):
            if reference[row - 1] == hypothesis[column - 1]:
                costs[row][column] = costs[row - 1][column - 1]
                operations[row][column] = "="
                continue

            candidates = (
                (costs[row - 1][column - 1] + 1, 0, "S"),
                (costs[row - 1][column] + 1, 1, "D"),
                (costs[row][column - 1] + 1, 2, "I"),
            )
            cost, _, operation = min(candidates)
            costs[row][column] = cost
            operations[row][column] = operation

    substitutions = deletions = insertions = 0
    row = len(reference)
    column = len(hypothesis)
    while row or column:
        operation = operations[row][column]
        if operation in {"=", "S"}:
            substitutions += operation == "S"
            row -= 1
            column -= 1
        elif operation == "D":
            deletions += 1
            row -= 1
        elif operation == "I":
            insertions += 1
            column -= 1
        else:
            raise RuntimeError(f"Invalid alignment state at ({row}, {column})")

    reference_length = len(reference)
    rate = (substitutions + deletions + insertions) / max(1, reference_length)
    return rate, substitutions, deletions, insertions, reference_length


def compute_dataset_wer_cer(
    df: pd.DataFrame,
    ref_col: str = "ref",
    hyp_col: str = "hyp",
) -> dict[str, float | int]:
    """Compute overall dataset WER/CER using corpus-level error aggregation."""
    missing = {ref_col, hyp_col} - set(df.columns)
    if missing:
        raise ValueError(f"Missing dataframe columns: {sorted(missing)}")

    total_word_sub = total_word_del = total_word_ins = 0
    total_words = 0
    total_char_sub = total_char_del = total_char_ins = 0
    total_chars = 0

    for ref, hyp in zip(df[ref_col], df[hyp_col]):
        ref = "" if pd.isna(ref) else str(ref)
        hyp = "" if pd.isna(hyp) else str(hyp)

        _, substitutions, deletions, insertions, reference_length = (
            _alignment_error_rate(ref.split(), hyp.split())
        )
        total_word_sub += substitutions
        total_word_del += deletions
        total_word_ins += insertions
        total_words += reference_length

        # Spaces are excluded from CER, matching the supplied scoring code.
        _, substitutions, deletions, insertions, reference_length = (
            _alignment_error_rate(
                list(ref.replace(" ", "")),
                list(hyp.replace(" ", "")),
            )
        )
        total_char_sub += substitutions
        total_char_del += deletions
        total_char_ins += insertions
        total_chars += reference_length

    wer = (total_word_sub + total_word_del + total_word_ins) / max(1, total_words)
    cer = (total_char_sub + total_char_del + total_char_ins) / max(1, total_chars)

    return {
        "WER": wer,
        "CER": cer,
        "word_sub": total_word_sub,
        "word_del": total_word_del,
        "word_ins": total_word_ins,
        "char_sub": total_char_sub,
        "char_del": total_char_del,
        "char_ins": total_char_ins,
        "total_words": total_words,
        "total_chars": total_chars,
    }


def _finite_series(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    return values[values.map(math.isfinite)]


def compute_full_summary(df: pd.DataFrame) -> dict[str, Any]:
    required = {
        "id",
        "file_name",
        "reference_normalized",
        "asr_raw",
        "duration_sec",
        "latency_sec",
        "rtf",
        "status",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Checkpoint is missing columns: {sorted(missing)}")
    if df["file_name"].duplicated().any():
        raise ValueError("Checkpoint contains duplicate file_name values")

    successful = df[df["status"] == "ok"].copy()
    if successful.empty:
        raise ValueError("Checkpoint contains no successful recordings")

    # reference_normalized preserves the official FLEURS field. Apply the same
    # deterministic evaluation normalizer used to create asr_normalized so the
    # two sides receive identical punctuation/ZWNJ/digit handling.
    successful["ref"] = successful["reference_normalized"].map(
        normalize_persian_for_evaluation
    )
    successful["hyp"] = successful["asr_raw"].map(
        normalize_persian_for_evaluation
    )
    error_metrics = compute_dataset_wer_cer(successful)

    durations = _finite_series(successful, "duration_sec")
    latencies = _finite_series(successful, "latency_sec")
    rtfs = _finite_series(successful, "rtf")
    total_duration = float(durations.sum())
    total_latency = float(latencies.sum())

    return {
        "scoring": {
            "type": "corpus_level_icefall_kaldi_style",
            "wer_tokens": "whitespace-separated words",
            "cer_tokens": "characters_excluding_spaces",
            "normalization": "normalize_persian_for_evaluation applied to both sides",
            "reference_source": "reference_normalized (official FLEURS transcription)",
            "hypothesis_source": "asr_raw",
        },
        "dataset": {
            "recordings": int(len(df)),
            "successful_recordings": int(len(successful)),
            "failed_recordings": int((df["status"] != "ok").sum()),
            "unique_file_names": int(df["file_name"].nunique()),
            "unique_semantic_ids": int(df["id"].nunique()),
        },
        "error_metrics": error_metrics,
        "runtime": {
            "total_audio_sec": total_duration,
            "total_audio_hours": total_duration / 3600,
            "total_latency_sec": total_latency,
            "mean_latency_sec": float(latencies.mean()),
            "median_latency_sec": float(latencies.median()),
            "p95_latency_sec": float(latencies.quantile(0.95)),
            "mean_rtf": float(rtfs.mean()),
            "median_rtf": float(rtfs.median()),
            "aggregate_rtf": total_latency / max(total_duration, 1e-12),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    dataframe = pd.read_csv(args.checkpoint, dtype={"id": str, "file_name": str})
    summary = compute_full_summary(dataframe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Metrics written to: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
