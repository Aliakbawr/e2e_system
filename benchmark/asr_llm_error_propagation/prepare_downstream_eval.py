#!/usr/bin/env python3
"""Join frozen FLEURS ASR recordings to frozen QA and analyze answer survival."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ASR_PATH = BASE_DIR / "asr/fleurs_fa_ir_asr_checkpoint.csv"
QA_PATH = BASE_DIR / "qa/fleurs_fa_ir_generated_qa_candidates_v1.csv"
OUT_PATH = BASE_DIR / "analysis/inputs/fleurs_asr_qa_eval_input_v1.csv"
ANSWER_SPAN_DIR = BASE_DIR / "analysis/results/answer_span/original_asr_v1"
SUMMARY_PATH = ANSWER_SPAN_DIR / "answer_preservation_summary_v1.json"
LOW_WER_LOSS_PATH = ANSWER_SPAN_DIR / "critical_low_wer_answer_loss_v1.csv"
HIGH_WER_SURVIVAL_PATH = ANSWER_SPAN_DIR / "robust_high_wer_answer_survival_v1.csv"

DIGIT_MAP = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)


def normalize_persian(text: Any) -> str:
    """Apply the frozen answer-preservation normalization from the protocol."""
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    for source, target in {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "\u200c": " ",
    }.items():
        text = text.replace(source, target)
    text = text.translate(DIGIT_MAP)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip().lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def contains_token_span(gold: str, text: str) -> bool:
    """Return whether all normalized gold tokens occur as one exact token span."""
    gold_tokens = gold.split()
    text_tokens = text.split()
    return bool(gold_tokens) and any(
        text_tokens[index : index + len(gold_tokens)] == gold_tokens
        for index in range(len(text_tokens) - len(gold_tokens) + 1)
    )


def load_frozen_inputs(asr_path: Path, qa_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not asr_path.is_file():
        raise FileNotFoundError(f"Frozen ASR checkpoint not found: {asr_path}")
    if not qa_path.is_file():
        raise FileNotFoundError(f"Frozen QA candidates not found: {qa_path}")

    asr_df = pd.read_csv(asr_path, dtype={"id": str, "file_name": str})
    qa_df = pd.read_csv(qa_path, encoding="utf-8-sig", dtype={"id": str})

    required_asr = {
        "id",
        "file_name",
        "reference_normalized",
        "asr_raw",
        "asr_normalized",
        "wer",
        "cer",
        "status",
    }
    required_qa = {
        "generation_index",
        "id",
        "num_recordings",
        "usable",
        "question",
        "gold_answer",
        "answer_type",
        "review_status",
    }
    if missing := required_asr - set(asr_df.columns):
        raise ValueError(f"ASR checkpoint is missing columns: {sorted(missing)}")
    if missing := required_qa - set(qa_df.columns):
        raise ValueError(f"QA benchmark is missing columns: {sorted(missing)}")
    if asr_df["file_name"].duplicated().any():
        raise ValueError("ASR checkpoint contains duplicate filenames")
    if qa_df["id"].duplicated().any():
        raise ValueError("QA benchmark contains duplicate semantic IDs")
    if not (asr_df["status"] == "ok").all():
        raise ValueError("ASR checkpoint contains failed recordings")

    return asr_df, qa_df


def build_downstream_table(asr_df: pd.DataFrame, qa_df: pd.DataFrame) -> pd.DataFrame:
    # Rejected rows have no frozen question/answer. Manual-review rows remain
    # frozen usable QA items and retain review_status for later stratification.
    usable = qa_df[qa_df["usable"].map(_bool_value)].copy()
    if usable[["question", "gold_answer"]].isna().any().any():
        raise ValueError("At least one usable QA item has no question or gold answer")

    usable = usable.rename(columns={"generation_index": "qa_id"})
    qa_columns = [
        "qa_id",
        "id",
        "num_recordings",
        "question",
        "gold_answer",
        "answer_type",
        "review_status",
        "quality_score",
        "qa_schema_version",
    ]
    qa_columns = [column for column in qa_columns if column in usable.columns]
    merged = asr_df.merge(
        usable[qa_columns],
        on="id",
        how="inner",
        validate="many_to_one",
    )

    expected_counts = usable.set_index("id")["num_recordings"].astype(int)
    actual_counts = merged.groupby("id")["file_name"].count()
    count_mismatches = actual_counts[actual_counts != expected_counts.loc[actual_counts.index]]
    if not count_mismatches.empty:
        raise ValueError(
            f"Recording counts disagree for {len(count_mismatches)} semantic IDs"
        )

    merged["gold_answer_normalized"] = merged["gold_answer"].map(normalize_persian)
    merged["reference_eval"] = merged["reference_normalized"].map(normalize_persian)
    merged["asr_eval"] = merged["asr_raw"].map(normalize_persian)
    if (merged["gold_answer_normalized"] == "").any():
        raise ValueError("At least one normalized gold answer is empty")

    merged["answer_preserved"] = [
        gold in hypothesis
        for gold, hypothesis in zip(
            merged["gold_answer_normalized"], merged["asr_eval"]
        )
    ]
    # Sensitivity analysis only. Keep answer_preserved above as the protocol's
    # primary exact-substring variable.
    merged["answer_preserved_token_span"] = [
        contains_token_span(gold, hypothesis)
        for gold, hypothesis in zip(
            merged["gold_answer_normalized"], merged["asr_eval"]
        )
    ]
    merged["answer_in_reference"] = [
        gold in reference
        for gold, reference in zip(
            merged["gold_answer_normalized"], merged["reference_eval"]
        )
    ]

    failed_reference_checks = merged[~merged["answer_in_reference"]]
    if not failed_reference_checks.empty:
        ids = sorted(failed_reference_checks["id"].unique())
        raise ValueError(
            f"Gold answer was not found in the reference for {len(ids)} IDs: {ids[:10]}"
        )
    return merged


def preservation_summary(merged: pd.DataFrame) -> dict[str, Any]:
    grouped = (
        merged.groupby("answer_preserved", observed=True)
        .agg(
            recordings=("file_name", "count"),
            semantic_ids=("id", "nunique"),
            mean_wer=("wer", "mean"),
            median_wer=("wer", "median"),
            mean_cer=("cer", "mean"),
            median_cer=("cer", "median"),
        )
        .reset_index()
    )
    by_state = {
        str(bool(row["answer_preserved"])).lower(): {
            key: int(value) if key in {"recordings", "semantic_ids"} else float(value)
            for key, value in row.items()
            if key != "answer_preserved"
        }
        for _, row in grouped.iterrows()
    }

    by_id = merged.groupby("id")["answer_preserved"].agg(["any", "all"])
    low_wer_loss = merged[(~merged["answer_preserved"]) & (merged["wer"] <= 0.20)]
    high_wer_survival = merged[merged["answer_preserved"] & (merged["wer"] >= 0.50)]

    # Rank-based ROC AUC: probability that a random answer-loss recording has
    # higher WER than a random answer-preserved recording.
    predicts_loss = (~merged["answer_preserved"]).astype(int)
    ranks = merged["wer"].rank(method="average")
    losses = int(predicts_loss.sum())
    preserved = int((1 - predicts_loss).sum())
    wer_loss_auc = float(
        (ranks[predicts_loss == 1].sum() - losses * (losses + 1) / 2)
        / (losses * preserved)
    )

    bin_edges = [-math.inf, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, math.inf]
    bin_labels = [
        "<=0.10",
        "0.10-0.20",
        "0.20-0.30",
        "0.30-0.40",
        "0.40-0.50",
        "0.50-0.60",
        "0.60-0.70",
        ">0.70",
    ]
    bins = pd.cut(merged["wer"], bins=bin_edges, labels=bin_labels)
    bin_table = (
        merged.assign(wer_bin=bins)
        .groupby("wer_bin", observed=False)["answer_preserved"]
        .agg(["count", "sum", "mean"])
    )
    wer_bins = [
        {
            "wer_bin": str(index),
            "recordings": int(row["count"]),
            "answer_preserved": int(row["sum"]),
            "answer_preservation_rate": float(row["mean"]),
        }
        for index, row in bin_table.iterrows()
    ]

    # Wilson 95% interval for the recording-level preservation probability.
    successes = int(merged["answer_preserved"].sum())
    trials = len(merged)
    z = 1.959963984540054
    denominator = 1 + z * z / trials
    center = (successes / trials + z * z / (2 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            successes / trials * (1 - successes / trials) / trials
            + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return {
        "matched_recording_qa_rows": int(len(merged)),
        "matched_semantic_ids": int(merged["id"].nunique()),
        "answer_in_reference_rate": float(merged["answer_in_reference"].mean()),
        "answer_preservation_rate_recordings": float(merged["answer_preserved"].mean()),
        "answer_preservation_rate_wilson_95ci": [
            center - half_width,
            center + half_width,
        ],
        "answer_preserved_recordings": int(merged["answer_preserved"].sum()),
        "answer_lost_recordings": int((~merged["answer_preserved"]).sum()),
        "answer_preserved_token_span_recordings": int(
            merged["answer_preserved_token_span"].sum()
        ),
        "substring_vs_token_span_disagreements": int(
            (merged["answer_preserved"] != merged["answer_preserved_token_span"]).sum()
        ),
        "semantic_ids_answer_preserved_in_any_recording": int(by_id["any"].sum()),
        "semantic_ids_answer_preserved_in_all_recordings": int(by_id["all"].sum()),
        "critical_low_wer_answer_loss_count": int(len(low_wer_loss)),
        "robust_high_wer_answer_survival_count": int(len(high_wer_survival)),
        "wer_predicts_answer_loss_roc_auc": wer_loss_auc,
        "wer_answer_preserved_pearson_correlation": float(
            merged["wer"].corr(merged["answer_preserved"].astype(int))
        ),
        "answer_preservation_by_wer_bin": wer_bins,
        "by_answer_preserved": by_state,
    }


def _analysis_columns() -> list[str]:
    return [
        "qa_id",
        "id",
        "file_name",
        "question",
        "gold_answer",
        "answer_type",
        "wer",
        "cer",
        "reference_normalized",
        "asr_raw",
        "answer_preserved",
        "answer_preserved_token_span",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-path", type=Path, default=ASR_PATH)
    parser.add_argument("--qa-path", type=Path, default=QA_PATH)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asr_hash_before = sha256_file(args.asr_path)
    qa_hash_before = sha256_file(args.qa_path)
    asr_df, qa_df = load_frozen_inputs(args.asr_path, args.qa_path)
    merged = build_downstream_table(asr_df, qa_df)
    summary = preservation_summary(merged)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False, encoding="utf-8-sig")

    low_wer_loss = merged[(~merged["answer_preserved"]) & (merged["wer"] <= 0.20)]
    low_wer_loss = low_wer_loss.sort_values("wer")
    high_wer_survival = merged[merged["answer_preserved"] & (merged["wer"] >= 0.50)]
    high_wer_survival = high_wer_survival.sort_values("wer", ascending=False)
    columns = _analysis_columns()
    low_wer_loss[columns].to_csv(LOW_WER_LOSS_PATH, index=False, encoding="utf-8-sig")
    high_wer_survival[columns].to_csv(
        HIGH_WER_SURVIVAL_PATH, index=False, encoding="utf-8-sig"
    )

    summary.update(
        {
            "source_hashes": {
                "asr_checkpoint_sha256": asr_hash_before,
                "frozen_qa_sha256": qa_hash_before,
            },
            "output_sha256": sha256_file(args.output),
            "normalization": "normalize_persian in prepare_downstream_eval.py",
            "answer_preserved_definition": (
                "normalized gold answer is an exact substring of normalized ASR text"
            ),
            "thresholds": {
                "critical_low_wer_max": 0.20,
                "robust_high_wer_min": 0.50,
            },
        }
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    if sha256_file(args.asr_path) != asr_hash_before:
        raise RuntimeError("Frozen ASR checkpoint changed during preparation")
    if sha256_file(args.qa_path) != qa_hash_before:
        raise RuntimeError("Frozen QA benchmark changed during preparation")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Downstream table: {args.output.resolve()}")
    print(f"Analysis summary: {args.summary.resolve()}")
    print(f"Low-WER answer-loss cases: {LOW_WER_LOSS_PATH.resolve()}")
    print(f"High-WER answer-survival cases: {HIGH_WER_SURVIVAL_PATH.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
