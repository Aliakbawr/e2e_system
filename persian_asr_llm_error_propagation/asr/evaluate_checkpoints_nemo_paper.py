#!/usr/bin/env python3
"""Re-evaluate original-ASR and Vosk checkpoints with NeMo-paper rules."""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_ORIGINAL_CHECKPOINT = SCRIPT_DIR / "fleurs_fa_ir_asr_checkpoint.csv"
DEFAULT_VOSK_CHECKPOINT = SCRIPT_DIR / "vosk_fleurs_checkpoint.csv"
DEFAULT_ORIGINAL_JOINED = (
    PROJECT_DIR
    / "analysis/results/initial_evaluation_v1/original_asr/gemma2_9b_final_joined_v1.csv"
)
DEFAULT_VOSK_JOINED = (
    PROJECT_DIR
    / "analysis/results/initial_evaluation_v1/vosk/gemma2_9b_final_joined_v1.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "analysis/results/nemo_paper_v1/asr_metrics"

SKIP = set(["ā", "š", "="])

REPLACEMENTS = {
    "أ": "ا",
    "ك": "ک",
    "ي": "ی",
    "ى": "ی",
    "ﯽ": "ی",
    "ﻮ": "و",
    "ە": "ه",
    "ۀ": "ه",
}

DISCARD = set(
    [
        "!",
        '"',
        "#",
        "&",
        "'",
        "(",
        ")",
        ",",
        "-",
        ".",
        ":",
        ";",
        "؟",
        "،",
        "؛",
        "ـ",
        "…",
        "«",
        "»",
        "–",
        "ً",
        "ٌ",
        "َ",
        "ُ",
        "ِ",
        "ّ",
        "ْ",
        "ٔ",
    ]
)


def nemo_paper_normalize(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(w for w in text.split() if not w.startswith("#"))
    for key, value in REPLACEMENTS.items():
        text = text.replace(key, value)
    for token in DISCARD:
        text = text.replace(token, " ")
    text = text.replace("ء", "")
    text = " ".join(text.split())
    return text


def _alignment_error_rate(
    reference: Sequence[str], hypothesis: Sequence[str]
) -> tuple[float, int, int, int, int]:
    n = len(reference)
    m = len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ops: list[list[str | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        ops[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        ops[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = "C"
            else:
                sub = dp[i - 1][j - 1] + 1
                delete = dp[i - 1][j] + 1
                insert = dp[i][j - 1] + 1
                best = min(sub, delete, insert)
                dp[i][j] = best
                if best == sub:
                    ops[i][j] = "S"
                elif best == delete:
                    ops[i][j] = "D"
                else:
                    ops[i][j] = "I"
    i = n
    j = m
    substitutions = deletions = insertions = 0
    while i > 0 or j > 0:
        operation = ops[i][j]
        if operation == "C":
            i -= 1
            j -= 1
        elif operation == "S":
            substitutions += 1
            i -= 1
            j -= 1
        elif operation == "D":
            deletions += 1
            i -= 1
        elif operation == "I":
            insertions += 1
            j -= 1
        else:
            break
    errors = substitutions + deletions + insertions
    error_rate = errors / n if n > 0 else float(errors > 0)
    return error_rate, substitutions, deletions, insertions, n


def compute_dataset_wer_cer(
    df: pd.DataFrame,
    ref_col: str = "ref_norm",
    hyp_col: str = "hyp_norm",
) -> dict[str, float | int]:
    total_word_sub = total_word_del = total_word_ins = total_words = 0
    total_char_sub = total_char_del = total_char_ins = total_chars = 0
    for ref, hyp in zip(df[ref_col], df[hyp_col]):
        ref = "" if pd.isna(ref) else str(ref)
        hyp = "" if pd.isna(hyp) else str(hyp)
        _, substitutions, deletions, insertions, length = _alignment_error_rate(
            ref.split(), hyp.split()
        )
        total_word_sub += substitutions
        total_word_del += deletions
        total_word_ins += insertions
        total_words += length
        _, substitutions, deletions, insertions, length = _alignment_error_rate(
            list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
        )
        total_char_sub += substitutions
        total_char_del += deletions
        total_char_ins += insertions
        total_chars += length
    return {
        "WER": (total_word_sub + total_word_del + total_word_ins)
        / max(1, total_words),
        "CER": (total_char_sub + total_char_del + total_char_ins)
        / max(1, total_chars),
        "word_sub": total_word_sub,
        "word_del": total_word_del,
        "word_ins": total_word_ins,
        "char_sub": total_char_sub,
        "char_del": total_char_del,
        "char_ins": total_char_ins,
        "total_words": total_words,
        "total_chars": total_chars,
    }


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path, condition: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path, encoding="utf-8-sig", dtype={"id": str, "file_name": str}
    )
    hypothesis_column = "asr_raw" if condition == "original_asr" else "prediction_raw"
    required = {"id", "file_name", "reference_normalized", hypothesis_column, "status"}
    if missing := required - set(frame.columns):
        raise ValueError(f"{condition} checkpoint missing columns: {sorted(missing)}")
    if len(frame) != 871 or frame["file_name"].nunique() != 871:
        raise ValueError(f"{condition} checkpoint must have 871 unique recordings")
    if frame["id"].nunique() != 324 or not frame["status"].eq("ok").all():
        raise ValueError(f"{condition} checkpoint ID/status validation failed")
    frame["ref_norm"] = frame["reference_normalized"].map(nemo_paper_normalize)
    frame["hyp_norm"] = frame[hypothesis_column].map(nemo_paper_normalize)
    row_values = []
    for reference, hypothesis in zip(frame["ref_norm"], frame["hyp_norm"]):
        word = _alignment_error_rate(reference.split(), hypothesis.split())
        char = _alignment_error_rate(
            list(reference.replace(" ", "")), list(hypothesis.replace(" ", ""))
        )
        row_values.append(
            {
                "wer_nemo_paper": word[0],
                "cer_nemo_paper": char[0],
                "word_sub": word[1],
                "word_del": word[2],
                "word_ins": word[3],
                "reference_words": word[4],
                "char_sub": char[1],
                "char_del": char[2],
                "char_ins": char[3],
                "reference_chars": char[4],
            }
        )
    metrics = pd.DataFrame(row_values)
    return pd.concat(
        [frame[["id", "file_name", "ref_norm", "hyp_norm"]].reset_index(drop=True), metrics],
        axis=1,
    )


def update_joined(
    joined_path: Path, recording_metrics: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    joined = pd.read_csv(
        joined_path,
        encoding="utf-8-sig",
        dtype={"id": str, "qa_id": str, "file_name": str},
    )
    if len(joined) != 769 or joined["file_name"].nunique() != 769:
        raise ValueError(f"Joined table has unexpected dimensions: {joined_path}")
    metrics = recording_metrics[
        ["id", "file_name", "wer_nemo_paper", "cer_nemo_paper"]
    ]
    updated = joined.merge(
        metrics,
        on=["id", "file_name"],
        how="left",
        validate="one_to_one",
    )
    if updated[["wer_nemo_paper", "cer_nemo_paper"]].isna().any().any():
        raise ValueError("At least one joined row has no NeMo-paper metric")
    updated = updated.rename(columns={"wer": "wer_previous", "cer": "cer_previous"})
    updated["wer"] = updated.pop("wer_nemo_paper")
    updated["cer"] = updated.pop("cer_nemo_paper")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(output_path, index=False, encoding="utf-8-sig")
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-checkpoint", type=Path, default=DEFAULT_ORIGINAL_CHECKPOINT)
    parser.add_argument("--vosk-checkpoint", type=Path, default=DEFAULT_VOSK_CHECKPOINT)
    parser.add_argument("--original-joined", type=Path, default=DEFAULT_ORIGINAL_JOINED)
    parser.add_argument("--vosk-joined", type=Path, default=DEFAULT_VOSK_JOINED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_paths = {
        "original_checkpoint": args.original_checkpoint,
        "vosk_checkpoint": args.vosk_checkpoint,
        "original_joined": args.original_joined,
        "vosk_joined": args.vosk_joined,
    }
    source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
    original = load_checkpoint(args.original_checkpoint, "original_asr")
    vosk = load_checkpoint(args.vosk_checkpoint, "vosk")
    if not original[["id", "file_name"]].equals(vosk[["id", "file_name"]]):
        raise ValueError("Checkpoint recording order or identifiers disagree")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_metrics_path = args.output_dir / "original_asr_recording_metrics_v1.csv"
    vosk_metrics_path = args.output_dir / "vosk_recording_metrics_v1.csv"
    original_joined_path = args.output_dir / "original_asr_final_joined_nemo_paper_v1.csv"
    vosk_joined_path = args.output_dir / "vosk_final_joined_nemo_paper_v1.csv"
    original_predictions_path = args.output_dir / "original_asr_predictions_nemo_paper_v1.csv"
    vosk_predictions_path = args.output_dir / "vosk_predictions_nemo_paper_v1.csv"
    original.to_csv(original_metrics_path, index=False, encoding="utf-8-sig")
    vosk.to_csv(vosk_metrics_path, index=False, encoding="utf-8-sig")
    original_updated = update_joined(args.original_joined, original, original_joined_path)
    vosk_updated = update_joined(args.vosk_joined, vosk, vosk_joined_path)
    oracle_derived_columns = [
        "oracle_answer_raw",
        "oracle_answer_normalized",
        "oracle_em",
        "oracle_f1",
        "delta_f1",
        "propagation_failure",
    ]
    original_updated.drop(
        columns=[column for column in oracle_derived_columns if column in original_updated],
    ).to_csv(original_predictions_path, index=False, encoding="utf-8-sig")
    vosk_updated.drop(
        columns=[column for column in oracle_derived_columns if column in vosk_updated],
    ).to_csv(vosk_predictions_path, index=False, encoding="utf-8-sig")

    matched_original = original[original["file_name"].isin(original_updated["file_name"])]
    matched_vosk = vosk[vosk["file_name"].isin(vosk_updated["file_name"])]
    summary: dict[str, Any] = {
        "evaluation": "nemo_paper_normalization_v1",
        "normalizer": "nemo_paper_normalize exactly as supplied",
        "skip_note": (
            "SKIP is declared but unused because the supplied nemo_paper_normalize "
            "function does not reference it."
        ),
        "cer_spaces": "excluded",
        "original_asr": {
            "full_871": compute_dataset_wer_cer(original),
            "matched_769": compute_dataset_wer_cer(matched_original),
            "mean_recording_wer_matched": float(original_updated["wer"].mean()),
            "mean_recording_cer_matched": float(original_updated["cer"].mean()),
        },
        "vosk": {
            "full_871": compute_dataset_wer_cer(vosk),
            "matched_769": compute_dataset_wer_cer(matched_vosk),
            "mean_recording_wer_matched": float(vosk_updated["wer"].mean()),
            "mean_recording_cer_matched": float(vosk_updated["cer"].mean()),
        },
        "source_hashes": source_hashes,
        "output_hashes": {
            "original_recording_metrics": sha256_file(original_metrics_path),
            "vosk_recording_metrics": sha256_file(vosk_metrics_path),
            "original_joined": sha256_file(original_joined_path),
            "vosk_joined": sha256_file(vosk_joined_path),
            "original_predictions": sha256_file(original_predictions_path),
            "vosk_predictions": sha256_file(vosk_predictions_path),
        },
    }
    summary_path = args.output_dir / "nemo_paper_metrics_summary_v1.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    for name, path in source_paths.items():
        if sha256_file(path) != source_hashes[name]:
            raise RuntimeError(f"Frozen source changed during evaluation: {name}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
