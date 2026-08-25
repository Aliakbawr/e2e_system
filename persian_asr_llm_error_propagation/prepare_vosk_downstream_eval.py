#!/usr/bin/env python3
"""Prepare the stored Vosk FLEURS transcripts for frozen Gemma evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from asr.compute_fleurs_metrics import compute_dataset_wer_cer
from asr.run_fleurs_fa_ir_asr import calculate_wer_cer, normalize_persian_for_evaluation
from prepare_downstream_eval import contains_token_span, normalize_persian, sha256_file


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_VOSK = PROJECT_DIR / "asr/vosk_fleurs_checkpoint.csv"
DEFAULT_FROZEN_QA_TABLE = PROJECT_DIR / "analysis/inputs/fleurs_asr_qa_eval_input_v1.csv"
DEFAULT_OUTPUT = PROJECT_DIR / "analysis/inputs/vosk_fleurs_qa_eval_input_v1.csv"
DEFAULT_METADATA = PROJECT_DIR / "analysis/inputs/vosk_fleurs_qa_eval_input_v1.metadata.json"


def load_inputs(vosk_path: Path, frozen_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    vosk = pd.read_csv(
        vosk_path,
        encoding="utf-8-sig",
        dtype={"id": str, "file_name": str},
    )
    frozen = pd.read_csv(
        frozen_path,
        encoding="utf-8-sig",
        dtype={"id": str, "qa_id": str, "file_name": str},
    )
    required_vosk = {
        "id",
        "file_name",
        "audio_path",
        "gender",
        "audio_duration",
        "reference_raw",
        "reference_normalized",
        "prediction_raw",
        "inference_time",
        "rtf",
        "status",
        "error",
    }
    if missing := required_vosk - set(vosk.columns):
        raise ValueError(f"Vosk checkpoint is missing columns: {sorted(missing)}")
    if len(vosk) != 871 or vosk["file_name"].nunique() != 871:
        raise ValueError("Vosk checkpoint must contain 871 unique recordings")
    if vosk["id"].nunique() != 324:
        raise ValueError("Vosk checkpoint must contain 324 semantic IDs")
    if not (vosk["status"] == "ok").all():
        raise ValueError("Vosk checkpoint contains failed recordings")
    if vosk[["id", "file_name", "reference_normalized", "prediction_raw"]].isna().any().any():
        raise ValueError("Vosk checkpoint contains missing required values")

    required_frozen = {
        "qa_id",
        "id",
        "file_name",
        "question",
        "gold_answer",
        "reference_normalized",
    }
    if missing := required_frozen - set(frozen.columns):
        raise ValueError(f"Frozen QA evaluation table is missing: {sorted(missing)}")
    if len(frozen) != 769 or frozen["file_name"].nunique() != 769:
        raise ValueError("Frozen QA evaluation table must contain 769 recordings")
    if frozen["id"].nunique() != 286:
        raise ValueError("Frozen QA evaluation table must contain 286 semantic IDs")
    return vosk, frozen


def prepare(vosk: pd.DataFrame, frozen: pd.DataFrame) -> pd.DataFrame:
    vosk_columns = [
        "id",
        "file_name",
        "audio_path",
        "gender",
        "audio_duration",
        "reference_raw",
        "reference_normalized",
        "prediction_raw",
        "inference_time",
        "rtf",
        "status",
        "error",
    ]
    qa_columns = [
        column
        for column in [
            "qa_id",
            "id",
            "file_name",
            "num_recordings",
            "question",
            "gold_answer",
            "answer_type",
            "review_status",
            "quality_score",
            "qa_schema_version",
        ]
        if column in frozen.columns
    ]
    merged = frozen[qa_columns].merge(
        vosk[vosk_columns],
        on=["id", "file_name"],
        how="left",
        validate="one_to_one",
    )
    if len(merged) != 769 or merged["id"].nunique() != 286:
        raise ValueError("Vosk/QA join did not preserve 769 rows and 286 IDs")
    if merged["prediction_raw"].isna().any():
        raise ValueError("At least one frozen QA recording has no Vosk transcript")

    frozen_refs = frozen.set_index("file_name")["reference_normalized"].map(
        normalize_persian_for_evaluation
    )
    vosk_refs = merged.set_index("file_name")["reference_normalized"].map(
        normalize_persian_for_evaluation
    )
    if not frozen_refs.sort_index().equals(vosk_refs.sort_index()):
        raise ValueError("Vosk and frozen QA references disagree")

    merged = merged.rename(
        columns={
            "audio_duration": "duration_sec",
            "prediction_raw": "asr_raw",
            "inference_time": "latency_sec",
        }
    )
    merged["asr_normalized"] = merged["asr_raw"].map(
        normalize_persian_for_evaluation
    )
    evaluation_references = merged["reference_normalized"].map(
        normalize_persian_for_evaluation
    )
    row_metrics = [
        calculate_wer_cer(reference, hypothesis)
        for reference, hypothesis in zip(
            evaluation_references,
            merged["asr_normalized"],
        )
    ]
    merged["wer"] = [wer for wer, _ in row_metrics]
    merged["cer"] = [cer for _, cer in row_metrics]
    merged["gold_answer_normalized"] = merged["gold_answer"].map(normalize_persian)
    merged["reference_eval"] = merged["reference_normalized"].map(normalize_persian)
    merged["asr_eval"] = merged["asr_raw"].map(normalize_persian)
    merged["answer_preserved"] = [
        gold in hypothesis
        for gold, hypothesis in zip(
            merged["gold_answer_normalized"], merged["asr_eval"]
        )
    ]
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
    if not merged["answer_in_reference"].all():
        raise ValueError("A frozen gold answer was not found in its reference")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vosk", type=Path, default=DEFAULT_VOSK)
    parser.add_argument("--frozen-qa-table", type=Path, default=DEFAULT_FROZEN_QA_TABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    vosk_hash = sha256_file(args.vosk)
    frozen_hash = sha256_file(args.frozen_qa_table)
    vosk, frozen = load_inputs(args.vosk, args.frozen_qa_table)
    merged = prepare(vosk, frozen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False, encoding="utf-8-sig")
    full_scoring = pd.DataFrame(
        {
            "ref": vosk["reference_normalized"].map(normalize_persian_for_evaluation),
            "hyp": vosk["prediction_raw"].map(normalize_persian_for_evaluation),
        }
    )
    matched_scoring = pd.DataFrame(
        {
            "ref": merged["reference_normalized"].map(normalize_persian_for_evaluation),
            "hyp": merged["asr_raw"].map(normalize_persian_for_evaluation),
        }
    )
    metadata: dict[str, Any] = {
        "stage": "vosk_fleurs_qa_eval_input_v1",
        "condition": "vosk_asr_path",
        "recordings": int(len(merged)),
        "semantic_ids": int(merged["id"].nunique()),
        "answer_preserved_recordings": int(merged["answer_preserved"].sum()),
        "answer_preservation_rate": float(merged["answer_preserved"].mean()),
        "mean_recording_wer": float(merged["wer"].mean()),
        "mean_recording_cer": float(merged["cer"].mean()),
        "corpus_metrics_full_871": compute_dataset_wer_cer(full_scoring),
        "corpus_metrics_matched_769": compute_dataset_wer_cer(matched_scoring),
        "vosk_checkpoint_sha256": vosk_hash,
        "frozen_qa_table_sha256": frozen_hash,
        "output_sha256": sha256_file(args.output),
        "normalization": "frozen normalize_persian_for_evaluation",
        "qa_source": "unchanged rows from frozen fleurs_asr_qa_eval_input_v1.csv",
    }
    with args.metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if sha256_file(args.vosk) != vosk_hash or sha256_file(args.frozen_qa_table) != frozen_hash:
        raise RuntimeError("A frozen input changed while preparing the Vosk condition")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
