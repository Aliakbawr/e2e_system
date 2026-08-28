"""Prepare a 769-row contextual-recovery Gemma condition and reusable checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from benchmark.asr_llm_error_propagation.asr.run_fleurs_fa_ir_asr import (
    calculate_wer_cer,
    normalize_persian_for_evaluation,
)
from benchmark.asr_llm_error_propagation.prepare_downstream_eval import (
    normalize_persian,
    sha256_file,
)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[1]
RESULTS_DIR = BASE_DIR / "results"
DEFAULT_SOURCE = PROJECT_DIR / (
    "benchmark/asr_audio_preprocessing/results/"
    "low_level_gain_fleurs_qa_eval_input_v1.csv"
)
DEFAULT_CASES = RESULTS_DIR / "contextual_recovery_cases_v1.csv"
DEFAULT_BASELINE_PREDICTIONS = PROJECT_DIR / (
    "benchmark/asr_audio_preprocessing/results/"
    "gemma2_9b_low_level_gain_predictions_v1.csv"
)
DEFAULT_OUTPUT = RESULTS_DIR / "contextual_recovery_fleurs_qa_eval_input_v1.csv"
DEFAULT_PREDICTION_CHECKPOINT = RESULTS_DIR / (
    "gemma2_9b_contextual_recovery_predictions_v1.csv"
)


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _write(
    path: Path,
    fields: list[str],
    rows: list[dict],
    encoding: str = "utf-8-sig",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--baseline-predictions", type=Path, default=DEFAULT_BASELINE_PREDICTIONS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--prediction-checkpoint", type=Path, default=DEFAULT_PREDICTION_CHECKPOINT
    )
    args = parser.parse_args()

    source_fields, source_rows = _read(args.source)
    _, cases = _read(args.cases)
    prediction_fields, baseline_predictions = _read(args.baseline_predictions)
    if len(source_rows) != 769 or len(baseline_predictions) != 769:
        raise ValueError("Expected complete source and baseline predictions")
    recovered = {
        row["file_name"]: row
        for row in cases
        if row["target_recovered"].lower() == "true"
        and row["selected_text"] != row["primary_text"]
    }

    output_rows = []
    for source in source_rows:
        row = dict(source)
        case = recovered.get(row["file_name"])
        if case is not None:
            row["asr_raw"] = case["selected_text"]
            row["asr_normalized"] = normalize_persian_for_evaluation(row["asr_raw"])
            row["asr_eval"] = normalize_persian(row["asr_raw"])
            reference = normalize_persian_for_evaluation(row["reference_normalized"])
            row["wer"], row["cer"] = calculate_wer_cer(
                reference, row["asr_normalized"]
            )
            gold = normalize_persian(row["gold_answer_normalized"])
            row["answer_preserved"] = bool(gold) and gold in row["asr_eval"]
        output_rows.append(row)
    _write(args.output, source_fields, output_rows)

    changed = set(recovered)
    reusable = [
        row for row in baseline_predictions if row["file_name"] not in changed
    ]
    if args.prediction_checkpoint.exists():
        raise FileExistsError(
            f"Refusing to overwrite prediction checkpoint: {args.prediction_checkpoint}"
        )
    # Prediction runners read this append-only checkpoint as plain UTF-8.
    _write(args.prediction_checkpoint, prediction_fields, reusable, encoding="utf-8")

    metadata = {
        "stage": "contextual_recovery_gemma_input_v1",
        "recordings": len(output_rows),
        "changed_contexts": len(changed),
        "reused_frozen_predictions": len(reusable),
        "new_gemma_calls_required": len(changed),
        "source_sha256": sha256_file(args.source),
        "cases_sha256": sha256_file(args.cases),
        "output_sha256": sha256_file(args.output),
        "initial_prediction_checkpoint_sha256": sha256_file(
            args.prediction_checkpoint
        ),
    }
    metadata_path = RESULTS_DIR / "contextual_recovery_gemma_input_v1.metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
