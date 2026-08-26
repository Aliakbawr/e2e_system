#!/usr/bin/env python3
"""Run one frozen-prompt Gemma 2 9B call for each of 769 ASR transcripts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from benchmark.asr_llm_error_propagation.llm.run_gemma2_9b_oracle import (
    MAX_NEW_TOKENS,
    PROMPT_TEMPLATE,
    exact_match_and_f1,
    generate_oracle_answer,
    load_model,
    normalize_persian_qa,
    prompt_sha256,
    sha256_directory,
    sha256_file,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPOSITORY_DIR = PROJECT_DIR.parents[1]
DEFAULT_INPUT = PROJECT_DIR / "analysis/inputs/fleurs_asr_qa_eval_input_v1.csv"
DEFAULT_ORACLE = SCRIPT_DIR / "gemma2_9b_oracle_predictions_v1.csv"
DEFAULT_MODEL = REPOSITORY_DIR / "models/llm/gemma-2-9b-it-4bit"
DEFAULT_OUTPUT = SCRIPT_DIR / "gemma2_9b_asr_path_predictions_v1.csv"
DEFAULT_METADATA = SCRIPT_DIR / "gemma2_9b_asr_path_predictions_v1.metadata.json"

OUTPUT_COLUMNS = (
    "qa_id",
    "id",
    "file_name",
    "question",
    "gold_answer",
    "reference_normalized",
    "asr_normalized",
    "wer",
    "cer",
    "answer_preserved",
    "gold_answer_token_count",
    "asr_llm_answer_raw",
    "asr_llm_answer_normalized",
    "asr_em",
    "asr_f1",
    "latency_sec",
    "output_tokens",
    "status",
    "error",
)


def load_asr_items(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Frozen downstream table not found: {input_path}")
    dataframe = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        dtype={"id": str, "qa_id": str, "file_name": str},
    )
    required = {
        "qa_id",
        "id",
        "file_name",
        "question",
        "gold_answer",
        "reference_normalized",
        "asr_normalized",
        "wer",
        "cer",
        "answer_preserved",
    }
    if missing := required - set(dataframe.columns):
        raise ValueError(f"Downstream table is missing columns: {sorted(missing)}")
    if len(dataframe) != 769:
        raise ValueError(f"Expected 769 ASR-path rows, found {len(dataframe)}")
    if dataframe["id"].nunique() != 286:
        raise ValueError("Expected 286 unique semantic IDs")
    if dataframe["file_name"].duplicated().any():
        raise ValueError("Downstream table contains duplicate filenames")
    if dataframe[list(required)].isna().any().any():
        raise ValueError("Downstream table contains missing required values")

    dataframe["gold_answer_token_count"] = dataframe["gold_answer"].map(
        lambda answer: len(normalize_persian_qa(answer).split())
    )
    if (dataframe["gold_answer_token_count"] < 1).any():
        raise ValueError("At least one gold answer normalizes to zero tokens")
    return dataframe.to_dict(orient="records")


def validate_frozen_oracle(oracle_path: Path, expected_ids: set[str]) -> str:
    if not oracle_path.is_file():
        raise FileNotFoundError(f"Frozen Oracle predictions not found: {oracle_path}")
    oracle = pd.read_csv(oracle_path, dtype={"id": str, "qa_id": str})
    if len(oracle) != 286 or oracle["id"].nunique() != 286:
        raise ValueError("Frozen Oracle file must contain exactly 286 unique IDs")
    if set(oracle["id"]) != expected_ids:
        raise ValueError("Oracle IDs disagree with ASR-path semantic IDs")
    if not (oracle["status"] == "ok").all():
        raise ValueError("Frozen Oracle file contains failed rows")
    return sha256_file(oracle_path)


def completed_filenames(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != OUTPUT_COLUMNS:
            raise ValueError("ASR-path checkpoint schema mismatch; refusing overwrite")
        rows = list(reader)
    filenames = [row["file_name"] for row in rows]
    if len(filenames) != len(set(filenames)):
        raise ValueError("ASR-path checkpoint contains duplicate filenames")
    return set(filenames)


def append_row(output_path: Path, row: dict[str, Any], write_header: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    input_hash = sha256_file(args.input)
    items = load_asr_items(args.input)
    expected_filenames = {str(item["file_name"]) for item in items}
    expected_ids = {str(item["id"]) for item in items}
    oracle_hash = validate_frozen_oracle(args.oracle, expected_ids)
    recorded = completed_filenames(args.output)
    if unknown := recorded - expected_filenames:
        raise ValueError(f"ASR-path checkpoint contains {len(unknown)} unknown files")

    if args.validate_only:
        print(
            f"Validation passed: 769 rows, 286 IDs; input SHA-256={input_hash}; "
            f"Oracle SHA-256={oracle_hash}; prompt SHA-256={prompt_sha256()}"
        )
        return 0
    if len(recorded) == len(items):
        print(f"ASR-path checkpoint is already complete: {args.output.resolve()}")
        return 0

    remaining = [item for item in items if str(item["file_name"]) not in recorded]
    model, tokenizer, torch = load_model(args.model.resolve())
    print(f"Gemma 2 9B ASR path | CUDA | remaining: {len(remaining)}/769")
    write_header = not args.output.exists()
    for index, item in enumerate(remaining, start=1):
        raw_answer = ""
        normalized_answer = ""
        em: int | str = ""
        f1: float | str = ""
        latency: float | str = ""
        output_tokens: int | str = ""
        status = "ok"
        error = ""
        try:
            # This is the only experimental difference from Oracle inference.
            raw_answer, latency, output_tokens = generate_oracle_answer(
                model,
                tokenizer,
                torch,
                str(item["asr_normalized"]),
                str(item["question"]),
            )
            normalized_answer = normalize_persian_qa(raw_answer)
            em, f1 = exact_match_and_f1(raw_answer, item["gold_answer"])
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        row = {
            "qa_id": item["qa_id"],
            "id": item["id"],
            "file_name": item["file_name"],
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "reference_normalized": item["reference_normalized"],
            "asr_normalized": item["asr_normalized"],
            "wer": item["wer"],
            "cer": item["cer"],
            "answer_preserved": item["answer_preserved"],
            "gold_answer_token_count": item["gold_answer_token_count"],
            "asr_llm_answer_raw": raw_answer,
            "asr_llm_answer_normalized": normalized_answer,
            "asr_em": em,
            "asr_f1": "" if f1 == "" else f"{float(f1):.9f}",
            "latency_sec": "" if latency == "" else f"{float(latency):.9f}",
            "output_tokens": output_tokens,
            "status": status,
            "error": error,
        }
        append_row(args.output, row, write_header)
        write_header = False
        print(
            f"[{len(recorded) + index:03d}/769] {item['file_name']}: {status}",
            flush=True,
        )

    if sha256_file(args.input) != input_hash:
        raise RuntimeError("Frozen downstream input changed during ASR-path inference")
    if sha256_file(args.oracle) != oracle_hash:
        raise RuntimeError("Frozen Oracle predictions changed during ASR-path inference")
    with args.output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 769 or len({row["file_name"] for row in rows}) != 769:
        raise RuntimeError("ASR-path checkpoint is incomplete or duplicated")

    successful = [row for row in rows if row["status"] == "ok"]
    metadata = {
        "stage": "gemma2_9b_asr_path_v1",
        "condition": "asr_path",
        "calls_expected": 769,
        "calls_recorded": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "model": "Gemma 2 9B IT, local NF4 4-bit archive",
        "model_path": str(args.model.resolve()),
        "model_directory_sha256": sha256_directory(args.model.resolve()),
        "input_path": str(args.input.resolve()),
        "input_sha256": input_hash,
        "oracle_path": str(args.oracle.resolve()),
        "oracle_sha256": oracle_hash,
        "output_path": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "prompt_template_sha256": prompt_sha256(),
        "prompt_template": PROMPT_TEMPLATE,
        "only_condition_difference": (
            "context=asr_normalized instead of context=reference_normalized"
        ),
        "context_column": "asr_normalized",
        "question_column": "question",
        "generation": {
            "chat_template": True,
            "add_generation_prompt": True,
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "input_max_length": 2048,
            "pad_token_id": "tokenizer.eos_token_id",
        },
        "scoring": {
            "normalizer": "normalize_persian_qa imported from frozen Oracle runner",
            "em": "exact equality after normalization",
            "f1": "whitespace-token multiset overlap after normalization",
        },
        "descriptive_metrics": {
            "asr_em_rate_all_769": sum(int(row["asr_em"]) for row in successful)
            / max(1, len(successful)),
            "asr_mean_f1_all_769": sum(float(row["asr_f1"]) for row in successful)
            / max(1, len(successful)),
        },
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Frozen ASR-path predictions: {args.output.resolve()}")
    print(f"ASR-path metadata: {args.metadata.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
