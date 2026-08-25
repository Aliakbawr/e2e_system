#!/usr/bin/env python3
"""Run and freeze one Gemma 2 9B Oracle prediction per semantic QA item."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPOSITORY_DIR = PROJECT_DIR.parent
DEFAULT_INPUT = PROJECT_DIR / "analysis/inputs/fleurs_asr_qa_eval_input_v1.csv"
DEFAULT_MODEL = REPOSITORY_DIR / "models/llm/gemma-2-9b-it-4bit"
DEFAULT_OUTPUT = SCRIPT_DIR / "gemma2_9b_oracle_predictions_v1.csv"
DEFAULT_METADATA = SCRIPT_DIR / "gemma2_9b_oracle_predictions_v1.metadata.json"
MAX_NEW_TOKENS = 150

PROMPT_TEMPLATE = """متن زیر تنها منبع اطلاعات شماست.

فقط بر اساس اطلاعات موجود در متن به سؤال پاسخ دهید.
از اطلاعات خارج از متن استفاده نکنید و حدس نزنید.

پاسخ را تا حد امکان کوتاه و مستقیم بنویسید.
اگر پاسخ از متن قابل تعیین نیست، فقط بنویسید:
نامشخص

متن:
{context}

سؤال:
{question}

پاسخ:"""

OUTPUT_COLUMNS = (
    "id",
    "qa_id",
    "question",
    "gold_answer",
    "oracle_answer_raw",
    "oracle_answer_normalized",
    "oracle_em",
    "oracle_f1",
    "latency_sec",
    "output_tokens",
    "status",
    "error",
)

DIGIT_MAP = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)


def normalize_persian_qa(text: Any) -> str:
    """Normalize Persian answers for EM/F1 without modifying stored raw text."""
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    for source, target in {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "\u200c": " ",
        "\u200d": "",
        "\ufeff": "",
        "ـ": "",
    }.items():
        text = text.replace(source, target)
    text = text.translate(DIGIT_MAP)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip().lower()


def exact_match_and_f1(prediction: Any, gold: Any) -> tuple[int, float]:
    prediction_normalized = normalize_persian_qa(prediction)
    gold_normalized = normalize_persian_qa(gold)
    exact_match = int(prediction_normalized == gold_normalized)
    prediction_tokens = prediction_normalized.split()
    gold_tokens = gold_normalized.split()
    if not prediction_tokens or not gold_tokens:
        return exact_match, float(exact_match)
    overlap = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
    if overlap == 0:
        return exact_match, 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return exact_match, 2 * precision * recall / (precision + recall)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    """Hash model files and relative names in deterministic order."""
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def prompt_sha256() -> str:
    return hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def load_oracle_items(input_path: Path) -> list[dict[str, str]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Frozen downstream table not found: {input_path}")
    dataframe = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        dtype={"id": str, "qa_id": str, "file_name": str},
    )
    required = {"id", "qa_id", "question", "gold_answer", "reference_normalized"}
    if missing := required - set(dataframe.columns):
        raise ValueError(f"Downstream table is missing columns: {sorted(missing)}")

    consistency_columns = ["qa_id", "question", "gold_answer", "reference_normalized"]
    inconsistent = [
        semantic_id
        for semantic_id, group in dataframe.groupby("id", sort=False)
        if any(group[column].nunique(dropna=False) != 1 for column in consistency_columns)
    ]
    if inconsistent:
        raise ValueError(
            f"Repeated recordings disagree for {len(inconsistent)} semantic IDs"
        )

    items = (
        dataframe[["id", *consistency_columns]]
        .drop_duplicates(subset=["id"], keep="first")
        .sort_values("qa_id", key=lambda values: pd.to_numeric(values, errors="raise"))
    )
    if len(items) != 286:
        raise ValueError(f"Expected 286 Oracle semantic items, found {len(items)}")
    if items[["question", "gold_answer", "reference_normalized"]].isna().any().any():
        raise ValueError("Oracle items contain missing context, question, or gold answer")
    return items.astype(str).to_dict(orient="records")


def completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != OUTPUT_COLUMNS:
            raise ValueError("Oracle checkpoint schema mismatch; refusing to overwrite it")
        rows = list(reader)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Oracle checkpoint contains duplicate semantic IDs")
    return set(ids)


def append_row(output_path: Path, row: dict[str, Any], write_header: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def load_model(model_path: Path) -> tuple[Any, Any, Any]:
    if not model_path.is_dir():
        raise FileNotFoundError(f"Gemma model directory not found: {model_path}")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("PyTorch and Transformers are required") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the local 4-bit Gemma 2 9B model")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map={"": "cuda"},
        dtype=torch.float16,
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer, torch


def generate_oracle_answer(
    model: Any,
    tokenizer: Any,
    torch: Any,
    context: str,
    question: str,
) -> tuple[str, float, int]:
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    latency = time.perf_counter() - started
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    raw_answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
    special_ids = set(tokenizer.all_special_ids)
    output_tokens = sum(int(token_id) not in special_ids for token_id in generated_ids)
    return raw_answer, latency, output_tokens


def build_metadata(
    args: argparse.Namespace,
    input_hash: str,
    model_hash: str,
    output_hash: str,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    successful = [row for row in rows if row["status"] == "ok"]
    em_values = [int(row["oracle_em"]) for row in successful]
    f1_values = [float(row["oracle_f1"]) for row in successful]
    return {
        "stage": "gemma2_9b_oracle_v1",
        "condition": "oracle",
        "calls_expected": 286,
        "calls_recorded": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "model": "Gemma 2 9B IT, local NF4 4-bit archive",
        "model_path": str(args.model.resolve()),
        "model_directory_sha256": model_hash,
        "input_path": str(args.input.resolve()),
        "input_sha256": input_hash,
        "output_path": str(args.output.resolve()),
        "output_sha256": output_hash,
        "prompt_template_sha256": prompt_sha256(),
        "prompt_template": PROMPT_TEMPLATE,
        "context_column": "reference_normalized",
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
            "normalizer": "normalize_persian_qa in run_gemma2_9b_oracle.py",
            "em": "exact equality after normalization",
            "f1": "whitespace-token multiset overlap after normalization",
            "oracle_correct_definition_for_propagation": "oracle_em == 1",
        },
        "descriptive_metrics": {
            "oracle_em_rate": sum(em_values) / max(1, len(em_values)),
            "oracle_mean_f1": sum(f1_values) / max(1, len(f1_values)),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the frozen input and scoring functions without loading Gemma.",
    )
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    input_hash = sha256_file(args.input)
    items = load_oracle_items(args.input)
    expected_ids = {item["id"] for item in items}
    recorded_ids = completed_ids(args.output)
    if unknown := recorded_ids - expected_ids:
        raise ValueError(f"Oracle checkpoint contains {len(unknown)} unknown IDs")

    if args.validate_only:
        print(
            f"Validation passed: {len(items)} unique Oracle items; "
            f"input SHA-256={input_hash}; prompt SHA-256={prompt_sha256()}"
        )
        return 0
    if len(recorded_ids) == len(items):
        print(f"Oracle checkpoint is already complete: {args.output.resolve()}")
        return 0

    remaining = [item for item in items if item["id"] not in recorded_ids]
    model, tokenizer, torch = load_model(args.model.resolve())
    print(f"Gemma 2 9B Oracle | CUDA | remaining: {len(remaining)}/286")
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
            raw_answer, latency, output_tokens = generate_oracle_answer(
                model,
                tokenizer,
                torch,
                item["reference_normalized"],
                item["question"],
            )
            normalized_answer = normalize_persian_qa(raw_answer)
            em, f1 = exact_match_and_f1(raw_answer, item["gold_answer"])
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        row = {
            "id": item["id"],
            "qa_id": item["qa_id"],
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "oracle_answer_raw": raw_answer,
            "oracle_answer_normalized": normalized_answer,
            "oracle_em": em,
            "oracle_f1": "" if f1 == "" else f"{float(f1):.9f}",
            "latency_sec": "" if latency == "" else f"{float(latency):.9f}",
            "output_tokens": output_tokens,
            "status": status,
            "error": error,
        }
        append_row(args.output, row, write_header)
        write_header = False
        print(f"[{len(recorded_ids) + index:03d}/286] id={item['id']}: {status}", flush=True)

    if sha256_file(args.input) != input_hash:
        raise RuntimeError("Frozen downstream input changed during Oracle inference")
    with args.output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 286 or len({row["id"] for row in rows}) != 286:
        raise RuntimeError("Oracle checkpoint is incomplete or contains duplicate IDs")

    model_hash = sha256_directory(args.model.resolve())
    metadata = build_metadata(args, input_hash, model_hash, sha256_file(args.output), rows)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"Frozen Oracle predictions: {args.output.resolve()}")
    print(f"Oracle metadata: {args.metadata.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
