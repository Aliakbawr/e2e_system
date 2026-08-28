"""Record a reproducible evaluation snapshot for one enhancement stage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import re
import subprocess
import unicodedata
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark.chatbot_multiturn.run_evaluation import evaluate


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[1]
DEFAULT_MULTITURN_DATASET = (
    PROJECT_DIR / "benchmark/chatbot_multiturn/cases_v1.json"
)
DEFAULT_PROPAGATION_DATASET = PROJECT_DIR / (
    "benchmark/asr_llm_error_propagation/analysis/inputs/"
    "fleurs_asr_qa_eval_input_v1.csv"
)
DEFAULT_RESULTS_DIR = BASE_DIR / "results"
STAGE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
DIGIT_MAP = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)


def normalize_for_frozen_eval(value: Any) -> str:
    """Mirror the frozen propagation benchmark's Persian normalization."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
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


def _token_edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for reference_index, reference_token in enumerate(reference, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_token in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    previous[hypothesis_index - 1]
                    + (reference_token != hypothesis_token),
                )
            )
        previous = current
    return previous[-1]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _identity(text: str) -> str:
    return text


def load_processor(name: str) -> tuple[Callable[[str], str], str]:
    if name == "identity":
        return _identity, "identity"
    if name != "runtime":
        raise ValueError(f"Unknown transcript processor: {name}")

    module = importlib.import_module("src.asr.text")
    processor = getattr(module, "preprocess_asr_text", None)
    if not callable(processor):
        raise AttributeError(
            "src.asr.text.preprocess_asr_text must be callable for --processor runtime"
        )
    return processor, "src.asr.text.preprocess_asr_text"


def evaluate_propagation_proxy(
    dataset_path: Path,
    processor: Callable[[str], str],
    processor_name: str,
) -> dict[str, Any]:
    """Measure transcript-level effects without making expensive LLM calls."""
    with dataset_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Propagation dataset is empty: {dataset_path}")

    required = {
        "id",
        "file_name",
        "reference_normalized",
        "asr_raw",
        "asr_eval",
        "gold_answer_normalized",
        "answer_preserved",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Propagation dataset is missing columns: {sorted(missing)}")

    paired_rows = []
    frozen_mismatches = 0
    before_preserved_count = 0
    after_preserved_count = 0
    before_distance_total = 0
    after_distance_total = 0
    reference_token_total = 0
    raw_changes = 0
    eval_changes = 0
    gained = 0
    lost = 0
    closer = 0
    farther = 0

    for row in rows:
        raw = row["asr_raw"]
        processed = processor(raw)
        if not isinstance(processed, str):
            raise TypeError(
                f"Processor returned {type(processed).__name__} for {row['file_name']}"
            )

        before_eval = normalize_for_frozen_eval(raw)
        after_eval = normalize_for_frozen_eval(processed)
        reference_eval = normalize_for_frozen_eval(row["reference_normalized"])
        gold_eval = normalize_for_frozen_eval(row["gold_answer_normalized"])
        before_preserved = bool(gold_eval) and gold_eval in before_eval
        after_preserved = bool(gold_eval) and gold_eval in after_eval

        if (
            before_eval != row["asr_eval"]
            or before_preserved != _bool_value(row["answer_preserved"])
        ):
            frozen_mismatches += 1

        reference_tokens = reference_eval.split()
        before_distance = _token_edit_distance(reference_tokens, before_eval.split())
        after_distance = _token_edit_distance(reference_tokens, after_eval.split())

        raw_changed = raw != processed
        eval_changed = before_eval != after_eval
        raw_changes += raw_changed
        eval_changes += eval_changed
        before_preserved_count += before_preserved
        after_preserved_count += after_preserved
        before_distance_total += before_distance
        after_distance_total += after_distance
        reference_token_total += len(reference_tokens)
        gained += not before_preserved and after_preserved
        lost += before_preserved and not after_preserved
        closer += after_distance < before_distance
        farther += after_distance > before_distance

        paired_rows.append(
            {
                "id": row["id"],
                "file_name": row["file_name"],
                "raw_changed": raw_changed,
                "normalized_text_changed": eval_changed,
                "answer_preserved_before": before_preserved,
                "answer_preserved_after": after_preserved,
                "token_edit_distance_before": before_distance,
                "token_edit_distance_after": after_distance,
            }
        )

    total = len(rows)
    if frozen_mismatches:
        raise ValueError(
            f"Frozen protocol validation failed for {frozen_mismatches} rows; "
            "the input or normalization no longer matches the benchmark"
        )

    return {
        "kind": "lexical_intermediate_proxy",
        "processor": processor_name,
        "interpretation": (
            "This measures transcript and answer-span effects only. It is not an "
            "LLM QA score; use a full frozen Gemma replay for downstream evidence."
        ),
        "summary": {
            "rows": total,
            "semantic_ids": len({row["id"] for row in rows}),
            "raw_transcripts_changed": raw_changes,
            "normalized_transcripts_changed": eval_changes,
            "answer_span_rate_before": _rate(before_preserved_count, total),
            "answer_span_rate_after": _rate(after_preserved_count, total),
            "answer_spans_gained": gained,
            "answer_spans_lost": lost,
            "corpus_token_error_rate_before": _rate(
                before_distance_total, reference_token_total
            ),
            "corpus_token_error_rate_after": _rate(
                after_distance_total, reference_token_total
            ),
            "rows_closer_to_reference": closer,
            "rows_farther_from_reference": farther,
            "rows_same_distance": total - closer - farther,
        },
        "rows": paired_rows,
    }


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=PROJECT_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    # Preserve the leading status column from `git status --porcelain`.
    return completed.stdout.rstrip()


def source_state() -> dict[str, Any]:
    changed_paths = [
        line[3:]
        for line in _git_output("status", "--porcelain").splitlines()
        if len(line) > 3
    ]
    return {
        "git_revision": _git_output("rev-parse", "HEAD"),
        "dirty": bool(changed_paths),
        "changed_paths": changed_paths,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record one immutable chatbot enhancement evaluation stage."
    )
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument(
        "--processor",
        choices=("identity", "runtime"),
        default="runtime",
        help="Transcript processor used for the frozen propagation proxy.",
    )
    parser.add_argument("--multiturn-dataset", type=Path, default=DEFAULT_MULTITURN_DATASET)
    parser.add_argument("--propagation-dataset", type=Path, default=DEFAULT_PROPAGATION_DATASET)
    parser.add_argument(
        "--llm-result",
        type=Path,
        help="Optional existing noisy multi-turn LLM result to attach.",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Run the configured local LLM now instead of attaching a result.",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    if not STAGE_ID_PATTERN.fullmatch(args.stage_id):
        raise SystemExit(
            "--stage-id must contain lowercase letters, digits, underscores, or hyphens"
        )
    if args.with_llm and args.llm_result:
        raise SystemExit("Use either --with-llm or --llm-result, not both")

    output_path = args.results_dir / f"{args.stage_id}.json"
    if output_path.exists():
        raise SystemExit(f"Stage already exists and will not be overwritten: {output_path}")

    multiturn_dataset = load_json(args.multiturn_dataset)
    control_result = evaluate(multiturn_dataset, with_llm=False)
    llm_result = None
    llm_source = None
    if args.with_llm:
        llm_result = evaluate(multiturn_dataset, with_llm=True)
        llm_source = "live_run"
    elif args.llm_result:
        llm_result = load_json(args.llm_result)
        if llm_result.get("dataset_version") != multiturn_dataset.get("version"):
            raise ValueError("Attached LLM result uses a different dataset version")
        llm_source = str(args.llm_result.resolve())

    processor, processor_name = load_processor(args.processor)
    artifact = {
        "schema_version": "enhancement-stage-v1",
        "stage": {
            "id": args.stage_id,
            "description": args.description,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        },
        "source_state": source_state(),
        "datasets": {
            "multiturn": {
                "path": str(args.multiturn_dataset.resolve()),
                "sha256": sha256_file(args.multiturn_dataset),
            },
            "propagation": {
                "path": str(args.propagation_dataset.resolve()),
                "sha256": sha256_file(args.propagation_dataset),
            },
        },
        "multiturn": {
            "control": control_result,
            "llm": llm_result,
            "llm_source": llm_source,
        },
        "propagation_proxy": evaluate_propagation_proxy(
            args.propagation_dataset, processor, processor_name
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    displayed = {
        "stage": artifact["stage"],
        "source_state": artifact["source_state"],
        "multiturn_summary": control_result["summary"],
        "llm_summary": llm_result["summary"] if llm_result else None,
        "propagation_proxy_summary": artifact["propagation_proxy"]["summary"],
        "output": str(output_path),
    }
    print(json.dumps(displayed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
