"""Evaluate explicit-correction N-best recovery on frozen audio evidence."""

from __future__ import annotations

import argparse
import csv
import json
from difflib import SequenceMatcher
from pathlib import Path

from benchmark.asr_llm_error_propagation.asr.run_fleurs_fa_ir_asr import (
    calculate_wer_cer,
    normalize_persian_for_evaluation,
)
from benchmark.asr_llm_error_propagation.prepare_downstream_eval import (
    normalize_persian,
)
from config.settings import ASR_ALTERNATIVE_SCORE_GAP
from src.asr.rerank import rerank_with_session_memory
from src.asr.types import TranscriptAlternative, TranscriptionResult
from src.core.session import ChatSession


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[1]
DEFAULT_EVIDENCE = BASE_DIR / "results/nbest_evidence_checkpoint_v1.csv"
DEFAULT_SOURCE = PROJECT_DIR / (
    "benchmark/asr_audio_preprocessing/results/"
    "low_level_gain_fleurs_qa_eval_input_v1.csv"
)
DEFAULT_RESULTS_DIR = BASE_DIR / "results"
OUTPUT_COLUMNS = (
    "id",
    "file_name",
    "split",
    "correction_key",
    "correction_value",
    "primary_text",
    "target_alternative",
    "primary_wer",
    "target_wer",
    "selected_text",
    "selected_wer",
    "target_recovered",
    "wrong_alternative_selected",
    "no_memory_changed",
    "irrelevant_context_changed",
    "answer_preserved_before",
    "answer_preserved_after",
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _eligible(alternatives: list[dict], primary: str) -> list[dict]:
    scores = [
        item["decoder_score"]
        for item in alternatives
        if item["decoder_score"] is not None
    ]
    best_score = max(scores) if scores else None
    return [
        item
        for item in alternatives
        if item["text"] != primary
        and (
            best_score is None
            or item["decoder_score"] is None
            or best_score - item["decoder_score"] <= ASR_ALTERNATIVE_SCORE_GAP
        )
    ]


def _single_replacement(primary: str, alternative: str) -> tuple[str, str] | None:
    primary_tokens = primary.split()
    alternative_tokens = alternative.split()
    changes = [
        opcode
        for opcode in SequenceMatcher(
            None, primary_tokens, alternative_tokens, autojunk=False
        ).get_opcodes()
        if opcode[0] != "equal"
    ]
    if len(changes) != 1:
        return None
    _, primary_start, primary_end, alt_start, alt_end = changes[0]
    key = " ".join(primary_tokens[primary_start:primary_end]).strip()
    value = " ".join(alternative_tokens[alt_start:alt_end]).strip()
    return (key, value) if key and value else None


def _transcription(primary: str, alternatives: list[dict]) -> TranscriptionResult:
    return TranscriptionResult(
        text=primary,
        alternatives=tuple(
            TranscriptAlternative(
                text=item["text"],
                decoder_score=item["decoder_score"],
            )
            for item in alternatives
        ),
    )


def _summarize(rows: list[dict]) -> dict:
    summaries = {}
    for split in ("development", "held_out", "all"):
        selected = [row for row in rows if split == "all" or row["split"] == split]
        total = len(selected)
        summaries[split] = {
            "positive_recovery_cases": total,
            "target_recovered": sum(row["target_recovered"] for row in selected),
            "target_recovery_rate": (
                sum(row["target_recovered"] for row in selected) / total if total else None
            ),
            "wrong_alternative_selections": sum(
                row["wrong_alternative_selected"] for row in selected
            ),
            "mean_recording_wer_before": (
                sum(row["primary_wer"] for row in selected) / total if total else None
            ),
            "mean_recording_wer_after": (
                sum(row["selected_wer"] for row in selected) / total if total else None
            ),
            "answer_span_rate_before": (
                sum(row["answer_preserved_before"] for row in selected) / total
                if total
                else None
            ),
            "answer_span_rate_after": (
                sum(row["answer_preserved_after"] for row in selected) / total
                if total
                else None
            ),
            "no_memory_control_changes": sum(
                row["no_memory_changed"] for row in selected
            ),
            "irrelevant_context_control_changes": sum(
                row["irrelevant_context_changed"] for row in selected
            ),
        }
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    evidence = _read(args.evidence)
    source = {row["file_name"]: row for row in _read(args.source)}
    if len(evidence) != 769 or len(source) != 769:
        raise ValueError("Expected complete 769-recording evidence and source tables")

    cases = []
    for row in evidence:
        frozen = source[row["file_name"]]
        primary = row["primary_text"]
        reference = normalize_persian_for_evaluation(frozen["reference_normalized"])
        primary_wer = calculate_wer_cer(
            reference, normalize_persian_for_evaluation(primary)
        )[0]
        alternatives = json.loads(row["alternatives_json"])
        better = []
        for item in _eligible(alternatives, primary):
            wer = calculate_wer_cer(
                reference, normalize_persian_for_evaluation(item["text"])
            )[0]
            replacement = _single_replacement(primary, item["text"])
            if wer < primary_wer - 1e-12 and replacement is not None:
                better.append((wer, item, replacement))
        if not better:
            continue
        target_wer, target, (key, value) = min(better, key=lambda item: item[0])
        transcription = _transcription(primary, alternatives)

        session = ChatSession()
        session.remember_correction(key, value, target["text"])
        result = rerank_with_session_memory(transcription, session)

        no_memory = rerank_with_session_memory(transcription, ChatSession())
        irrelevant = ChatSession()
        irrelevant.remember_correction(
            key, value, "موضوع آزمایشی کاملا نامرتبط"
        )
        irrelevant_result = rerank_with_session_memory(transcription, irrelevant)

        selected_text = result.transcription.text
        selected_wer = calculate_wer_cer(
            reference, normalize_persian_for_evaluation(selected_text)
        )[0]
        gold = normalize_persian(frozen["gold_answer_normalized"])
        cases.append(
            {
                "id": row["id"],
                "file_name": row["file_name"],
                "split": row["split"],
                "correction_key": key,
                "correction_value": value,
                "primary_text": primary,
                "target_alternative": target["text"],
                "primary_wer": primary_wer,
                "target_wer": target_wer,
                "selected_text": selected_text,
                "selected_wer": selected_wer,
                "target_recovered": selected_text == target["text"],
                "wrong_alternative_selected": (
                    result.changed and selected_text != target["text"]
                ),
                "no_memory_changed": no_memory.changed,
                "irrelevant_context_changed": irrelevant_result.changed,
                "answer_preserved_before": bool(gold)
                and gold in normalize_persian(primary),
                "answer_preserved_after": bool(gold)
                and gold in normalize_persian(selected_text),
            }
        )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.results_dir / "contextual_recovery_cases_v1.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(cases)

    summary = {
        "stage": "explicit_correction_contextual_nbest_recovery_v1",
        "case_construction": (
            "Reference text labels a better in-gap single-replacement hypothesis "
            "offline; runtime receives only explicit correction memory and N-best evidence."
        ),
        "summary": _summarize(cases),
    }
    summary_path = args.results_dir / "contextual_recovery_summary_v1.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
