"""Run deterministic control and optional local-LLM multi-turn evaluation."""

import argparse
import json
import re
from pathlib import Path

from config.settings import (
    ASR_ALTERNATIVE_SCORE_GAP,
    ASR_CLARIFICATION_MAX_OPTIONS,
    ASR_WORD_CONFIDENCE_THRESHOLD,
    LLM_MODEL_PATH,
    MAX_LLM_INPUT_TOKENS,
    MAX_LLM_TOKENS,
)
from src.asr.types import RecognizedWord, TranscriptAlternative, TranscriptionResult
from src.core.dialogue import interpret_transcription
from src.core.session import ChatSession


DEFAULT_DATASET = Path(__file__).with_name("cases_v1.json")
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def _normalize(text: str) -> str:
    text = str(text).translate(PERSIAN_DIGITS).lower().replace("\u200c", " ")
    text = text.translate(str.maketrans({"ي": "ی", "ك": "ک", "آ": "ا"}))
    return re.sub(r"[^\w]+", " ", text).strip()


def _transcription(turn: dict) -> TranscriptionResult:
    words = tuple(
        RecognizedWord(text=word, confidence=float(confidence))
        for word, confidence in turn.get("words", [])
    )
    alternatives = tuple(
        TranscriptAlternative(text=text, decoder_score=float(score))
        for text, score in turn.get("alternatives", [])
    )
    return TranscriptionResult(
        text=turn.get("transcript", ""),
        words=words,
        alternatives=alternatives,
    )


def _contains_memory(session: ChatSession, expected: dict | None) -> bool | None:
    if expected is None:
        return None
    return any(
        all(getattr(item, key) == value for key, value in expected.items())
        for item in session.memory
    )


def _keyword_score(answer: str, groups: list[list[str]] | None) -> bool | None:
    if not groups:
        return None
    normalized_answer = _normalize(answer)
    return all(
        any(_normalize(keyword) in normalized_answer for keyword in group)
        for group in groups
    )


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _summarize(rows: list[dict], tier: str) -> dict:
    selected = [row for row in rows if row["tier"] == tier]
    expected_clarify = [row for row in selected if row["expected_decision"] == "clarify"]
    predicted_clarify = [row for row in selected if row["actual_decision"] == "clarify"]
    true_clarify = [
        row
        for row in selected
        if row["expected_decision"] == row["actual_decision"] == "clarify"
    ]
    resolved = [row for row in selected if row["resolved_query_correct"] is not None]
    memory = [row for row in selected if row["memory_correct"] is not None]
    llm_rows = [row for row in selected if row["answer_keywords_correct"] is not None]
    return {
        "turns": len(selected),
        "decision_accuracy": _safe_rate(
            sum(row["decision_correct"] for row in selected), len(selected)
        ),
        "clarification_precision": _safe_rate(len(true_clarify), len(predicted_clarify)),
        "clarification_recall": _safe_rate(len(true_clarify), len(expected_clarify)),
        "resolved_query_accuracy": _safe_rate(
            sum(row["resolved_query_correct"] for row in resolved), len(resolved)
        ),
        "memory_accuracy": _safe_rate(
            sum(row["memory_correct"] for row in memory), len(memory)
        ),
        "llm_keyword_turns": len(llm_rows),
        "llm_keyword_accuracy": _safe_rate(
            sum(row["answer_keywords_correct"] for row in llm_rows), len(llm_rows)
        ),
    }


def evaluate(dataset: dict, with_llm: bool = False) -> dict:
    generate_answer = None
    if with_llm:
        from src.llm.generator import generate_answer as local_generate_answer

        generate_answer = local_generate_answer

    rows = []
    for dialogue in dataset["dialogues"]:
        session = ChatSession()
        for turn in dialogue["turns"]:
            for user_message, assistant_message in turn.get("prefill_history", []):
                session.add_turn(user_message, assistant_message)
            decision = interpret_transcription(_transcription(turn), session)
            answer = None
            if decision.action == "answer":
                if generate_answer is not None:
                    generated = generate_answer(
                        decision.effective_question,
                        history=session.messages(),
                        memory=session.memory_prompt(),
                        turn_context=(
                            "این پیام از پاسخ کاربر به رفع ابهام بازسازی شده است. "
                            "در پاسخ، واژه، عدد یا نفی تأییدشده را صریحاً تکرار کن."
                            if decision.resolution is not None
                            and decision.resolution.resolved
                            else None
                        ),
                    )
                    answer = generated["answer"]
                else:
                    answer = turn.get("assistant_context", "پاسخ ثبت‌نشده")
                session.add_turn(decision.effective_question, answer)

            expected_resolved = turn.get("expected_resolved_question")
            actual_resolution = (
                decision.resolution.resolved
                if decision.resolution is not None
                else None
            )
            expected_resolution = turn.get("expected_resolution")
            resolution_state_correct = (
                actual_resolution == expected_resolution
                if expected_resolution is not None
                else None
            )
            expected_options = turn.get("expected_options")
            row = {
                "dialogue_id": dialogue["id"],
                "turn_id": turn["id"],
                "category": dialogue["category"],
                "tier": turn.get("tier", "core"),
                "expected_decision": turn["expected_decision"],
                "actual_decision": decision.action,
                "decision_correct": decision.action == turn["expected_decision"],
                "expected_options": expected_options,
                "actual_options": list(decision.risk.options),
                "options_correct": (
                    list(decision.risk.options) == expected_options
                    if expected_options is not None
                    else None
                ),
                "expected_resolved_question": expected_resolved,
                "actual_resolved_question": decision.effective_question,
                "resolved_query_correct": (
                    decision.effective_question == expected_resolved
                    if expected_resolved is not None
                    else None
                ),
                "resolution_state_correct": resolution_state_correct,
                "memory_correct": _contains_memory(
                    session, turn.get("expected_memory_contains")
                ),
                "answer": answer,
                "answer_keywords_correct": (
                    _keyword_score(answer, turn.get("answer_keyword_groups"))
                    if with_llm and answer is not None
                    else None
                ),
            }
            rows.append(row)

    return {
        "dataset_version": dataset["version"],
        "with_llm": with_llm,
        "configuration": {
            "asr_word_confidence_threshold": ASR_WORD_CONFIDENCE_THRESHOLD,
            "asr_alternative_score_gap": ASR_ALTERNATIVE_SCORE_GAP,
            "asr_clarification_max_options": ASR_CLARIFICATION_MAX_OPTIONS,
            "llm_model_path": str(LLM_MODEL_PATH) if with_llm else None,
            "max_llm_input_tokens": MAX_LLM_INPUT_TOKENS if with_llm else None,
            "max_llm_output_tokens": MAX_LLM_TOKENS if with_llm else None,
        },
        "summary": {
            "core": _summarize(rows, "core"),
            "challenge": _summarize(rows, "challenge"),
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--with-llm", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--show-rows", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    result = evaluate(dataset, with_llm=args.with_llm)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    displayed = result if args.show_rows else {
        "dataset_version": result["dataset_version"],
        "with_llm": result["with_llm"],
        "configuration": result["configuration"],
        "summary": result["summary"],
    }
    print(json.dumps(displayed, ensure_ascii=False, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    if args.strict and result["summary"]["core"]["decision_accuracy"] != 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
