"""Tune ASR confidence and N-best score-gap parameters without test leakage."""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import contextmanager
from pathlib import Path

import src.asr.risk as risk_module
from benchmark.asr_parameter_selection.common import (
    answer_preserved,
    choose_development_configuration,
    read_csv,
    word_error_counts,
)
from config.settings import SAMPLE_RATE
from src.asr.audio import prepare_audio_file
from src.asr.text import preprocess_transcription
from src.asr.transcriber import _recognize_pcm
from src.asr.types import RecognizedWord, TranscriptAlternative, TranscriptionResult


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[1]
DEFAULT_SOURCE = PROJECT_DIR / "benchmark/asr_audio_preprocessing/results/low_level_gain_fleurs_qa_eval_input_v1.csv"
DEFAULT_NBEST = PROJECT_DIR / "benchmark/asr_nbest/results/nbest_evidence_checkpoint_v1.csv"
DEFAULT_AUDIO = PROJECT_DIR / "benchmark/asr_llm_error_propagation/data/test"
DEFAULT_RESULTS = BASE_DIR / "results"
CONFIDENCE_VALUES = (0.45, 0.55, 0.65, 0.75, 0.85)
GAP_VALUES = (0.25, 0.5, 1.0, 2.0, 3.0)
CACHE_COLUMNS = ("file_name", "text", "words_json", "status", "error")


def _append(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CACHE_COLUMNS)
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def build_confidence_cache(
    source: list[dict[str, str]], cache_path: Path, audio_dir: Path, max_new: int | None
) -> dict[str, dict[str, str]]:
    cache = {row["file_name"]: row for row in read_csv(cache_path)} if cache_path.exists() else {}
    added = 0
    for index, row in enumerate(source, 1):
        filename = row["file_name"]
        if filename in cache or (max_new is not None and added >= max_new):
            continue
        status, error, result = "ok", "", TranscriptionResult(text="")
        try:
            prepared = prepare_audio_file(audio_dir / filename, SAMPLE_RATE, "low_level_gain")
            result = _recognize_pcm(prepared.pcm16)
            if result.text != row["asr_raw"]:
                raise RuntimeError("primary decoder no longer reproduces frozen transcript")
        except Exception as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"
        saved = {
            "file_name": filename,
            "text": result.text,
            "words_json": json.dumps([word.to_dict() for word in result.words], ensure_ascii=False),
            "status": status,
            "error": error,
        }
        _append(cache_path, saved)
        cache[filename] = {key: str(value) for key, value in saved.items()}
        added += 1
        print(f"confidence_cache {index}/{len(source)} {filename}: {status}", flush=True)
    return cache


@contextmanager
def _risk_settings(confidence: float, gap: float):
    old = (risk_module.ASR_WORD_CONFIDENCE_THRESHOLD, risk_module.ASR_ALTERNATIVE_SCORE_GAP)
    risk_module.ASR_WORD_CONFIDENCE_THRESHOLD = confidence
    risk_module.ASR_ALTERNATIVE_SCORE_GAP = gap
    try:
        yield
    finally:
        risk_module.ASR_WORD_CONFIDENCE_THRESHOLD, risk_module.ASR_ALTERNATIVE_SCORE_GAP = old


def _transcription(source: dict[str, str], cached: dict[str, str], nbest: dict[str, str]) -> TranscriptionResult:
    words = tuple(RecognizedWord(**item) for item in json.loads(cached["words_json"]))
    alternatives = tuple(
        TranscriptAlternative(
            text=item["text"],
            decoder_score=item.get("decoder_score"),
        )
        for item in json.loads(nbest["alternatives_json"])
    )
    return preprocess_transcription(
        TranscriptionResult(text=source["asr_raw"], words=words, alternatives=alternatives)
    )


def evaluate_configuration(
    records: list[tuple[dict[str, str], TranscriptionResult]],
    split: str,
    confidence: float,
    gap: float,
) -> dict[str, object]:
    selected = [item for item in records if split == "all" or item[0]["split"] == split]
    baseline_edits = oracle_edits = reference_words = 0
    baseline_spans = oracle_spans = clarifications = useful = unnecessary = opportunities = 0
    with _risk_settings(confidence, gap):
        for source, transcription in selected:
            assessment = risk_module.assess_asr_risk(transcription)
            candidates = [transcription.text]
            scores = [item.decoder_score for item in transcription.alternatives if item.decoder_score is not None]
            best_score = max(scores) if scores else None
            for alternative in transcription.alternatives:
                if alternative.text == transcription.text:
                    continue
                if best_score is not None and alternative.decoder_score is not None and best_score - alternative.decoder_score > gap:
                    continue
                candidates.append(alternative.text)
            errors = [word_error_counts(source["reference_normalized"], text)[0] for text in candidates]
            ref_count = word_error_counts(source["reference_normalized"], transcription.text)[1]
            spans = [answer_preserved(source["gold_answer_normalized"], text) for text in candidates]
            beneficial = min(errors) < errors[0] or (not spans[0] and any(spans[1:]))
            opportunities += beneficial
            clarifications += assessment.requires_clarification
            useful += assessment.requires_clarification and beneficial
            unnecessary += assessment.requires_clarification and not beneficial
            chosen = min(range(len(candidates)), key=lambda i: (-int(spans[i]), errors[i])) if assessment.requires_clarification else 0
            baseline_edits += errors[0]
            oracle_edits += errors[chosen]
            reference_words += ref_count
            baseline_spans += spans[0]
            oracle_spans += spans[chosen]
    total = len(selected)
    return {
        "split": split,
        "word_confidence_threshold": confidence,
        "alternative_score_gap": gap,
        "recordings": total,
        "baseline_corpus_wer": baseline_edits / reference_words,
        "oracle_corpus_wer": oracle_edits / reference_words,
        "baseline_answer_span_rate": baseline_spans / total,
        "oracle_answer_span_rate": oracle_spans / total,
        "clarification_rate": clarifications / total,
        "useful_clarification_rate": useful / total,
        "unnecessary_clarification_rate": unnecessary / total,
        "opportunity_recall": useful / opportunities if opportunities else 0.0,
        "clarification_precision": useful / clarifications if clarifications else 0.0,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, object]], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    development = [row for row in rows if row["split"] == "development"]
    confidence = sorted({float(row["word_confidence_threshold"]) for row in development})
    gaps = sorted({float(row["alternative_score_gap"]) for row in development})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "oracle_corpus_wer", "Oracle-assisted corpus WER"),
        (axes[1], "clarification_rate", "Clarification rate"),
    ):
        matrix = np.array([[next(float(row[metric]) for row in development if float(row["word_confidence_threshold"]) == c and float(row["alternative_score_gap"]) == g) for g in gaps] for c in confidence])
        image = axis.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
        axis.set_xticks(range(len(gaps)), gaps)
        axis.set_yticks(range(len(confidence)), confidence)
        axis.set_xlabel("Alternative decoder-score gap")
        axis.set_ylabel("Word-confidence threshold")
        axis.set_title(title)
        figure.colorbar(image, ax=axis)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--nbest", type=Path, default=DEFAULT_NBEST)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--max-new", type=int)
    parser.add_argument("--max-clarification-rate", type=float, default=0.10)
    parser.add_argument("--max-unnecessary-rate", type=float, default=0.05)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    source = read_csv(args.source)
    split = {row["id"]: row["split"] for row in read_csv(DEFAULT_NBEST.parent / "semantic_split_v1.csv")}
    for row in source:
        row["split"] = split[row["id"]]
    nbest = {row["file_name"]: row for row in read_csv(args.nbest)}
    cache_path = args.results_dir / "primary_confidence_cache_v1.csv"
    cache = build_confidence_cache(source, cache_path, args.audio_dir, args.max_new)
    if len(cache) != len(source):
        print(f"Confidence cache incomplete: {len(cache)}/{len(source)}; rerun to resume")
        return 0
    failures = [row for row in cache.values() if row["status"] != "ok"]
    valid_source = [row for row in source if cache[row["file_name"]]["status"] == "ok"]
    records = [
        (row, _transcription(row, cache[row["file_name"]], nbest[row["file_name"]]))
        for row in valid_source
    ]
    rows = [evaluate_configuration(records, split_name, confidence, gap) for split_name in ("development", "held_out", "all") for confidence in CONFIDENCE_VALUES for gap in GAP_VALUES]
    selected = choose_development_configuration(rows, max_clarification_rate=args.max_clarification_rate, max_unnecessary_rate=args.max_unnecessary_rate)
    selected_held_out = next(row for row in rows if row["split"] == "held_out" and row["word_confidence_threshold"] == selected["word_confidence_threshold"] and row["alternative_score_gap"] == selected["alternative_score_gap"])
    _write_csv(args.results_dir / "uncertainty_grid_v1.csv", rows)
    summary = {
        "cohort": {
            "requested_recordings": len(source),
            "aligned_recordings": len(valid_source),
            "excluded_decoder_drift": len(failures),
            "excluded_files": [
                {"file_name": row["file_name"], "reason": row["error"]}
                for row in failures
            ],
        },
        "selection_rule": {"development_only": True, "max_clarification_rate": args.max_clarification_rate, "max_unnecessary_clarification_rate": args.max_unnecessary_rate, "ranking": ["maximize oracle answer-span rate", "minimize oracle corpus WER", "minimize unnecessary clarifications", "minimize clarification rate"]},
        "selected_on_development": selected,
        "frozen_held_out_result": selected_held_out,
        "oracle_assumption": "When clarification fires, the user is assumed to choose the eligible transcript that preserves the answer span and then minimizes WER.",
    }
    (args.results_dir / "uncertainty_selection_v1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot(rows, args.results_dir / "uncertainty_heatmaps_v1.png")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
