"""Freeze Vosk N-best hypotheses for the enhanced 769-recording cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

from benchmark.asr_llm_error_propagation.asr.run_fleurs_fa_ir_asr import (
    calculate_wer_cer,
    normalize_persian_for_evaluation,
)
from benchmark.asr_llm_error_propagation.prepare_downstream_eval import (
    normalize_persian,
)
from config.settings import ASR_MAX_ALTERNATIVES, SAMPLE_RATE
from src.asr.audio import prepare_audio_file
from src.asr.transcriber import _recognize_pcm


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[1]
DEFAULT_INPUT = PROJECT_DIR / (
    "benchmark/asr_audio_preprocessing/results/"
    "low_level_gain_fleurs_qa_eval_input_v1.csv"
)
DEFAULT_AUDIO_DIR = PROJECT_DIR / "benchmark/asr_llm_error_propagation/data/test"
DEFAULT_RESULTS_DIR = BASE_DIR / "results"
PROFILE = "low_level_gain"
SPLIT_SALT = "persian-assistant-nbest-v1"
EVIDENCE_COLUMNS = (
    "id",
    "file_name",
    "split",
    "primary_text",
    "alternatives_json",
    "latency_sec",
    "status",
    "error",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _assigned_split(semantic_id: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}:{semantic_id}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "development" if fraction < 0.70 else "held_out"


def _write_or_validate_split(path: Path, semantic_ids: set[str]) -> dict[str, str]:
    expected = {_id: _assigned_split(_id) for _id in sorted(semantic_ids)}
    if path.exists():
        existing = {row["id"]: row["split"] for row in _read_csv(path)}
        if existing != expected:
            raise ValueError("Existing semantic split does not match the frozen rule")
        return existing
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("id", "split"))
        writer.writeheader()
        writer.writerows(
            {"id": _id, "split": expected[_id]} for _id in sorted(expected)
        )
    return expected


def _load_checkpoint(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    rows = _read_csv(path)
    if rows and tuple(rows[0]) != EVIDENCE_COLUMNS:
        raise ValueError("N-best checkpoint schema mismatch")
    if len(rows) != len({row["file_name"] for row in rows}):
        raise ValueError("N-best checkpoint contains duplicate filenames")
    return {row["file_name"]: row for row in rows}


def _append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVIDENCE_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def _validate_primary(
    rows: list[dict[str, str]], audio_dir: Path, count: int
) -> None:
    if count <= 0:
        return
    # Cover both repaired and untouched audio deterministically.
    selected = []
    for wanted_applied in (True, False):
        for row in rows:
            prepared = prepare_audio_file(audio_dir / row["file_name"], SAMPLE_RATE, PROFILE)
            if prepared.applied == wanted_applied:
                selected.append((row, prepared))
                if sum(item[1].applied == wanted_applied for item in selected) >= count:
                    break
    for index, (row, prepared) in enumerate(selected, start=1):
        current = _recognize_pcm(prepared.pcm16).text
        if normalize_persian_for_evaluation(current) != row["asr_normalized"]:
            raise RuntimeError(
                f"Current primary decoder does not reproduce {row['file_name']}"
            )
        print(f"primary_validation {index}/{len(selected)} {row['file_name']}: exact")


def _candidates(primary: str, alternatives: list[dict]) -> list[str]:
    values = [primary, *(item["text"] for item in alternatives)]
    unique = []
    seen = set()
    for value in values:
        normalized = normalize_persian_for_evaluation(value)
        if normalized and normalized not in seen:
            unique.append(value)
            seen.add(normalized)
    return unique


def _summarize(
    source: dict[str, dict[str, str]], evidence: dict[str, dict[str, str]]
) -> dict:
    summaries = {}
    for split in ("development", "held_out", "all"):
        selected = [
            row
            for row in evidence.values()
            if split == "all" or row["split"] == split
        ]
        primary_wer = oracle_wer = top_nbest_wer = 0.0
        oracle_better = top_better = top_worse = 0
        primary_spans = oracle_spans = 0
        for row in selected:
            frozen = source[row["file_name"]]
            reference = normalize_persian_for_evaluation(
                frozen["reference_normalized"]
            )
            gold = normalize_persian(frozen["gold_answer_normalized"])
            alternatives = json.loads(row["alternatives_json"])
            candidates = _candidates(row["primary_text"], alternatives)
            wers = [
                calculate_wer_cer(
                    reference, normalize_persian_for_evaluation(candidate)
                )[0]
                for candidate in candidates
            ]
            primary_value = wers[0]
            oracle_value = min(wers)
            top_value = calculate_wer_cer(
                reference,
                normalize_persian_for_evaluation(alternatives[0]["text"]),
            )[0] if alternatives else primary_value
            primary_wer += primary_value
            oracle_wer += oracle_value
            top_nbest_wer += top_value
            oracle_better += oracle_value < primary_value - 1e-12
            top_better += top_value < primary_value - 1e-12
            top_worse += top_value > primary_value + 1e-12
            primary_has = bool(gold) and gold in normalize_persian(row["primary_text"])
            oracle_has = any(gold in normalize_persian(item) for item in candidates)
            primary_spans += primary_has
            oracle_spans += oracle_has
        total = len(selected)
        summaries[split] = {
            "recordings": total,
            "mean_recording_wer_primary": primary_wer / total,
            "mean_recording_wer_oracle_nbest": oracle_wer / total,
            "oracle_nbest_better_recordings": oracle_better,
            "mean_recording_wer_top_decoder": top_nbest_wer / total,
            "top_decoder_better_recordings": top_better,
            "top_decoder_worse_recordings": top_worse,
            "answer_span_rate_primary": primary_spans / total,
            "answer_span_rate_oracle_nbest": oracle_spans / total,
            "answer_spans_recoverable_from_nbest": oracle_spans - primary_spans,
        }
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--primary-validation", type=int, default=2)
    parser.add_argument("--max-new", type=int)
    args = parser.parse_args()

    input_hash = _sha256(args.input)
    rows = _read_csv(args.input)
    if len(rows) != 769 or len({row["id"] for row in rows}) != 286:
        raise ValueError("Expected 769 recordings and 286 semantic IDs")
    source = {row["file_name"]: row for row in rows}
    split_path = args.results_dir / "semantic_split_v1.csv"
    split_map = _write_or_validate_split(split_path, {row["id"] for row in rows})
    if args.primary_validation:
        _validate_primary(rows, args.audio_dir, args.primary_validation)

    checkpoint_path = args.results_dir / "nbest_evidence_checkpoint_v1.csv"
    evidence = _load_checkpoint(checkpoint_path)
    new_count = 0
    for index, source_row in enumerate(rows, start=1):
        filename = source_row["file_name"]
        if filename in evidence:
            continue
        if args.max_new is not None and new_count >= args.max_new:
            break
        prepared = prepare_audio_file(args.audio_dir / filename, SAMPLE_RATE, PROFILE)
        started = time.perf_counter()
        status = "ok"
        error = ""
        alternatives = []
        try:
            result = _recognize_pcm(
                prepared.pcm16, max_alternatives=ASR_MAX_ALTERNATIVES
            )
            alternatives = [item.to_dict(include_words=True) for item in result.alternatives]
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        row = {
            "id": source_row["id"],
            "file_name": filename,
            "split": split_map[source_row["id"]],
            "primary_text": source_row["asr_raw"],
            "alternatives_json": json.dumps(alternatives, ensure_ascii=False),
            "latency_sec": f"{time.perf_counter() - started:.9f}",
            "status": status,
            "error": error,
        }
        _append(checkpoint_path, row)
        evidence[filename] = {key: str(value) for key, value in row.items()}
        new_count += 1
        print(f"[{index:03d}/769] {filename}: {status}", flush=True)

    if len(evidence) != len(rows):
        print(f"Checkpoint incomplete: {len(evidence)}/769; rerun to resume")
        return 0
    if any(row["status"] != "ok" for row in evidence.values()):
        raise RuntimeError("N-best checkpoint contains failed rows")

    summary = {
        "stage": "vosk_nbest_evidence_v1",
        "configuration": {
            "max_alternatives": ASR_MAX_ALTERNATIVES,
            "audio_profile": PROFILE,
            "split_salt": SPLIT_SALT,
            "development_fraction": 0.70,
        },
        "input_sha256": input_hash,
        "split_sha256": _sha256(split_path),
        "evidence_sha256": _sha256(checkpoint_path),
        "summary": _summarize(source, evidence),
    }
    summary_path = args.results_dir / "nbest_evidence_summary_v1.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if _sha256(args.input) != input_hash:
        raise RuntimeError("Frozen enhanced-ASR input changed during decoding")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
