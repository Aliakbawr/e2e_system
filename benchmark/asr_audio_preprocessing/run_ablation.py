"""Evaluate low-level audio repair against frozen Vosk transcripts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from benchmark.asr_llm_error_propagation.asr.run_fleurs_fa_ir_asr import (
    calculate_wer_cer,
    edit_distance,
    normalize_persian_for_evaluation,
)
from benchmark.asr_llm_error_propagation.prepare_downstream_eval import (
    normalize_persian,
)
from config.settings import SAMPLE_RATE
from src.asr.audio import (
    LOW_LEVEL_THRESHOLD_DBFS,
    MAX_GAIN_DB,
    PEAK_HEADROOM_DBFS,
    TARGET_RMS_DBFS,
    prepare_audio_file,
)
from src.asr.transcriber import _recognize_pcm


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[1]
DEFAULT_INPUT = PROJECT_DIR / (
    "benchmark/asr_llm_error_propagation/analysis/inputs/"
    "vosk_fleurs_qa_eval_input_v1.csv"
)
DEFAULT_AUDIO_DIR = PROJECT_DIR / "benchmark/asr_llm_error_propagation/data/test"
DEFAULT_RESULTS_DIR = BASE_DIR / "results"
PROFILE = "low_level_gain"
CHECKPOINT_COLUMNS = (
    "file_name",
    "prediction_raw",
    "latency_sec",
    "preprocessing_applied",
    "input_rms_dbfs",
    "output_rms_dbfs",
    "gain_db",
    "status",
    "error",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    return fields, rows


def _load_checkpoint(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    fields, rows = _load_csv(path)
    if tuple(fields) != CHECKPOINT_COLUMNS:
        raise ValueError(f"Checkpoint schema mismatch: {path}")
    filenames = [row["file_name"] for row in rows]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Checkpoint contains duplicate filenames")
    return {row["file_name"]: row for row in rows}


def _append_checkpoint(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHECKPOINT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def _corpus_error_rate(rows: list[dict[str, str]], hypothesis_key: str) -> float:
    edits = 0
    reference_tokens = 0
    for row in rows:
        reference = normalize_persian_for_evaluation(row["reference_normalized"])
        hypothesis = normalize_persian_for_evaluation(row[hypothesis_key])
        reference_list = reference.split()
        edits += edit_distance(reference_list, hypothesis.split())
        reference_tokens += len(reference_list)
    return edits / reference_tokens


def _validate_baseline(
    rows: list[dict[str, str]],
    audio_dir: Path,
    sample_count: int,
) -> None:
    checked = 0
    for row in rows:
        prepared = prepare_audio_file(
            audio_dir / row["file_name"], SAMPLE_RATE, PROFILE
        )
        if not prepared.applied:
            continue
        baseline = prepare_audio_file(
            audio_dir / row["file_name"], SAMPLE_RATE, "none"
        )
        current = _recognize_pcm(baseline.pcm16).text
        if current != row["asr_raw"]:
            raise RuntimeError(
                "Current decoder does not reproduce the frozen Vosk baseline for "
                f"{row['file_name']}; a paired reuse comparison is invalid"
            )
        checked += 1
        print(f"baseline_validation {checked}/{sample_count} {row['file_name']}: exact")
        if checked >= sample_count:
            return
    if checked < sample_count:
        raise ValueError(f"Only {checked} gain-eligible recordings were available")


def _build_output(
    original_fields: list[str],
    frozen_rows: list[dict[str, str]],
    checkpoint: dict[str, dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    extra_fields = [
        "preprocessing_profile",
        "preprocessing_applied",
        "input_rms_dbfs",
        "output_rms_dbfs",
        "gain_db",
    ]
    fields = original_fields + [field for field in extra_fields if field not in original_fields]
    output_rows = []
    for frozen in frozen_rows:
        result = checkpoint[frozen["file_name"]]
        if result["status"] not in {"ok", "reused_frozen_baseline"}:
            raise RuntimeError(f"Failed checkpoint row: {frozen['file_name']}")
        row = dict(frozen)
        row["asr_raw"] = result["prediction_raw"]
        row["asr_normalized"] = normalize_persian_for_evaluation(row["asr_raw"])
        row["asr_eval"] = normalize_persian(row["asr_raw"])
        row["wer"], row["cer"] = calculate_wer_cer(
            normalize_persian_for_evaluation(row["reference_normalized"]),
            row["asr_normalized"],
        )
        gold = normalize_persian(row["gold_answer_normalized"])
        row["answer_preserved"] = bool(gold) and gold in row["asr_eval"]
        row["latency_sec"] = result["latency_sec"]
        duration = float(row["duration_sec"])
        row["rtf"] = float(result["latency_sec"]) / duration if duration else ""
        row["preprocessing_profile"] = PROFILE
        row["preprocessing_applied"] = result["preprocessing_applied"]
        row["input_rms_dbfs"] = result["input_rms_dbfs"]
        row["output_rms_dbfs"] = result["output_rms_dbfs"]
        row["gain_db"] = result["gain_db"]
        output_rows.append(row)
    return fields, output_rows


def _summary(
    frozen_rows: list[dict[str, str]], enhanced_rows: list[dict[str, str]]
) -> dict:
    improved = regressed = unchanged = 0
    spans_gained = spans_lost = 0
    transcripts_changed = 0
    for before, after in zip(frozen_rows, enhanced_rows):
        before_wer = float(before["wer"])
        after_wer = float(after["wer"])
        improved += after_wer < before_wer - 1e-12
        regressed += after_wer > before_wer + 1e-12
        unchanged += abs(after_wer - before_wer) <= 1e-12
        before_span = _bool(before["answer_preserved"])
        after_span = _bool(after["answer_preserved"])
        spans_gained += not before_span and after_span
        spans_lost += before_span and not after_span
        transcripts_changed += before["asr_raw"] != after["asr_raw"]

    total = len(frozen_rows)
    applied = sum(_bool(row["preprocessing_applied"]) for row in enhanced_rows)
    before_preserved = sum(_bool(row["answer_preserved"]) for row in frozen_rows)
    after_preserved = sum(_bool(row["answer_preserved"]) for row in enhanced_rows)
    return {
        "recordings": total,
        "preprocessing_applied": applied,
        "transcripts_changed": transcripts_changed,
        "row_wer_improved": improved,
        "row_wer_regressed": regressed,
        "row_wer_unchanged": unchanged,
        "mean_recording_wer_before": sum(float(row["wer"]) for row in frozen_rows) / total,
        "mean_recording_wer_after": sum(float(row["wer"]) for row in enhanced_rows) / total,
        "corpus_wer_before": _corpus_error_rate(frozen_rows, "asr_raw"),
        "corpus_wer_after": _corpus_error_rate(enhanced_rows, "asr_raw"),
        "answer_preservation_rate_before": before_preserved / total,
        "answer_preservation_rate_after": after_preserved / total,
        "answer_spans_gained": spans_gained,
        "answer_spans_lost": spans_lost,
        "cluster_bootstrap_95_ci": _cluster_bootstrap(frozen_rows, enhanced_rows),
    }


def _write_paired_changes(
    path: Path,
    frozen_rows: list[dict[str, str]],
    enhanced_rows: list[dict[str, str]],
) -> None:
    columns = (
        "id",
        "file_name",
        "input_rms_dbfs",
        "gain_db",
        "reference_normalized",
        "baseline_asr",
        "enhanced_asr",
        "wer_before",
        "wer_after",
        "wer_outcome",
        "answer_preserved_before",
        "answer_preserved_after",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for before, after in zip(frozen_rows, enhanced_rows):
            if before["asr_raw"] == after["asr_raw"]:
                continue
            before_wer = float(before["wer"])
            after_wer = float(after["wer"])
            outcome = "unchanged"
            if after_wer < before_wer - 1e-12:
                outcome = "improved"
            elif after_wer > before_wer + 1e-12:
                outcome = "regressed"
            writer.writerow(
                {
                    "id": before["id"],
                    "file_name": before["file_name"],
                    "input_rms_dbfs": after["input_rms_dbfs"],
                    "gain_db": after["gain_db"],
                    "reference_normalized": before["reference_normalized"],
                    "baseline_asr": before["asr_raw"],
                    "enhanced_asr": after["asr_raw"],
                    "wer_before": before["wer"],
                    "wer_after": after["wer"],
                    "wer_outcome": outcome,
                    "answer_preserved_before": before["answer_preserved"],
                    "answer_preserved_after": after["answer_preserved"],
                }
            )


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _cluster_bootstrap(
    frozen_rows: list[dict[str, str]],
    enhanced_rows: list[dict[str, str]],
    replicates: int = 5000,
    seed: int = 20260828,
) -> dict:
    """Bootstrap semantic IDs so repeated recordings stay in one cluster."""
    clusters: dict[str, list[tuple[int, int, int, int, int]]] = defaultdict(list)
    for before, after in zip(frozen_rows, enhanced_rows):
        reference = normalize_persian_for_evaluation(
            before["reference_normalized"]
        ).split()
        before_tokens = normalize_persian_for_evaluation(before["asr_raw"]).split()
        after_tokens = normalize_persian_for_evaluation(after["asr_raw"]).split()
        clusters[before["id"]].append(
            (
                len(reference),
                edit_distance(reference, before_tokens),
                edit_distance(reference, after_tokens),
                int(_bool(before["answer_preserved"])),
                int(_bool(after["answer_preserved"])),
            )
        )

    cluster_ids = sorted(clusters)
    rng = random.Random(seed)
    wer_deltas = []
    answer_deltas = []
    for _ in range(replicates):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        observations = [item for cluster_id in sampled for item in clusters[cluster_id]]
        reference_tokens = sum(item[0] for item in observations)
        before_edits = sum(item[1] for item in observations)
        after_edits = sum(item[2] for item in observations)
        wer_deltas.append((after_edits - before_edits) / reference_tokens)
        answer_deltas.append(
            (sum(item[4] for item in observations) - sum(item[3] for item in observations))
            / len(observations)
        )

    return {
        "unit": "semantic_id",
        "replicates": replicates,
        "seed": seed,
        "corpus_wer_delta_after_minus_before": {
            "lower": _percentile(wer_deltas, 0.025),
            "upper": _percentile(wer_deltas, 0.975),
            "probability_of_improvement": sum(value < 0 for value in wer_deltas)
            / replicates,
        },
        "answer_preservation_delta_after_minus_before": {
            "lower": _percentile(answer_deltas, 0.025),
            "upper": _percentile(answer_deltas, 0.975),
            "probability_of_improvement": sum(value > 0 for value in answer_deltas)
            / replicates,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--baseline-validation", type=int, default=3)
    parser.add_argument(
        "--max-new",
        type=int,
        help="Decode at most this many new rows, leaving a resumable checkpoint.",
    )
    args = parser.parse_args()

    input_hash = _sha256(args.input)
    original_fields, rows = _load_csv(args.input)
    if len(rows) != 769 or len({row["file_name"] for row in rows}) != 769:
        raise ValueError("Expected the frozen 769-recording Vosk propagation table")
    missing_audio = [
        row["file_name"]
        for row in rows
        if not (args.audio_dir / row["file_name"]).is_file()
    ]
    if missing_audio:
        raise FileNotFoundError(f"Missing {len(missing_audio)} frozen audio files")

    if args.baseline_validation:
        _validate_baseline(rows, args.audio_dir, args.baseline_validation)

    checkpoint_path = args.results_dir / f"{PROFILE}_checkpoint_v1.csv"
    checkpoint = _load_checkpoint(checkpoint_path)
    new_rows = 0
    for index, row in enumerate(rows, start=1):
        filename = row["file_name"]
        if filename in checkpoint:
            continue
        if args.max_new is not None and new_rows >= args.max_new:
            break

        prepared = prepare_audio_file(args.audio_dir / filename, SAMPLE_RATE, PROFILE)
        started = time.perf_counter()
        status = "ok"
        error = ""
        prediction = row["asr_raw"]
        latency = float(row["latency_sec"])
        try:
            if prepared.applied:
                prediction = _recognize_pcm(prepared.pcm16).text
                latency = time.perf_counter() - started
            else:
                status = "reused_frozen_baseline"
        except Exception as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"

        checkpoint_row = {
            "file_name": filename,
            "prediction_raw": prediction,
            "latency_sec": f"{latency:.9f}",
            "preprocessing_applied": prepared.applied,
            "input_rms_dbfs": f"{prepared.input_rms_dbfs:.6f}",
            "output_rms_dbfs": f"{prepared.output_rms_dbfs:.6f}",
            "gain_db": f"{prepared.gain_db:.6f}",
            "status": status,
            "error": error,
        }
        _append_checkpoint(checkpoint_path, checkpoint_row)
        checkpoint[filename] = {key: str(value) for key, value in checkpoint_row.items()}
        new_rows += 1
        if prepared.applied or index % 100 == 0:
            print(
                f"[{index:03d}/769] {filename} applied={prepared.applied} "
                f"gain_db={prepared.gain_db:.2f} status={status}",
                flush=True,
            )

    if len(checkpoint) != len(rows):
        print(f"Checkpoint incomplete: {len(checkpoint)}/769 rows; rerun to resume")
        return 0
    if any(row["status"] == "error" for row in checkpoint.values()):
        raise RuntimeError("Checkpoint contains failed rows")

    fields, enhanced = _build_output(original_fields, rows, checkpoint)
    output_path = args.results_dir / f"{PROFILE}_fleurs_qa_eval_input_v1.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(enhanced)

    summary = _summary(rows, enhanced)
    changes_path = args.results_dir / f"{PROFILE}_paired_changes_v1.csv"
    _write_paired_changes(changes_path, rows, enhanced)
    metadata = {
        "stage": "asr_audio_preprocessing_low_level_gain_v1",
        "profile": PROFILE,
        "configuration": {
            "threshold_dbfs": LOW_LEVEL_THRESHOLD_DBFS,
            "target_rms_dbfs": TARGET_RMS_DBFS,
            "max_gain_db": MAX_GAIN_DB,
            "peak_headroom_dbfs": PEAK_HEADROOM_DBFS,
        },
        "input_path": str(args.input.resolve()),
        "input_sha256": input_hash,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "output_path": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "paired_changes_path": str(changes_path.resolve()),
        "paired_changes_sha256": _sha256(changes_path),
        "summary": summary,
    }
    metadata_path = args.results_dir / f"{PROFILE}_summary_v1.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if _sha256(args.input) != input_hash:
        raise RuntimeError("Frozen input changed during the ablation")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
