"""Tune low-level audio gain on development data, then evaluate held-out once."""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark.asr_parameter_selection.common import answer_preserved, read_csv, word_error_counts
from config.settings import SAMPLE_RATE
from src.asr.audio import prepare_audio_file
from src.asr.transcriber import _load_model, _recognize_pcm


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[1]
DEFAULT_SOURCE = PROJECT_DIR / "benchmark/asr_llm_error_propagation/analysis/inputs/vosk_fleurs_qa_eval_input_v1.csv"
DEFAULT_SPLIT = PROJECT_DIR / "benchmark/asr_nbest/results/semantic_split_v1.csv"
DEFAULT_AUDIO = PROJECT_DIR / "benchmark/asr_llm_error_propagation/data/test"
DEFAULT_RESULTS = BASE_DIR / "results"
THRESHOLDS = (-55.0, -50.0, -45.0, -40.0, -35.0)
TARGETS = (-30.0, -27.0, -24.0, -21.0, -18.0)
COLUMNS = ("configuration", "file_name", "prediction", "applied", "gain_db", "latency_sec", "status", "error")


def configuration_id(threshold: float, target: float, max_gain: float, headroom: float) -> str:
    return f"threshold_{threshold:g}__target_{target:g}__maxgain_{max_gain:g}__headroom_{headroom:g}"


def _load_checkpoint(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    rows = read_csv(path)
    return {(row["configuration"], row["file_name"]): row for row in rows}


def _append(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def decode_configuration(
    rows: list[dict[str, str]], audio_dir: Path, checkpoint_path: Path,
    threshold: float, target: float, max_gain: float, headroom: float,
    max_new: int | None, workers: int,
) -> tuple[dict[tuple[str, str], dict[str, str]], bool]:
    checkpoint = _load_checkpoint(checkpoint_path)
    config = configuration_id(threshold, target, max_gain, headroom)
    pending = []
    for index, source in enumerate(rows, 1):
        key = (config, source["file_name"])
        if key in checkpoint:
            continue
        if max_new is not None and len(pending) >= max_new:
            break
        pending.append((index, source))

    def process(item: tuple[int, dict[str, str]]) -> tuple[int, dict[str, object]]:
        index, source = item
        status, error, prediction, latency = "ok", "", source["asr_raw"], 0.0
        prepared = prepare_audio_file(
            audio_dir / source["file_name"], SAMPLE_RATE, "low_level_gain",
            threshold_dbfs=threshold, target_rms_dbfs=target,
            max_gain_db=max_gain, peak_headroom_dbfs=headroom,
        )
        try:
            if prepared.applied:
                # Threshold only controls whether gain is applied. For fixed
                # target/safety settings, an equal gain creates identical PCM,
                # so reuse an earlier threshold's exact decoder result.
                reusable = next(
                    (
                        row
                        for row in checkpoint.values()
                        if row["file_name"] == source["file_name"]
                        and row["status"] == "ok"
                        and row["applied"].lower() == "true"
                        and abs(float(row["gain_db"]) - prepared.gain_db) < 1e-9
                    ),
                    None,
                )
                if reusable is not None:
                    prediction = reusable["prediction"]
                else:
                    started = time.perf_counter()
                    prediction = _recognize_pcm(prepared.pcm16).text
                    latency = time.perf_counter() - started
        except Exception as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"
        saved = {"configuration": config, "file_name": source["file_name"], "prediction": prediction, "applied": prepared.applied, "gain_db": prepared.gain_db, "latency_sec": latency, "status": status, "error": error}
        return index, saved

    if pending:
        _load_model()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        completed = executor.map(process, pending)
        for index, saved in completed:
            source = rows[index - 1]
            key = (config, source["file_name"])
            _append(checkpoint_path, saved)
            checkpoint[key] = {name: str(value) for name, value in saved.items()}
            print(f"audio_grid {config} {index}/{len(rows)}: {saved['status']}", flush=True)
    complete = all((config, row["file_name"]) in checkpoint for row in rows)
    return checkpoint, complete


def summarize(rows: list[dict[str, str]], checkpoint: dict[tuple[str, str], dict[str, str]], config: str, split: str, threshold: float, target: float) -> dict[str, object]:
    edits = reference_words = spans = applied = 0
    for source in rows:
        result = checkpoint[(config, source["file_name"])]
        if result["status"] != "ok":
            raise RuntimeError(f"Failed audio result: {config}/{source['file_name']}")
        error_count, count = word_error_counts(source["reference_normalized"], result["prediction"])
        edits += error_count
        reference_words += count
        spans += answer_preserved(source["gold_answer_normalized"], result["prediction"])
        applied += result["applied"].lower() == "true"
    return {"split": split, "low_level_threshold_dbfs": threshold, "target_rms_dbfs": target, "recordings": len(rows), "preprocessing_applied": applied, "corpus_wer": edits / reference_words, "answer_span_rate": spans / len(rows)}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, object]], selected: dict[str, object], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    thresholds = sorted({float(row["low_level_threshold_dbfs"]) for row in rows})
    targets = sorted({float(row["target_rms_dbfs"]) for row in rows})
    matrix = np.array([[next(float(row["corpus_wer"]) for row in rows if float(row["low_level_threshold_dbfs"]) == threshold and float(row["target_rms_dbfs"]) == target) for target in targets] for threshold in thresholds])
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="viridis_r")
    axis.set_xticks(range(len(targets)), targets)
    axis.set_yticks(range(len(thresholds)), thresholds)
    axis.set_xlabel("Target RMS (dBFS)")
    axis.set_ylabel("Low-level threshold (dBFS)")
    axis.set_title("Development corpus WER")
    axis.scatter(targets.index(float(selected["target_rms_dbfs"])), thresholds.index(float(selected["low_level_threshold_dbfs"])), marker="*", s=220, color="red", edgecolor="white")
    figure.colorbar(image, ax=axis)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--max-gain-db", type=float, default=40.0)
    parser.add_argument("--peak-headroom-dbfs", type=float, default=-1.0)
    parser.add_argument("--max-new", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    split_map = {row["id"]: row["split"] for row in read_csv(args.split)}
    source = read_csv(args.source)
    development = [row for row in source if split_map[row["id"]] == "development"]
    held_out = [row for row in source if split_map[row["id"]] == "held_out"]
    checkpoint_path = args.results_dir / "audio_grid_checkpoint_v1.csv"
    summaries = []
    checkpoint = _load_checkpoint(checkpoint_path)
    for threshold in THRESHOLDS:
        for target in TARGETS:
            checkpoint, complete = decode_configuration(development, args.audio_dir, checkpoint_path, threshold, target, args.max_gain_db, args.peak_headroom_dbfs, args.max_new, args.workers)
            if not complete:
                print("Development grid incomplete; rerun to resume")
                return 0
            config = configuration_id(threshold, target, args.max_gain_db, args.peak_headroom_dbfs)
            summaries.append(summarize(development, checkpoint, config, "development", threshold, target))
    selected = min(summaries, key=lambda row: (float(row["corpus_wer"]), -float(row["answer_span_rate"]), float(row["preprocessing_applied"])))
    threshold, target = float(selected["low_level_threshold_dbfs"]), float(selected["target_rms_dbfs"])
    checkpoint, complete = decode_configuration(held_out, args.audio_dir, checkpoint_path, threshold, target, args.max_gain_db, args.peak_headroom_dbfs, args.max_new, args.workers)
    if not complete:
        print("Held-out evaluation incomplete; rerun to resume")
        return 0
    config = configuration_id(threshold, target, args.max_gain_db, args.peak_headroom_dbfs)
    held_summary = summarize(held_out, checkpoint, config, "held_out", threshold, target)
    _write_csv(args.results_dir / "audio_development_grid_v1.csv", summaries)
    report = {"selection_protocol": "minimum development corpus WER; answer-span rate and fewer modified recordings break ties", "safety_constraints": {"max_gain_db": args.max_gain_db, "peak_headroom_dbfs": args.peak_headroom_dbfs}, "selected_on_development": selected, "frozen_held_out_result": held_summary}
    (args.results_dir / "audio_selection_v1.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _plot(summaries, selected, args.results_dir / "audio_wer_heatmap_v1.png")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
