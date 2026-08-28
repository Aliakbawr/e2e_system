"""Compare enhanced-audio Gemma predictions with the frozen Vosk condition."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[1]
DEFAULT_BASELINE = PROJECT_DIR / (
    "benchmark/asr_llm_error_propagation/llm/"
    "gemma2_9b_vosk_path_predictions_v1.csv"
)
DEFAULT_ENHANCED = BASE_DIR / (
    "results/gemma2_9b_low_level_gain_predictions_v1.csv"
)
DEFAULT_ORACLE = PROJECT_DIR / (
    "benchmark/asr_llm_error_propagation/llm/gemma2_9b_oracle_predictions_v1.csv"
)
DEFAULT_RESULTS_DIR = BASE_DIR / "results"


def _read_index(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = [row[key] for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {key} values in {path}")
    if any(row.get("status") != "ok" for row in rows):
        raise ValueError(f"At least one failed prediction in {path}")
    return {row[key]: row for row in rows}


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _bootstrap(
    paired: list[dict],
    replicates: int = 5000,
    seed: int = 20260828,
) -> dict:
    clusters: dict[str, list[dict]] = defaultdict(list)
    for row in paired:
        clusters[row["id"]].append(row)
    ids = sorted(clusters)
    rng = random.Random(seed)
    em_deltas = []
    f1_deltas = []
    for _ in range(replicates):
        sample = [item for _id in ids for item in clusters[rng.choice(ids)]]
        em_deltas.append(
            sum(item["enhanced_em"] - item["baseline_em"] for item in sample)
            / len(sample)
        )
        f1_deltas.append(
            sum(item["enhanced_f1"] - item["baseline_f1"] for item in sample)
            / len(sample)
        )
    return {
        "unit": "semantic_id",
        "replicates": replicates,
        "seed": seed,
        "em_delta": {
            "lower": _percentile(em_deltas, 0.025),
            "upper": _percentile(em_deltas, 0.975),
            "probability_of_improvement": sum(value > 0 for value in em_deltas)
            / replicates,
        },
        "mean_f1_delta": {
            "lower": _percentile(f1_deltas, 0.025),
            "upper": _percentile(f1_deltas, 0.975),
            "probability_of_improvement": sum(value > 0 for value in f1_deltas)
            / replicates,
        },
    }


def _metrics(rows: list[dict]) -> dict:
    total = len(rows)
    baseline_em = sum(row["baseline_em"] for row in rows) / total
    enhanced_em = sum(row["enhanced_em"] for row in rows) / total
    baseline_f1 = sum(row["baseline_f1"] for row in rows) / total
    enhanced_f1 = sum(row["enhanced_f1"] for row in rows) / total
    return {
        "recordings": total,
        "baseline_em": baseline_em,
        "enhanced_em": enhanced_em,
        "em_delta": enhanced_em - baseline_em,
        "baseline_mean_f1": baseline_f1,
        "enhanced_mean_f1": enhanced_f1,
        "mean_f1_delta": enhanced_f1 - baseline_f1,
        "em_gained": sum(
            not row["baseline_em"] and row["enhanced_em"] for row in rows
        ),
        "em_lost": sum(
            row["baseline_em"] and not row["enhanced_em"] for row in rows
        ),
        "asr_contexts_changed": sum(row["asr_context_changed"] for row in rows),
        "llm_answers_changed": sum(row["llm_answer_changed"] for row in rows),
        "cluster_bootstrap_95_ci": _bootstrap(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--enhanced", type=Path, default=DEFAULT_ENHANCED)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    baseline = _read_index(args.baseline, "file_name")
    enhanced = _read_index(args.enhanced, "file_name")
    oracle = _read_index(args.oracle, "id")
    if baseline.keys() != enhanced.keys() or len(baseline) != 769:
        raise ValueError("Baseline and enhanced predictions must share 769 filenames")

    paired = []
    for filename in sorted(baseline):
        before = baseline[filename]
        after = enhanced[filename]
        if before["id"] != after["id"]:
            raise ValueError(f"Semantic ID mismatch for {filename}")
        paired.append(
            {
                "id": before["id"],
                "file_name": filename,
                "gold_answer": before["gold_answer"],
                "baseline_asr": before["asr_normalized"],
                "enhanced_asr": after["asr_normalized"],
                "baseline_answer": before["asr_llm_answer_raw"],
                "enhanced_answer": after["asr_llm_answer_raw"],
                "baseline_em": int(before["asr_em"]),
                "enhanced_em": int(after["asr_em"]),
                "baseline_f1": float(before["asr_f1"]),
                "enhanced_f1": float(after["asr_f1"]),
                "asr_context_changed": (
                    before["asr_normalized"] != after["asr_normalized"]
                ),
                "llm_answer_changed": (
                    before["asr_llm_answer_normalized"]
                    != after["asr_llm_answer_normalized"]
                ),
                "oracle_em": int(oracle[before["id"]]["oracle_em"]),
            }
        )

    primary = [row for row in paired if row["oracle_em"]]
    summary = {
        "all_769": _metrics(paired),
        "primary_oracle_em_cohort": _metrics(primary),
        "interpretation": (
            "Only the ASR context differs. Confidence intervals cluster repeated "
            "recordings by semantic ID."
        ),
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.results_dir / "downstream_comparison_v1.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    changed_path = args.results_dir / "downstream_paired_changes_v1.csv"
    fields = list(paired[0])
    with changed_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            row for row in paired if row["asr_context_changed"] or row["llm_answer_changed"]
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
