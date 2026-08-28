"""Analyze paired Gemma effects of explicit-correction N-best recovery."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from benchmark.asr_audio_preprocessing.analyze_downstream import _bootstrap


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[1]
RESULTS_DIR = BASE_DIR / "results"
DEFAULT_BASELINE = PROJECT_DIR / (
    "benchmark/asr_audio_preprocessing/results/"
    "gemma2_9b_low_level_gain_predictions_v1.csv"
)
DEFAULT_ENHANCED = RESULTS_DIR / "gemma2_9b_contextual_recovery_predictions_v1.csv"
DEFAULT_EVIDENCE = RESULTS_DIR / "nbest_evidence_checkpoint_v1.csv"


def _index(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len({row["file_name"] for row in rows}):
        raise ValueError(f"Duplicate filenames in {path}")
    return {row["file_name"]: row for row in rows}


def _metrics(rows: list[dict]) -> dict:
    total = len(rows)
    before_em = sum(row["baseline_em"] for row in rows) / total
    after_em = sum(row["enhanced_em"] for row in rows) / total
    before_f1 = sum(row["baseline_f1"] for row in rows) / total
    after_f1 = sum(row["enhanced_f1"] for row in rows) / total
    return {
        "recordings": total,
        "baseline_em": before_em,
        "enhanced_em": after_em,
        "em_delta": after_em - before_em,
        "baseline_mean_f1": before_f1,
        "enhanced_mean_f1": after_f1,
        "mean_f1_delta": after_f1 - before_f1,
        "em_gained": sum(
            not row["baseline_em"] and row["enhanced_em"] for row in rows
        ),
        "em_lost": sum(
            row["baseline_em"] and not row["enhanced_em"] for row in rows
        ),
        "llm_answers_changed": sum(row["llm_answer_changed"] for row in rows),
        "cluster_bootstrap_95_ci": _bootstrap(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--enhanced", type=Path, default=DEFAULT_ENHANCED)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args()

    baseline = _index(args.baseline)
    enhanced = _index(args.enhanced)
    evidence = _index(args.evidence)
    if baseline.keys() != enhanced.keys() or baseline.keys() != evidence.keys():
        raise ValueError("Baseline, enhanced, and evidence filename sets differ")
    if any(row["status"] != "ok" for row in enhanced.values()):
        raise ValueError("Enhanced predictions contain a failed row")

    paired = []
    for filename in sorted(baseline):
        before = baseline[filename]
        after = enhanced[filename]
        paired.append(
            {
                "id": before["id"],
                "file_name": filename,
                "split": evidence[filename]["split"],
                "asr_context_changed": (
                    before["asr_normalized"] != after["asr_normalized"]
                ),
                "baseline_em": int(before["asr_em"]),
                "enhanced_em": int(after["asr_em"]),
                "baseline_f1": float(before["asr_f1"]),
                "enhanced_f1": float(after["asr_f1"]),
                "llm_answer_changed": (
                    before["asr_llm_answer_normalized"]
                    != after["asr_llm_answer_normalized"]
                ),
                "baseline_asr": before["asr_normalized"],
                "enhanced_asr": after["asr_normalized"],
                "baseline_answer": before["asr_llm_answer_raw"],
                "enhanced_answer": after["asr_llm_answer_raw"],
            }
        )

    changed = [row for row in paired if row["asr_context_changed"]]
    summary = {
        "all_769": _metrics(paired),
        "development_all": _metrics(
            [row for row in paired if row["split"] == "development"]
        ),
        "held_out_all": _metrics(
            [row for row in paired if row["split"] == "held_out"]
        ),
        "changed_contexts_all": _metrics(changed),
        "changed_contexts_development": _metrics(
            [row for row in changed if row["split"] == "development"]
        ),
        "changed_contexts_held_out": _metrics(
            [row for row in changed if row["split"] == "held_out"]
        ),
        "interpretation": (
            "Only explicitly recovered ASR contexts differ; unchanged frozen "
            "audio-enhanced Gemma predictions are reused exactly."
        ),
    }
    summary_path = RESULTS_DIR / "contextual_downstream_comparison_v1.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    changes_path = RESULTS_DIR / "contextual_downstream_changes_v1.csv"
    with changes_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(changed)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
