#!/usr/bin/env python3
"""Paired comparison of frozen original-ASR and Vosk Gemma conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from benchmark.asr_llm_error_propagation.llm.analyze_gemma2_9b_propagation import parse_bool, sha256_file


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_ORIGINAL = (
    PROJECT_DIR
    / "analysis/results/initial_evaluation_v1/original_asr/gemma2_9b_final_joined_v1.csv"
)
DEFAULT_VOSK = (
    PROJECT_DIR
    / "analysis/results/initial_evaluation_v1/vosk/gemma2_9b_final_joined_v1.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "analysis/results/initial_evaluation_v1/comparisons/asr_conditions"
)
N_BOOT = 10_000
SEED = 31415


def load_pair(original_path: Path, vosk_path: Path) -> pd.DataFrame:
    dtype = {"id": str, "qa_id": str, "file_name": str}
    original = pd.read_csv(original_path, encoding="utf-8-sig", dtype=dtype)
    vosk = pd.read_csv(vosk_path, encoding="utf-8-sig", dtype=dtype)
    for name, frame in (("original", original), ("Vosk", vosk)):
        if len(frame) != 769 or frame["file_name"].nunique() != 769:
            raise ValueError(f"{name} table must contain 769 unique recordings")
        if frame["id"].nunique() != 286:
            raise ValueError(f"{name} table must contain 286 semantic IDs")

    invariant = ["id", "qa_id", "file_name", "question", "gold_answer", "oracle_em", "oracle_f1"]
    joined = original.merge(
        vosk,
        on=["id", "qa_id", "file_name"],
        how="inner",
        validate="one_to_one",
        suffixes=("_original", "_vosk"),
    )
    if len(joined) != 769:
        raise ValueError("Conditions do not contain the same recording-question pairs")
    for column in ["question", "gold_answer", "oracle_em", "oracle_f1"]:
        if not (joined[f"{column}_original"] == joined[f"{column}_vosk"]).all():
            raise ValueError(f"Frozen invariant differs between conditions: {column}")
    for suffix in ["original", "vosk"]:
        joined[f"answer_preserved_{suffix}"] = joined[f"answer_preserved_{suffix}"].map(parse_bool)
        for column in ["wer", "cer", "asr_em", "asr_f1", "oracle_em", "oracle_f1"]:
            joined[f"{column}_{suffix}"] = pd.to_numeric(
                joined[f"{column}_{suffix}"], errors="raise"
            )
    return joined


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    seed: int,
) -> dict[str, float | int]:
    working = frame.reset_index(drop=True)
    clusters = [group.index.to_numpy() for _, group in working.groupby("id", sort=False)]
    rng = np.random.default_rng(seed)
    estimates = np.empty(N_BOOT)
    for iteration in range(N_BOOT):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.concatenate([clusters[index] for index in sampled])
        estimates[iteration] = statistic(working.iloc[indices])
    return {
        "estimate": float(statistic(working)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "confidence_level": 0.95,
        "bootstrap_resamples": N_BOOT,
        "seed": seed,
        "clusters": len(clusters),
        "recordings": len(working),
    }


def condition_summary(frame: pd.DataFrame, suffix: str) -> dict[str, float | int]:
    primary = frame[frame["oracle_em_original"] == 1]
    return {
        "all_recordings": len(frame),
        "all_semantic_ids": frame["id"].nunique(),
        "mean_recording_wer": float(frame[f"wer_{suffix}"].mean()),
        "mean_recording_cer": float(frame[f"cer_{suffix}"].mean()),
        "answer_preservation_rate": float(frame[f"answer_preserved_{suffix}"].mean()),
        "asr_path_em_all": float(frame[f"asr_em_{suffix}"].mean()),
        "asr_path_mean_f1_all": float(frame[f"asr_f1_{suffix}"].mean()),
        "primary_recordings": len(primary),
        "primary_semantic_ids": primary["id"].nunique(),
        "primary_retention": float(primary[f"asr_em_{suffix}"].mean()),
        "primary_propagation_failure": float(1 - primary[f"asr_em_{suffix}"].mean()),
        "primary_mean_f1": float(primary[f"asr_f1_{suffix}"].mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--vosk", type=Path, default=DEFAULT_VOSK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    original_hash = sha256_file(args.original)
    vosk_hash = sha256_file(args.vosk)
    pair = load_pair(args.original, args.vosk)
    primary = pair[pair["oracle_em_original"] == 1].copy()
    if len(primary) != 470 or primary["id"].nunique() != 174:
        raise ValueError("Expected the frozen primary cohort of 470 recordings/174 IDs")

    differences = {
        "primary_retention_vosk_minus_original": paired_cluster_bootstrap(
            primary,
            lambda frame: float((frame["asr_em_vosk"] - frame["asr_em_original"]).mean()),
            SEED,
        ),
        "primary_mean_f1_vosk_minus_original": paired_cluster_bootstrap(
            primary,
            lambda frame: float((frame["asr_f1_vosk"] - frame["asr_f1_original"]).mean()),
            SEED + 1,
        ),
        "all_mean_f1_vosk_minus_original": paired_cluster_bootstrap(
            pair,
            lambda frame: float((frame["asr_f1_vosk"] - frame["asr_f1_original"]).mean()),
            SEED + 2,
        ),
        "answer_preservation_vosk_minus_original": paired_cluster_bootstrap(
            pair,
            lambda frame: float(
                (
                    frame["answer_preserved_vosk"].astype(int)
                    - frame["answer_preserved_original"].astype(int)
                ).mean()
            ),
            SEED + 3,
        ),
        "mean_recording_wer_vosk_minus_original": paired_cluster_bootstrap(
            pair,
            lambda frame: float((frame["wer_vosk"] - frame["wer_original"]).mean()),
            SEED + 4,
        ),
    }
    primary_pairing = {
        "both_correct": int(((primary["asr_em_original"] == 1) & (primary["asr_em_vosk"] == 1)).sum()),
        "vosk_only_correct": int(((primary["asr_em_original"] == 0) & (primary["asr_em_vosk"] == 1)).sum()),
        "original_only_correct": int(((primary["asr_em_original"] == 1) & (primary["asr_em_vosk"] == 0)).sum()),
        "both_failed": int(((primary["asr_em_original"] == 0) & (primary["asr_em_vosk"] == 0)).sum()),
    }
    result = {
        "comparison": "Gemma 2 9B with Vosk versus original ASR context",
        "paired_on": ["id", "qa_id", "file_name"],
        "frozen_oracle_primary_definition": "oracle_em == 1",
        "original_asr": condition_summary(pair, "original"),
        "vosk": condition_summary(pair, "vosk"),
        "paired_differences": differences,
        "primary_paired_outcomes": primary_pairing,
        "bootstrap_note": "Paired resampling of semantic IDs; all recordings for each sampled ID are retained.",
        "source_hashes": {
            "original_final_joined_sha256": original_hash,
            "vosk_final_joined_sha256": vosk_hash,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "gemma2_9b_vosk_vs_original_comparison_v1.json"
    csv_path = args.output_dir / "gemma2_9b_vosk_vs_original_primary_pairs_v1.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    primary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    if sha256_file(args.original) != original_hash or sha256_file(args.vosk) != vosk_hash:
        raise RuntimeError("A frozen condition changed during comparison")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Comparison: {json_path.resolve()}")
    print(f"Primary paired rows: {csv_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
