#!/usr/bin/env python3
"""Paired cluster-bootstrap ΔAUC for WER versus answer-span loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark.asr_llm_error_propagation.llm.analyze_gemma2_9b_propagation import parse_bool, roc_auc, sha256_file


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
    PROJECT_DIR / "analysis/results/initial_evaluation_v1/comparisons/paired_delta_auc"
)
N_BOOT = 10_000
BASE_SEED = 271_828


def load_primary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"id": str, "qa_id": str, "file_name": str},
    )
    if len(frame) != 769 or frame["id"].nunique() != 286:
        raise ValueError(f"Expected 769 rows and 286 semantic IDs: {path}")
    frame["oracle_em"] = pd.to_numeric(frame["oracle_em"], errors="raise")
    frame["asr_em"] = pd.to_numeric(frame["asr_em"], errors="raise")
    frame["wer"] = pd.to_numeric(frame["wer"], errors="raise")
    frame["answer_preserved"] = frame["answer_preserved"].map(parse_bool)
    primary = frame[frame["oracle_em"] == 1].copy().reset_index(drop=True)
    if len(primary) != 470 or primary["id"].nunique() != 174:
        raise ValueError(f"Expected primary cohort of 470 rows/174 IDs: {path}")
    primary["propagation_failure"] = (primary["asr_em"] == 0).astype(int)
    primary["answer_span_loss"] = (~primary["answer_preserved"]).astype(int)
    if primary["propagation_failure"].nunique() != 2:
        raise ValueError("Primary cohort must contain both downstream outcomes")
    return primary


def auc_values(frame: pd.DataFrame) -> tuple[float, float, float]:
    outcomes = frame["propagation_failure"].to_numpy(dtype=int)
    auc_wer = roc_auc(frame["wer"].to_numpy(dtype=float), outcomes)
    auc_span_loss = roc_auc(
        frame["answer_span_loss"].to_numpy(dtype=float), outcomes
    )
    return auc_wer, auc_span_loss, auc_span_loss - auc_wer


def paired_cluster_bootstrap(
    frame: pd.DataFrame,
    condition: str,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    clusters = [
        group.index.to_numpy()
        for _, group in frame.groupby("id", sort=False)
    ]
    rng = np.random.default_rng(seed)
    replicates: list[dict[str, float | int | str]] = []
    for iteration in range(N_BOOT):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.concatenate([clusters[index] for index in sampled])
        bootstrap_sample = frame.iloc[indices]
        if bootstrap_sample["propagation_failure"].nunique() != 2:
            continue
        auc_wer, auc_span_loss, delta_auc = auc_values(bootstrap_sample)
        if np.isfinite(delta_auc):
            replicates.append(
                {
                    "condition": condition,
                    "iteration": iteration,
                    "auc_wer": auc_wer,
                    "auc_answer_span_loss": auc_span_loss,
                    "delta_auc": delta_auc,
                }
            )
    replicate_frame = pd.DataFrame(replicates)
    if len(replicate_frame) != N_BOOT:
        raise RuntimeError(
            f"Expected {N_BOOT} finite replicates for {condition}, "
            f"found {len(replicate_frame)}"
        )
    auc_wer, auc_span_loss, delta_auc = auc_values(frame)
    delta_values = replicate_frame["delta_auc"].to_numpy()
    summary = {
        "condition": condition,
        "primary_recordings": len(frame),
        "primary_semantic_ids": frame["id"].nunique(),
        "outcome": "propagation_failure among oracle_em == 1 recordings",
        "predictor_1": "wer",
        "predictor_2": "answer_span_loss = 1 - answer_preserved",
        "auc_wer": auc_wer,
        "auc_answer_span_loss": auc_span_loss,
        "delta_auc_definition": "AUC(answer_span_loss) - AUC(WER)",
        "delta_auc": delta_auc,
        "delta_auc_ci_low": float(np.quantile(delta_values, 0.025)),
        "delta_auc_ci_high": float(np.quantile(delta_values, 0.975)),
        "confidence_level": 0.95,
        "bootstrap_resamples": N_BOOT,
        "finite_resamples": len(replicate_frame),
        "seed": seed,
        "resampling_unit": "semantic id",
        "recordings_retained_per_sampled_id": True,
        "bootstrap_probability_delta_auc_gt_zero": float(
            (delta_values > 0).mean()
        ),
    }
    return summary, replicate_frame


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
    original = load_primary(args.original)
    vosk = load_primary(args.vosk)
    identity_columns = ["id", "qa_id", "file_name"]
    if not original[identity_columns].equals(vosk[identity_columns]):
        raise ValueError("Original and Vosk primary rows are not identically ordered")

    original_summary, original_replicates = paired_cluster_bootstrap(
        original, "original_asr", BASE_SEED
    )
    vosk_summary, vosk_replicates = paired_cluster_bootstrap(
        vosk, "vosk", BASE_SEED + 1
    )
    replicates = pd.concat(
        [original_replicates, vosk_replicates], ignore_index=True
    )
    result = {
        "analysis": "paired_delta_auc_v1",
        "interpretation": (
            "Positive delta_auc means answer-span loss discriminates downstream "
            "propagation failure better than WER. Each delta is calculated within "
            "the same semantic-ID bootstrap sample."
        ),
        "original_asr": original_summary,
        "vosk": vosk_summary,
        "source_hashes": {
            "original_final_joined_sha256": original_hash,
            "vosk_final_joined_sha256": vosk_hash,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "paired_delta_auc_summary_v1.json"
    replicates_path = args.output_dir / "paired_delta_auc_bootstrap_replicates_v1.csv"
    replicates.to_csv(replicates_path, index=False, encoding="utf-8-sig")
    result["output_artifacts"] = {
        "bootstrap_replicates_sha256": sha256_file(replicates_path),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if sha256_file(args.original) != original_hash or sha256_file(args.vosk) != vosk_hash:
        raise RuntimeError("A frozen joined input changed during ΔAUC analysis")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path.resolve()}")
    print(f"Replicates: {replicates_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
