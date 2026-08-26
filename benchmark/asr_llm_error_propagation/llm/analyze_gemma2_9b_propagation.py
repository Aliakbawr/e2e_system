#!/usr/bin/env python3
"""Analyze frozen Oracle and ASR-path Gemma 2 9B predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_ASR = SCRIPT_DIR / "gemma2_9b_asr_path_predictions_v1.csv"
DEFAULT_ORACLE = SCRIPT_DIR / "gemma2_9b_oracle_predictions_v1.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "analysis/results/initial_evaluation_v1/original_asr"
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def roc_auc(scores: np.ndarray, outcomes: np.ndarray) -> float:
    outcomes = np.asarray(outcomes, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = int(outcomes.sum())
    negatives = len(outcomes) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float(
        (ranks[outcomes == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )


def cluster_bootstrap_statistic(
    df: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    cluster_col: str = "id",
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    clusters = [group.index.to_numpy() for _, group in df.groupby(cluster_col, sort=False)]
    if not clusters:
        raise ValueError("Cannot bootstrap an empty dataframe")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    for iteration in range(n_boot):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.concatenate([clusters[index] for index in sampled])
        # iloc needs positional indices, so operate on a reset-index frame.
        estimates[iteration] = statistic(df.loc[indices])
    finite = estimates[np.isfinite(estimates)]
    if not len(finite):
        raise RuntimeError("No finite bootstrap estimates")
    return {
        "estimate": float(statistic(df)),
        "ci_low": float(np.quantile(finite, 0.025)),
        "ci_high": float(np.quantile(finite, 0.975)),
        "confidence_level": 0.95,
        "bootstrap_resamples": int(n_boot),
        "finite_resamples": int(len(finite)),
        "seed": int(seed),
        "clusters": int(len(clusters)),
        "recordings": int(len(df)),
    }


def logistic_fit(matrix: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Fit unpenalized logistic regression using stable Newton/IRLS updates."""
    matrix = np.asarray(matrix, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    coefficients = np.zeros(matrix.shape[1], dtype=float)
    ridge = np.eye(matrix.shape[1]) * 1e-8
    ridge[0, 0] = 0.0
    for _ in range(100):
        linear = np.clip(matrix @ coefficients, -30, 30)
        probability = 1 / (1 + np.exp(-linear))
        weights = np.maximum(probability * (1 - probability), 1e-8)
        hessian = matrix.T @ (weights[:, None] * matrix) + ridge
        gradient = matrix.T @ (outcomes - probability) - ridge @ coefficients
        step = np.linalg.solve(hessian, gradient)
        coefficients += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return coefficients


def logistic_cluster_bootstrap(
    df: pd.DataFrame,
    predictors: list[str],
    n_boot: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    working = df.reset_index(drop=True)
    names = ["intercept", *predictors]
    matrix = np.column_stack(
        [np.ones(len(working)), *[working[column].astype(float) for column in predictors]]
    )
    outcomes = working["propagation_failure"].astype(int).to_numpy()
    coefficients = logistic_fit(matrix, outcomes)
    clusters = [group.index.to_numpy() for _, group in working.groupby("id", sort=False)]
    rng = np.random.default_rng(seed)
    boot = np.full((n_boot, len(names)), np.nan, dtype=float)
    for iteration in range(n_boot):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.concatenate([clusters[index] for index in sampled])
        sampled_outcomes = outcomes[indices]
        if sampled_outcomes.min() == sampled_outcomes.max():
            continue
        try:
            boot[iteration] = logistic_fit(matrix[indices], sampled_outcomes)
        except np.linalg.LinAlgError:
            continue
    result: dict[str, Any] = {
        "outcome": "propagation_failure",
        "recordings": int(len(working)),
        "clusters": int(len(clusters)),
        "bootstrap_resamples": int(n_boot),
        "seed": int(seed),
        "predictors": {},
    }
    for position, name in enumerate(names):
        finite = boot[np.isfinite(boot[:, position]), position]
        result["predictors"][name] = {
            "coefficient": float(coefficients[position]),
            "odds_ratio": float(math.exp(np.clip(coefficients[position], -30, 30))),
            "coefficient_ci_low": float(np.quantile(finite, 0.025)),
            "coefficient_ci_high": float(np.quantile(finite, 0.975)),
            "odds_ratio_ci_low": float(math.exp(np.clip(np.quantile(finite, 0.025), -30, 30))),
            "odds_ratio_ci_high": float(math.exp(np.clip(np.quantile(finite, 0.975), -30, 30))),
            "finite_resamples": int(len(finite)),
        }
    fitted_scores = matrix @ coefficients
    result["descriptive_in_sample_auc"] = roc_auc(fitted_scores, outcomes)
    return result


def describe_series(series: pd.Series) -> dict[str, float]:
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "q25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "q75": float(series.quantile(0.75)),
        "max": float(series.max()),
    }


def load_and_join(asr_path: Path, oracle_path: Path) -> pd.DataFrame:
    asr = pd.read_csv(asr_path, dtype={"id": str, "qa_id": str, "file_name": str})
    oracle = pd.read_csv(oracle_path, dtype={"id": str, "qa_id": str})
    if len(asr) != 769 or asr["file_name"].nunique() != 769:
        raise ValueError("ASR predictions must contain 769 unique recordings")
    if len(oracle) != 286 or oracle["id"].nunique() != 286:
        raise ValueError("Oracle predictions must contain 286 unique semantic IDs")
    if not (asr["status"] == "ok").all() or not (oracle["status"] == "ok").all():
        raise ValueError("Prediction files contain failed rows")

    oracle_columns = [
        "id",
        "oracle_answer_raw",
        "oracle_answer_normalized",
        "oracle_em",
        "oracle_f1",
    ]
    final = asr.merge(
        oracle[oracle_columns],
        on="id",
        how="left",
        validate="many_to_one",
    )
    if len(final) != 769 or final["id"].nunique() != 286:
        raise ValueError("Joined predictions have unexpected dimensions")
    if final["oracle_em"].isna().any():
        raise ValueError("Joined predictions contain missing Oracle scores")
    final["answer_preserved"] = final["answer_preserved"].map(parse_bool)
    for column in ["wer", "cer", "asr_em", "asr_f1", "oracle_em", "oracle_f1"]:
        final[column] = pd.to_numeric(final[column], errors="raise")
    final["delta_f1"] = final["oracle_f1"] - final["asr_f1"]
    final["propagation_failure"] = pd.array([pd.NA] * len(final), dtype="Int64")
    oracle_correct = final["oracle_em"] == 1
    final.loc[oracle_correct, "propagation_failure"] = (
        final.loc[oracle_correct, "asr_em"] == 0
    ).astype(int)
    return final


def build_analysis(final: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    primary = final[final["oracle_em"] == 1].copy().reset_index(drop=True)
    if len(primary) != 470 or primary["id"].nunique() != 174:
        raise ValueError(
            f"Expected primary subset 470 rows/174 IDs, found "
            f"{len(primary)} rows/{primary['id'].nunique()} IDs"
        )
    primary["propagation_failure"] = primary["propagation_failure"].astype(int)
    primary["answer_not_preserved"] = (~primary["answer_preserved"]).astype(int)
    primary["wer_x_answer_preserved"] = (
        primary["wer"] * primary["answer_preserved"].astype(int)
    )

    retention = cluster_bootstrap_statistic(
        primary, lambda frame: float(frame["asr_em"].mean()), seed=42
    )
    # Failure is exactly 1 - retention, including its interval endpoints.
    failure = {
        **retention,
        "estimate": 1 - float(retention["estimate"]),
        "ci_low": 1 - float(retention["ci_high"]),
        "ci_high": 1 - float(retention["ci_low"]),
        "derived_as": "1 - retention",
    }
    delta_all = cluster_bootstrap_statistic(
        final.reset_index(drop=True), lambda frame: float(frame["delta_f1"].mean()), seed=44
    )
    delta_primary = cluster_bootstrap_statistic(
        primary, lambda frame: float(frame["delta_f1"].mean()), seed=45
    )
    auc_wer = cluster_bootstrap_statistic(
        primary,
        lambda frame: roc_auc(frame["wer"].to_numpy(), frame["propagation_failure"].to_numpy()),
        seed=46,
    )
    auc_preservation = cluster_bootstrap_statistic(
        primary,
        lambda frame: roc_auc(
            frame["answer_not_preserved"].to_numpy(),
            frame["propagation_failure"].to_numpy(),
        ),
        seed=47,
    )

    preservation_table = (
        primary.groupby("answer_preserved", observed=True)
        .agg(
            oracle_correct_recordings=("file_name", "count"),
            semantic_ids=("id", "nunique"),
            asr_em=("asr_em", "mean"),
            propagation_failure=("propagation_failure", "mean"),
            mean_wer=("wer", "mean"),
        )
        .reset_index()
    )
    preservation_records = [
        {
            "answer_preserved": bool(row["answer_preserved"]),
            "oracle_correct_recordings": int(row["oracle_correct_recordings"]),
            "semantic_ids": int(row["semantic_ids"]),
            "asr_em": float(row["asr_em"]),
            "propagation_failure": float(row["propagation_failure"]),
            "mean_wer": float(row["mean_wer"]),
        }
        for _, row in preservation_table.iterrows()
    ]

    edges = [-0.001, 0.10, 0.20, 0.30, 0.50, 1.00, math.inf]
    labels = ["0–10%", "10–20%", "20–30%", "30–50%", "50–100%", ">100%"]
    primary["wer_bin"] = pd.cut(primary["wer"], bins=edges, labels=labels)
    wer_bins = (
        primary.groupby("wer_bin", observed=False)
        .agg(
            recordings=("file_name", "count"),
            semantic_ids=("id", "nunique"),
            downstream_retention=("asr_em", "mean"),
            propagation_failure=("propagation_failure", "mean"),
        )
        .reset_index()
    )
    wer_bin_records = [
        {
            "wer_bin": str(row["wer_bin"]),
            "recordings": int(row["recordings"]),
            "semantic_ids": int(row["semantic_ids"]),
            "downstream_retention": None
            if pd.isna(row["downstream_retention"])
            else float(row["downstream_retention"]),
            "propagation_failure": None
            if pd.isna(row["propagation_failure"])
            else float(row["propagation_failure"]),
        }
        for _, row in wer_bins.iterrows()
    ]

    adjusted = logistic_cluster_bootstrap(
        primary,
        ["wer", "answer_preserved", "gold_answer_token_count"],
        seed=48,
    )
    interaction = logistic_cluster_bootstrap(
        primary,
        [
            "wer",
            "answer_preserved",
            "gold_answer_token_count",
            "wer_x_answer_preserved",
        ],
        seed=49,
    )

    result = {
        "primary_definition": "oracle_em == 1",
        "all_recordings": 769,
        "all_semantic_ids": 286,
        "primary_recordings": 470,
        "primary_semantic_ids": 174,
        "retention": retention,
        "propagation_failure": failure,
        "delta_f1_all_769": {
            "distribution": describe_series(final["delta_f1"]),
            "cluster_bootstrap_mean": delta_all,
            "asr_better_count_delta_below_zero": int((final["delta_f1"] < 0).sum()),
            "no_change_count": int((final["delta_f1"] == 0).sum()),
            "asr_worse_count_delta_above_zero": int((final["delta_f1"] > 0).sum()),
        },
        "delta_f1_primary": {
            "distribution": describe_series(primary["delta_f1"]),
            "cluster_bootstrap_mean": delta_primary,
        },
        "answer_preservation_primary_table": preservation_records,
        "predictor_auc": {
            "wer_to_propagation_failure": auc_wer,
            "one_minus_answer_preserved_to_propagation_failure": auc_preservation,
        },
        "adjusted_logistic_model": adjusted,
        "exploratory_interaction_model": interaction,
        "wer_cer_pearson_correlation_primary": float(primary["wer"].corr(primary["cer"])),
        "predefined_wer_bins": wer_bin_records,
        "bootstrap_note": (
            "All reported bootstrap intervals resample semantic IDs and retain all "
            "recordings belonging to each sampled ID."
        ),
    }
    return result, primary


def save_qualitative_groups(primary: pd.DataFrame, output_dir: Path) -> dict[str, int]:
    groups = {
        "low_wer_failure": primary[(primary["wer"] <= 0.20) & (primary["asr_em"] == 0)],
        "high_wer_success": primary[(primary["wer"] >= 0.50) & (primary["asr_em"] == 1)],
        "answer_lost_llm_correct": primary[
            (~primary["answer_preserved"]) & (primary["asr_em"] == 1)
        ],
        "answer_preserved_llm_failure": primary[
            primary["answer_preserved"] & (primary["asr_em"] == 0)
        ],
    }
    counts: dict[str, int] = {}
    for name, group in groups.items():
        path = output_dir / f"qualitative_{name}_v1.csv"
        group.to_csv(path, index=False, encoding="utf-8-sig")
        counts[name] = int(len(group))
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr", type=Path, default=DEFAULT_ASR)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asr_hash = sha256_file(args.asr)
    oracle_hash = sha256_file(args.oracle)
    final = load_and_join(args.asr, args.oracle)
    analysis, primary = build_analysis(final)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    final_path = args.output_dir / "gemma2_9b_final_joined_v1.csv"
    analysis_path = args.output_dir / "gemma2_9b_propagation_analysis_v1.json"
    final.to_csv(final_path, index=False, encoding="utf-8-sig")
    analysis["qualitative_group_counts"] = save_qualitative_groups(primary, args.output_dir)
    analysis["source_hashes"] = {
        "asr_predictions_sha256": asr_hash,
        "oracle_predictions_sha256": oracle_hash,
    }
    analysis["final_joined_sha256"] = sha256_file(final_path)
    with analysis_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    if sha256_file(args.asr) != asr_hash or sha256_file(args.oracle) != oracle_hash:
        raise RuntimeError("Frozen prediction input changed during analysis")
    print(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Final joined table: {final_path.resolve()}")
    print(f"Analysis: {analysis_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
