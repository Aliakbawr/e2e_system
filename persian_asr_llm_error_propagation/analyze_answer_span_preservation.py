#!/usr/bin/env python3
"""Cluster-aware analysis of exact gold answer-span survival after ASR.

This script reads—but never modifies—the frozen downstream evaluation table.
`answer_preserved` is interpreted only as exact normalized answer-span
preservation, not as general semantic preservation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "analysis/inputs/fleurs_asr_qa_eval_input_v1.csv"
DEFAULT_OUTPUT = (
    BASE_DIR / "analysis/results/answer_span/original_asr_v1/exact_answer_span_analysis_v1.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cluster_bootstrap_mean(
    df: pd.DataFrame,
    value_col: str,
    cluster_col: str = "id",
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Bootstrap a recording-level mean by resampling whole semantic IDs.

    If an ID is sampled more than once, all of its recordings are duplicated,
    matching a conventional nonparametric cluster bootstrap.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    if missing := {value_col, cluster_col} - set(df.columns):
        raise ValueError(f"Missing dataframe columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Cannot bootstrap an empty dataframe")

    numeric = pd.to_numeric(df[value_col], errors="raise").astype(float)
    working = pd.DataFrame({cluster_col: df[cluster_col].to_numpy(), "value": numeric})
    clusters = working.groupby(cluster_col, sort=False)["value"].agg(["sum", "count"])
    cluster_sums = clusters["sum"].to_numpy(dtype=float)
    cluster_counts = clusters["count"].to_numpy(dtype=float)
    number_of_clusters = len(clusters)

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    # Process in chunks to avoid allocating a large n_boot x n_clusters matrix.
    chunk_size = 1_000
    for start in range(0, n_boot, chunk_size):
        stop = min(start + chunk_size, n_boot)
        sampled_indices = rng.integers(
            0,
            number_of_clusters,
            size=(stop - start, number_of_clusters),
        )
        sampled_sums = cluster_sums[sampled_indices].sum(axis=1)
        sampled_counts = cluster_counts[sampled_indices].sum(axis=1)
        estimates[start:stop] = sampled_sums / sampled_counts

    return {
        "mean": float(numeric.mean()),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "confidence_level": 0.95,
        "bootstrap_resamples": int(n_boot),
        "seed": int(seed),
        "clusters": int(number_of_clusters),
        "recordings": int(len(df)),
    }


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Descriptive non-clustered Wilson interval, retained for comparison."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return center - half_width, center + half_width


def answer_length_analysis(
    df: pd.DataFrame, n_boot: int, seed: int
) -> list[dict[str, Any]]:
    bins = [0, 1, 2, 4, math.inf]
    labels = ["1 token", "2 tokens", "3-4 tokens", "5+ tokens"]
    length_bins = pd.cut(
        df["gold_answer_token_count"],
        bins=bins,
        labels=labels,
        include_lowest=False,
    )
    analysis: list[dict[str, Any]] = []
    for offset, label in enumerate(labels):
        subset = df[length_bins == label]
        if subset.empty:
            continue
        bootstrap = cluster_bootstrap_mean(
            subset,
            "answer_preserved",
            n_boot=n_boot,
            seed=seed + offset + 1,
        )
        analysis.append(
            {
                "answer_length_bin": label,
                "recordings": int(len(subset)),
                "semantic_ids": int(subset["id"].nunique()),
                "exact_spans_preserved": int(subset["answer_preserved"].sum()),
                "exact_span_preservation_rate": float(
                    subset["answer_preserved"].mean()
                ),
                "cluster_bootstrap_ci_low": bootstrap["ci_low"],
                "cluster_bootstrap_ci_high": bootstrap["ci_high"],
                "mean_wer": float(subset["wer"].mean()),
                "median_wer": float(subset["wer"].median()),
            }
        )
    return analysis


def analyze(df: pd.DataFrame, n_boot: int, seed: int) -> dict[str, Any]:
    required = {
        "id",
        "file_name",
        "gold_answer_normalized",
        "answer_preserved",
        "wer",
    }
    if missing := required - set(df.columns):
        raise ValueError(f"Downstream table is missing columns: {sorted(missing)}")
    if df["file_name"].duplicated().any():
        raise ValueError("Downstream table contains duplicate filenames")
    if df["gold_answer_normalized"].isna().any():
        raise ValueError("Downstream table contains a missing normalized gold answer")

    df = df.copy()
    df["answer_preserved"] = df["answer_preserved"].astype(bool)
    df["gold_answer_token_count"] = df["gold_answer_normalized"].map(
        lambda text: len(str(text).split())
    )
    if (df["gold_answer_token_count"] < 1).any():
        raise ValueError("At least one normalized gold answer contains no tokens")

    successes = int(df["answer_preserved"].sum())
    wilson_low, wilson_high = wilson_interval(successes, len(df))
    cluster_result = cluster_bootstrap_mean(
        df,
        "answer_preserved",
        n_boot=n_boot,
        seed=seed,
    )

    return {
        "measure_name": "exact_answer_span_preservation_rate",
        "interpretation": (
            "Exact normalized gold string survival; this is an intermediate "
            "lexical measure, not semantic preservation or downstream QA accuracy."
        ),
        "unit_of_observation": "recording",
        "dependency_structure": (
            "Recordings are clustered by semantic id; each id shares one QA item."
        ),
        "recordings": int(len(df)),
        "semantic_ids": int(df["id"].nunique()),
        "exact_spans_preserved": successes,
        "exact_spans_lost": int(len(df) - successes),
        "recording_level_rate": float(df["answer_preserved"].mean()),
        "descriptive_wilson_95ci_ignoring_clusters": [wilson_low, wilson_high],
        "cluster_bootstrap": cluster_result,
        "gold_answer_token_count": {
            "minimum": int(df["gold_answer_token_count"].min()),
            "median": float(df["gold_answer_token_count"].median()),
            "mean": float(df["gold_answer_token_count"].mean()),
            "maximum": int(df["gold_answer_token_count"].max()),
        },
        "preservation_by_answer_length": answer_length_analysis(df, n_boot, seed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Frozen downstream table not found: {args.input}")
    input_hash_before = sha256_file(args.input)
    dataframe = pd.read_csv(
        args.input,
        encoding="utf-8-sig",
        dtype={"id": str, "file_name": str},
    )
    result = analyze(dataframe, args.n_boot, args.seed)
    result["input_sha256"] = input_hash_before

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if sha256_file(args.input) != input_hash_before:
        raise RuntimeError("Frozen downstream table changed during analysis")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Analysis written to: {args.output.resolve()}")
    print(f"Analysis SHA-256: {sha256_file(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
