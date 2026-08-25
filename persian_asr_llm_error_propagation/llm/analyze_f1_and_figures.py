#!/usr/bin/env python3
"""Supplementary F1 degradation analysis and thesis figures for frozen Gemma V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/persian_error_propagation_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from analyze_gemma2_9b_propagation import cluster_bootstrap_statistic, parse_bool


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = (
    PROJECT_DIR
    / "analysis/results/initial_evaluation_v1/original_asr/gemma2_9b_final_joined_v1.csv"
)
DEFAULT_ANALYSIS = (
    PROJECT_DIR
    / "analysis/results/initial_evaluation_v1/original_asr/gemma2_9b_propagation_analysis_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "analysis/results/initial_evaluation_v1/original_asr/supplementary"
)
N_BOOT = 10_000
SEED = 2026


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_mean(
    dataframe: pd.DataFrame, column: str, seed: int
) -> dict[str, float | int]:
    return cluster_bootstrap_statistic(
        dataframe,
        lambda sampled: float(sampled[column].mean()),
        n_boot=N_BOOT,
        seed=seed,
    )


def f1_summary(final: pd.DataFrame) -> dict[str, Any]:
    primary = final[final["oracle_em"] == 1].copy()
    if len(final) != 769 or final["id"].nunique() != 286:
        raise ValueError("Expected 769 rows and 286 semantic IDs")
    if len(primary) != 470 or primary["id"].nunique() != 174:
        raise ValueError("Expected primary cohort of 470 rows and 174 semantic IDs")

    overall = {
        "recordings": 769,
        "semantic_ids": 286,
        "mean_oracle_f1": bootstrap_mean(final, "oracle_f1", SEED),
        "mean_asr_path_f1": bootstrap_mean(final, "asr_f1", SEED + 1),
        "mean_delta_f1": bootstrap_mean(final, "delta_f1", SEED + 2),
        "median_delta_f1": float(final["delta_f1"].median()),
        "decreased": {
            "count": int((final["delta_f1"] > 0).sum()),
            "percentage": float((final["delta_f1"] > 0).mean()),
        },
        "unchanged": {
            "count": int((final["delta_f1"] == 0).sum()),
            "percentage": float((final["delta_f1"] == 0).mean()),
        },
        "increased": {
            "count": int((final["delta_f1"] < 0).sum()),
            "percentage": float((final["delta_f1"] < 0).mean()),
        },
    }
    primary_summary = {
        "recordings": 470,
        "semantic_ids": 174,
        "mean_oracle_f1": bootstrap_mean(primary, "oracle_f1", SEED + 3),
        "mean_asr_path_f1": bootstrap_mean(primary, "asr_f1", SEED + 4),
        "mean_delta_f1": bootstrap_mean(primary, "delta_f1", SEED + 5),
        "median_delta_f1": float(primary["delta_f1"].median()),
        "decreased": {
            "count": int((primary["delta_f1"] > 0).sum()),
            "percentage": float((primary["delta_f1"] > 0).mean()),
        },
        "unchanged": {
            "count": int((primary["delta_f1"] == 0).sum()),
            "percentage": float((primary["delta_f1"] == 0).mean()),
        },
        "increased": {
            "count": int((primary["delta_f1"] < 0).sum()),
            "percentage": float((primary["delta_f1"] < 0).mean()),
        },
    }

    groups: list[dict[str, Any]] = []
    for position, preserved in enumerate((False, True)):
        subset = final[final["answer_preserved"] == preserved]
        primary_subset = primary[primary["answer_preserved"] == preserved]
        groups.append(
            {
                "answer_preserved": preserved,
                "all_recordings": int(len(subset)),
                "all_semantic_ids": int(subset["id"].nunique()),
                "all_mean_oracle_f1": float(subset["oracle_f1"].mean()),
                "all_mean_asr_path_f1": float(subset["asr_f1"].mean()),
                "all_mean_delta_f1": bootstrap_mean(
                    subset, "delta_f1", SEED + 10 + position
                ),
                "all_median_delta_f1": float(subset["delta_f1"].median()),
                "primary_recordings": int(len(primary_subset)),
                "primary_semantic_ids": int(primary_subset["id"].nunique()),
                "primary_mean_delta_f1": bootstrap_mean(
                    primary_subset, "delta_f1", SEED + 12 + position
                ),
                "primary_median_delta_f1": float(primary_subset["delta_f1"].median()),
            }
        )
    return {"all_769": overall, "primary_oracle_em_correct": primary_summary, "by_answer_preserved": groups}


def retention_by_answer_span(final: pd.DataFrame) -> pd.DataFrame:
    primary = final[final["oracle_em"] == 1]
    records: list[dict[str, Any]] = []
    for position, preserved in enumerate((False, True)):
        subset = primary[primary["answer_preserved"] == preserved]
        interval = bootstrap_mean(subset, "asr_em", SEED + 20 + position)
        records.append(
            {
                "answer_span": "Preserved" if preserved else "Lost",
                "answer_preserved": preserved,
                "recordings": int(len(subset)),
                "semantic_ids": int(subset["id"].nunique()),
                "gemma_correct": int(subset["asr_em"].sum()),
                "gemma_failed": int((subset["asr_em"] == 0).sum()),
                "retention": interval["estimate"],
                "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
            }
        )
    return pd.DataFrame(records)


def retention_by_wer_bin(final: pd.DataFrame) -> pd.DataFrame:
    primary = final[final["oracle_em"] == 1].copy()
    edges = [-0.001, 0.10, 0.20, 0.30, 0.50, 1.00, math.inf]
    labels = ["0–10%", "10–20%", "20–30%", "30–50%", "50–100%", ">100%"]
    primary["wer_bin"] = pd.cut(primary["wer"], bins=edges, labels=labels)
    records: list[dict[str, Any]] = []
    for position, label in enumerate(labels):
        subset = primary[primary["wer_bin"] == label]
        if subset.empty:
            continue
        interval = bootstrap_mean(subset, "asr_em", SEED + 30 + position)
        records.append(
            {
                "wer_bin": label,
                "recordings": int(len(subset)),
                "semantic_ids": int(subset["id"].nunique()),
                "retention": interval["estimate"],
                "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
            }
        )
    return pd.DataFrame(records)


def roc_data(final: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    primary = final[final["oracle_em"] == 1].copy()
    outcome = (primary["asr_em"] == 0).astype(int).to_numpy()
    predictors = {
        "WER": primary["wer"].to_numpy(dtype=float),
        "Answer-span loss": (~primary["answer_preserved"]).astype(int).to_numpy(),
    }
    frames: list[pd.DataFrame] = []
    aucs: dict[str, float] = {}
    for predictor, scores in predictors.items():
        false_positive, true_positive, thresholds = roc_curve(outcome, scores)
        aucs[predictor] = float(roc_auc_score(outcome, scores))
        frames.append(
            pd.DataFrame(
                {
                    "predictor": predictor,
                    "false_positive_rate": false_positive,
                    "true_positive_rate": true_positive,
                    "threshold": thresholds,
                }
            )
        )
    return pd.concat(frames, ignore_index=True), aucs


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_answer_span(data: pd.DataFrame, figures_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    colors = ["#D55E00", "#0072B2"]
    values = data["retention"].to_numpy()
    errors = np.vstack([values - data["ci_low"], data["ci_high"] - values])
    bars = axis.bar(data["answer_span"], values, color=colors, width=0.62)
    axis.errorbar(
        np.arange(len(data)), values, yerr=errors, fmt="none", color="black", capsize=5
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("Downstream exact-match retention")
    axis.set_title("Retention by exact answer-span preservation")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for bar, value, ci_high in zip(bars, values, data["ci_high"]):
        label_y = min(ci_high + 0.025, 0.97)
        axis.text(bar.get_x() + bar.get_width() / 2, label_y, f"{value:.1%}", ha="center")
    save_figure(fig, figures_dir / "figure1_retention_by_answer_span_v1")


def plot_roc(data: pd.DataFrame, aucs: dict[str, float], analysis: dict[str, Any], figures_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.2, 5.2))
    colors = {"WER": "#D55E00", "Answer-span loss": "#0072B2"}
    ci_lookup = {
        "WER": analysis["predictor_auc"]["wer_to_propagation_failure"],
        "Answer-span loss": analysis["predictor_auc"][
            "one_minus_answer_preserved_to_propagation_failure"
        ],
    }
    for predictor, group in data.groupby("predictor", sort=False):
        ci = ci_lookup[predictor]
        label = (
            f"{predictor}: AUC {aucs[predictor]:.3f} "
            f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
        )
        axis.plot(
            group["false_positive_rate"],
            group["true_positive_rate"],
            linewidth=2.4,
            color=colors[predictor],
            label=label,
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("Predicting downstream propagation failure")
    axis.legend(loc="lower right", frameon=False)
    axis.set_aspect("equal", adjustable="box")
    save_figure(fig, figures_dir / "figure2_roc_comparison_v1")


def plot_wer_bins(data: pd.DataFrame, figures_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.6, 4.6))
    positions = np.arange(len(data))
    values = data["retention"].to_numpy()
    errors = np.vstack([values - data["ci_low"], data["ci_high"] - values])
    bars = axis.bar(positions, values, color="#009E73", width=0.68)
    axis.errorbar(positions, values, yerr=errors, fmt="none", color="black", capsize=4)
    axis.set_xticks(positions, data["wer_bin"])
    axis.set_ylim(0, 1)
    axis.set_xlabel("Predefined WER bin")
    axis.set_ylabel("Downstream exact-match retention")
    axis.set_title("Downstream retention by word error rate")
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for bar, value, ci_high in zip(bars, values, data["ci_high"]):
        label_y = min(ci_high + 0.025, 0.97)
        axis.text(bar.get_x() + bar.get_width() / 2, label_y, f"{value:.1%}", ha="center", fontsize=9)
    save_figure(fig, figures_dir / "figure3_retention_by_wer_bin_v1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_hash = sha256_file(args.input)
    analysis_hash = sha256_file(args.analysis)
    final = pd.read_csv(args.input, encoding="utf-8-sig", dtype={"id": str, "file_name": str})
    final["answer_preserved"] = final["answer_preserved"].map(parse_bool)
    for column in ["oracle_em", "oracle_f1", "asr_em", "asr_f1", "delta_f1", "wer"]:
        final[column] = pd.to_numeric(final[column], errors="raise")
    frozen_analysis = json.loads(args.analysis.read_text(encoding="utf-8"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = f1_summary(final)
    span_data = retention_by_answer_span(final)
    wer_data = retention_by_wer_bin(final)
    curve_data, aucs = roc_data(final)

    span_path = args.output_dir / "figure1_retention_by_answer_span_data_v1.csv"
    roc_path = args.output_dir / "figure2_roc_comparison_data_v1.csv"
    wer_path = args.output_dir / "figure3_retention_by_wer_bin_data_v1.csv"
    span_data.to_csv(span_path, index=False, encoding="utf-8-sig")
    curve_data.to_csv(roc_path, index=False, encoding="utf-8-sig")
    wer_data.to_csv(wer_path, index=False, encoding="utf-8-sig")

    configure_plot()
    plot_answer_span(span_data, figures_dir)
    plot_roc(curve_data, aucs, frozen_analysis, figures_dir)
    plot_wer_bins(wer_data, figures_dir)

    summary.update(
        {
            "measure": "delta_f1 = oracle_f1 - asr_f1",
            "source_hashes": {
                "final_joined_sha256": input_hash,
                "frozen_analysis_sha256": analysis_hash,
            },
            "bootstrap": {
                "resampling_unit": "semantic id",
                "resamples": N_BOOT,
                "base_seed": SEED,
            },
            "figure_data_hashes": {
                span_path.name: sha256_file(span_path),
                roc_path.name: sha256_file(roc_path),
                wer_path.name: sha256_file(wer_path),
            },
        }
    )
    summary_path = args.output_dir / "f1_degradation_and_figures_summary_v1.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    if sha256_file(args.input) != input_hash or sha256_file(args.analysis) != analysis_hash:
        raise RuntimeError("Frozen source changed during supplementary analysis")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Summary: {summary_path.resolve()}")
    print(f"Figures: {figures_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
