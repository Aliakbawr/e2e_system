"""Build publication-ready ASR parameter and enhancement figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark.asr_parameter_selection.common import answer_preserved, word_error_counts


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parents[1]
RESULTS = BASE_DIR / "results"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _matrix(rows, row_key, column_key, metric):
    row_values = sorted({float(row[row_key]) for row in rows})
    column_values = sorted({float(row[column_key]) for row in rows})
    values = np.array(
        [
            [
                next(
                    float(row[metric])
                    for row in rows
                    if float(row[row_key]) == row_value
                    and float(row[column_key]) == column_value
                )
                for column_value in column_values
            ]
            for row_value in row_values
        ]
    )
    return row_values, column_values, values


def _annotated_heatmap(axis, matrix, xlabels, ylabels, title, xlabel, ylabel, *, percent=True):
    image = axis.imshow(matrix, origin="lower", aspect="auto", cmap="viridis_r")
    axis.set_xticks(range(len(xlabels)), [f"{value:g}" for value in xlabels])
    axis.set_yticks(range(len(ylabels)), [f"{value:g}" for value in ylabels])
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontweight="bold")
    midpoint = (float(matrix.min()) + float(matrix.max())) / 2
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            label = f"{100 * value:.2f}%" if percent else f"{value:.3f}"
            color = "white" if value > midpoint else "black"
            axis.text(column_index, row_index, label, ha="center", va="center", fontsize=7, color=color)
    return image


def main() -> int:
    audio_rows = _csv(RESULTS / "audio_development_grid_v1.csv")
    uncertainty_rows = _csv(RESULTS / "uncertainty_grid_v1.csv")
    audio_selection = _json(RESULTS / "audio_selection_v1.json")
    uncertainty_selection = _json(RESULTS / "uncertainty_selection_v1.json")
    audio_previous = _json(PROJECT_DIR / "benchmark/asr_audio_preprocessing/results/low_level_gain_summary_v1.json")
    audio_downstream = _json(PROJECT_DIR / "benchmark/asr_audio_preprocessing/results/downstream_comparison_v1.json")["all_769"]
    contextual = _json(PROJECT_DIR / "benchmark/asr_nbest/results/contextual_downstream_comparison_v1.json")["all_769"]

    development_uncertainty = [row for row in uncertainty_rows if row["split"] == "development"]
    audio_y, audio_x, audio_wer = _matrix(
        audio_rows, "low_level_threshold_dbfs", "target_rms_dbfs", "corpus_wer"
    )
    uncertainty_y, uncertainty_x, uncertainty_wer = _matrix(
        development_uncertainty,
        "word_confidence_threshold",
        "alternative_score_gap",
        "oracle_corpus_wer",
    )

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    image = _annotated_heatmap(
        axes[0, 0], audio_wer, audio_x, audio_y,
        "A. Audio parameter tuning (development)",
        "Target RMS (dBFS)", "Low-level threshold (dBFS)",
    )
    selected_audio = audio_selection["selected_on_development"]
    source = _csv(PROJECT_DIR / "benchmark/asr_llm_error_propagation/analysis/inputs/vosk_fleurs_qa_eval_input_v1.csv")
    split_map = {row["id"]: row["split"] for row in _csv(PROJECT_DIR / "benchmark/asr_nbest/results/semantic_split_v1.csv")}
    audio_baseline = {}
    for split_name in ("development", "held_out"):
        split_rows = [row for row in source if split_map[row["id"]] == split_name]
        edits = reference_words = preserved = 0
        for row in split_rows:
            row_edits, row_words = word_error_counts(row["reference_normalized"], row["asr_raw"])
            edits += row_edits
            reference_words += row_words
            preserved += answer_preserved(row["gold_answer_normalized"], row["asr_raw"])
        audio_baseline[split_name] = {
            "corpus_wer": edits / reference_words,
            "answer_span_rate": preserved / len(split_rows),
        }
    axes[0, 0].scatter(
        audio_x.index(float(selected_audio["target_rms_dbfs"])),
        audio_y.index(float(selected_audio["low_level_threshold_dbfs"])),
        marker="*", s=260, color="#e41a1c", edgecolor="white", linewidth=1.2,
        label="Selected on development",
    )
    axes[0, 0].legend(loc="upper left", fontsize=8)
    figure.colorbar(image, ax=axes[0, 0], label="Corpus WER")

    image = _annotated_heatmap(
        axes[0, 1], uncertainty_wer, uncertainty_x, uncertainty_y,
        "B. Confidence/N-best tuning (development)",
        "Alternative decoder-score gap", "Word-confidence threshold",
    )
    selected_uncertainty = uncertainty_selection["selected_on_development"]
    axes[0, 1].scatter(
        uncertainty_x.index(float(selected_uncertainty["alternative_score_gap"])),
        uncertainty_y.index(float(selected_uncertainty["word_confidence_threshold"])),
        marker="*", s=260, color="#e41a1c", edgecolor="white", linewidth=1.2,
    )
    figure.colorbar(image, ax=axes[0, 1], label="Oracle-assisted corpus WER")

    clarification = np.array([float(row["clarification_rate"]) for row in development_uncertainty])
    improvement = np.array([
        100 * (float(row["baseline_corpus_wer"]) - float(row["oracle_corpus_wer"]))
        for row in development_uncertainty
    ])
    precision = np.array([float(row["clarification_precision"]) for row in development_uncertainty])
    scatter = axes[1, 0].scatter(
        100 * clarification, improvement, c=precision, cmap="plasma", s=65,
        edgecolor="black", linewidth=0.35,
    )
    axes[1, 0].axvline(10, linestyle="--", color="gray", linewidth=1, label="10% development limit")
    axes[1, 0].scatter(
        100 * float(selected_uncertainty["clarification_rate"]),
        100 * (float(selected_uncertainty["baseline_corpus_wer"]) - float(selected_uncertainty["oracle_corpus_wer"])),
        marker="*", s=260, color="#e41a1c", edgecolor="white", linewidth=1.2,
        label="Selected development setting",
    )
    held = uncertainty_selection["frozen_held_out_result"]
    axes[1, 0].scatter(
        100 * float(held["clarification_rate"]),
        100 * (float(held["baseline_corpus_wer"]) - float(held["oracle_corpus_wer"])),
        marker="X", s=130, color="#377eb8", edgecolor="black", linewidth=0.7,
        label="Selected setting on held out",
    )
    axes[1, 0].set_title("C. Accuracy–clarification trade-off", fontweight="bold")
    axes[1, 0].set_xlabel("Clarification rate (%)")
    axes[1, 0].set_ylabel("Oracle-assisted WER reduction (percentage points)")
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(fontsize=8)
    figure.colorbar(scatter, ax=axes[1, 0], label="Useful clarification precision")

    old = audio_previous["summary"]
    enhancement_labels = [
        "Audio gain\nWER", "Audio gain\nanswer span", "Audio gain\nLLM F1",
        "Context recovery\nLLM EM", "Context recovery\nLLM F1",
    ]
    enhancement_values = [
        100 * (old["corpus_wer_before"] - old["corpus_wer_after"]),
        100 * (old["answer_preservation_rate_after"] - old["answer_preservation_rate_before"]),
        100 * audio_downstream["mean_f1_delta"],
        100 * contextual["em_delta"],
        100 * contextual["mean_f1_delta"],
    ]
    colors = ["#4daf4a" if value >= 0 else "#e41a1c" for value in enhancement_values]
    bars = axes[1, 1].bar(range(len(enhancement_values)), enhancement_values, color=colors)
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_xticks(range(len(enhancement_labels)), enhancement_labels, fontsize=8)
    axes[1, 1].set_ylabel("Improvement (percentage points)")
    axes[1, 1].set_title("D. Previously validated end-to-end enhancements", fontweight="bold")
    axes[1, 1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, enhancement_values):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:+.2f}", ha="center", va="bottom", fontsize=8)

    figure.suptitle("Persian ASR parameter selection and downstream enhancements", fontsize=16, fontweight="bold")
    figure.savefig(RESULTS / "asr_parameter_showcase_v1.png", dpi=220)
    figure.savefig(RESULTS / "asr_parameter_showcase_v1.pdf")
    plt.close(figure)

    current = next(
        row for row in development_uncertainty
        if float(row["word_confidence_threshold"]) == 0.65
        and float(row["alternative_score_gap"]) == 1.0
    )
    report = f"""# ASR parameter-selection results

## Selected audio parameters

The 25-cell development grid selected a low-level threshold of
**{selected_audio['low_level_threshold_dbfs']:.0f} dBFS** and target RMS of
**{selected_audio['target_rms_dbfs']:.0f} dBFS**, with maximum gain fixed at
40 dB and peak headroom fixed at -1 dBFS as safety constraints.

| Split | Baseline WER | Selected WER | WER change | Baseline answer span | Selected answer span |
|---|---:|---:|---:|---:|---:|
| Development (selection) | {100*audio_baseline['development']['corpus_wer']:.3f}% | {100*selected_audio['corpus_wer']:.3f}% | {100*(selected_audio['corpus_wer']-audio_baseline['development']['corpus_wer']):+.3f} pp | {100*audio_baseline['development']['answer_span_rate']:.2f}% | {100*selected_audio['answer_span_rate']:.2f}% |
| Held out (one frozen evaluation) | {100*audio_baseline['held_out']['corpus_wer']:.3f}% | {100*audio_selection['frozen_held_out_result']['corpus_wer']:.3f}% | {100*(audio_selection['frozen_held_out_result']['corpus_wer']-audio_baseline['held_out']['corpus_wer']):+.3f} pp | {100*audio_baseline['held_out']['answer_span_rate']:.2f}% | {100*audio_selection['frozen_held_out_result']['answer_span_rate']:.2f}% |

The selected audio profile therefore reduced held-out WER by
{100*(audio_baseline['held_out']['corpus_wer']-audio_selection['frozen_held_out_result']['corpus_wer']):.3f}
percentage points and increased held-out answer-span preservation by
{100*(audio_selection['frozen_held_out_result']['answer_span_rate']-audio_baseline['held_out']['answer_span_rate']):.2f}
percentage points.

## Selected uncertainty parameters

Under the development limits of 10% total clarification and 5% unnecessary
clarification, the selected setting is confidence **{selected_uncertainty['word_confidence_threshold']}**
and decoder-score gap **{selected_uncertainty['alternative_score_gap']}**.

| Setting/split | Corpus WER before | Oracle-assisted WER | Answer span before | Answer span after | Clarification | Unnecessary clarification |
|---|---:|---:|---:|---:|---:|---:|
| Existing 0.65/1.0, development | {100*float(current['baseline_corpus_wer']):.3f}% | {100*float(current['oracle_corpus_wer']):.3f}% | {100*float(current['baseline_answer_span_rate']):.2f}% | {100*float(current['oracle_answer_span_rate']):.2f}% | {100*float(current['clarification_rate']):.2f}% | {100*float(current['unnecessary_clarification_rate']):.2f}% |
| Selected, development | {100*selected_uncertainty['baseline_corpus_wer']:.3f}% | {100*selected_uncertainty['oracle_corpus_wer']:.3f}% | {100*selected_uncertainty['baseline_answer_span_rate']:.2f}% | {100*selected_uncertainty['oracle_answer_span_rate']:.2f}% | {100*selected_uncertainty['clarification_rate']:.2f}% | {100*selected_uncertainty['unnecessary_clarification_rate']:.2f}% |
| Selected, held out | {100*held['baseline_corpus_wer']:.3f}% | {100*held['oracle_corpus_wer']:.3f}% | {100*held['baseline_answer_span_rate']:.2f}% | {100*held['oracle_answer_span_rate']:.2f}% | {100*held['clarification_rate']:.2f}% | {100*held['unnecessary_clarification_rate']:.2f}% |

The confidence analysis uses {uncertainty_selection['cohort']['aligned_recordings']}
aligned recordings. {uncertainty_selection['cohort']['excluded_decoder_drift']}
rows were excluded because a fresh primary pass did not exactly reproduce the
frozen transcript. Oracle-assisted metrics assume a user chooses the eligible
hypothesis that preserves the answer span and then minimizes WER. They measure
clarification potential, not autonomous ASR correction.

The held-out clarification rate exceeds the development limit, so 0.45/0.5 is
the best setting under the stated development rule but should not be described
as satisfying the operational constraint on unseen data.

![ASR parameter showcase](asr_parameter_showcase_v1.png)
"""
    (RESULTS / "showcase_report_v1.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
