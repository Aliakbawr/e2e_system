#!/usr/bin/env python3
"""Render the Vosk answer-span figure with room for a near-100% CI label."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.asr_llm_error_propagation.llm.analyze_f1_and_figures import configure_plot, save_figure


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = (
    PROJECT_DIR
    / "analysis/results/initial_evaluation_v1/vosk/supplementary/figure1_retention_by_answer_span_data_v1.csv"
)
OUTPUT_STEM = (
    PROJECT_DIR
    / "analysis/results/initial_evaluation_v1/vosk/supplementary/figures/figure1_retention_by_answer_span_v1"
)


def main() -> int:
    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    configure_plot()
    fig, axis = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
    values = data["retention"].to_numpy()
    errors = np.vstack([values - data["ci_low"], data["ci_high"] - values])
    bars = axis.bar(data["answer_span"], values, color=["#D55E00", "#0072B2"], width=0.62)
    axis.errorbar(
        np.arange(len(data)), values, yerr=errors, fmt="none", color="black", capsize=5
    )
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Downstream exact-match retention")
    axis.set_title("Retention by exact answer-span preservation", pad=14)
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    for bar, value, ci_high in zip(bars, values, data["ci_high"]):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            ci_high + 0.025,
            f"{value:.1%}",
            ha="center",
        )
    save_figure(fig, OUTPUT_STEM)
    print(f"Saved: {OUTPUT_STEM.with_suffix('.png')}")
    print(f"Saved: {OUTPUT_STEM.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
