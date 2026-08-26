# Gemma 2 9B: token-F1 degradation and thesis figures

## Frozen scope

This supplementary analysis uses the already frozen Gemma 2 9B joined propagation table. It does not alter the ASR model or outputs, QA benchmark, prompt, LLM configuration, normalization, predictions, or primary propagation definition.

- Full dataset: 769 recordings nested in 286 semantic IDs.
- Primary subset: 470 recordings nested in 174 semantic IDs for which Oracle EM = 1.
- F1 degradation: `delta_f1 = oracle_f1 - asr_f1`; positive values mean that replacing the clean transcript with ASR output reduced token F1.
- All reported confidence intervals use 10,000 cluster-bootstrap resamples with semantic `id` as the sampling unit.

## Continuous F1 degradation

### All 769 recording-question pairs

| Measure | Result |
|---|---:|
| Mean Oracle F1 | 0.8466 (95% CI 0.8141–0.8765) |
| Mean ASR-path F1 | 0.3402 (95% CI 0.3045–0.3767) |
| Mean F1 drop | 0.5064 (95% CI 0.4648–0.5489) |
| Median F1 drop | 0.5000 |
| F1 decreased | 604/769 (78.54%) |
| F1 unchanged | 147/769 (19.12%) |
| F1 increased | 18/769 (2.34%) |

### Primary Oracle-EM-correct subset

| Measure | Result |
|---|---:|
| Mean Oracle F1 | 1.0000 |
| Mean ASR-path F1 | 0.3677 (95% CI 0.3186–0.4168) |
| Mean F1 drop | 0.6323 (95% CI 0.5842–0.6813) |
| Median F1 drop | 0.6667 |
| F1 decreased | 401/470 (85.32%) |
| F1 unchanged | 69/470 (14.68%) |
| F1 increased | 0/470 (0.00%) |

The continuous result confirms the primary EM finding: ASR replacement caused a large performance reduction, including when partial lexical overlap is credited rather than requiring exact match.

## F1 degradation by exact answer-span preservation

Across all 769 recordings:

| Exact gold span | Recordings | Mean Oracle F1 | Mean ASR F1 | Mean F1 drop | Median drop |
|---|---:|---:|---:|---:|---:|
| Lost | 651 | 0.8597 | 0.2728 | 0.5870 (95% CI 0.5460–0.6275) | 0.6000 |
| Preserved | 118 | 0.7743 | 0.7124 | 0.0619 (95% CI -0.0042–0.1416) | 0.0000 |

In the primary Oracle-EM-correct subset, mean F1 drop was 0.7210 (95% CI 0.6762–0.7653) when the exact span was lost, compared with 0.1420 (95% CI 0.0543–0.2543) when it was preserved. This supports treating exact span preservation as a strong intermediate lexical measure, while the exceptions and nonzero degradation in preserved cases show that it is not itself a complete measure of semantic preservation.

## Thesis figures

1. [Retention by exact answer-span preservation (PNG)](../../../results/initial_evaluation_v1/original_asr/supplementary/figures/figure1_retention_by_answer_span_v1.png) · [PDF](../../../results/initial_evaluation_v1/original_asr/supplementary/figures/figure1_retention_by_answer_span_v1.pdf)
2. [ROC comparison: WER versus answer-span loss (PNG)](../../../results/initial_evaluation_v1/original_asr/supplementary/figures/figure2_roc_comparison_v1.png) · [PDF](../../../results/initial_evaluation_v1/original_asr/supplementary/figures/figure2_roc_comparison_v1.pdf)
3. [Retention by predefined WER bin (PNG)](../../../results/initial_evaluation_v1/original_asr/supplementary/figures/figure3_retention_by_wer_bin_v1.png) · [PDF](../../../results/initial_evaluation_v1/original_asr/supplementary/figures/figure3_retention_by_wer_bin_v1.pdf)

Figure 3 uses the prespecified bins 0–10%, 10–20%, 20–30%, 30–50%, 50–100%, and >100%. Retention declines from 62.5% in the first bin to 4.8% in the 50–100% bin. The 0–10% bin contains only 8 recordings and the >100% bin only one, so their estimates and confidence intervals should be interpreted with their small sample sizes in view.

## Reproducibility files

- Analysis script: `llm/analyze_f1_and_figures.py`
- Machine-readable summary: `analysis/results/initial_evaluation_v1/original_asr/supplementary/f1_degradation_and_figures_summary_v1.json`
- Figure source tables: `analysis/results/initial_evaluation_v1/original_asr/supplementary/figure{1,2,3}_*_data_v1.csv`
- Historical manifest: `analysis/manifests/historical_pre_restructure/initial_evaluation_v1/FROZEN_GEMMA2_9B_SUPPLEMENTARY_V1.sha256`
