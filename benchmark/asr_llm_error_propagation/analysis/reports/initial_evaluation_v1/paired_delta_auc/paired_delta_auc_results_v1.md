# Paired ΔAUC analysis

## Method

For each ASR condition, discrimination of downstream propagation failure was compared between:

- WER
- Exact answer-span loss, defined as `1 - answer_preserved`

The comparison quantity was calculated directly as:

`ΔAUC = AUC(answer-span loss → propagation failure) − AUC(WER → propagation failure)`

The analysis used the frozen primary cohort: 470 recordings nested within 174 semantic IDs for which `oracle_em == 1`. For each of 10,000 bootstrap iterations, the 174 semantic IDs were sampled with replacement, all recordings belonging to each sampled ID were retained, and both AUCs and their difference were calculated within the same resample. This paired procedure retains the covariance between the two AUC estimates.

## Results

| ASR condition | AUC: WER | AUC: answer-span loss | Paired ΔAUC | Cluster-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Original ASR | 0.7633 | 0.8943 | **0.1310** | **0.0478–0.2178** |
| Vosk | 0.8045 | 0.9422 | **0.1377** | **0.0895–0.1889** |

Both confidence intervals exclude zero. Therefore, in both frozen ASR conditions, exact answer-span loss discriminated downstream Gemma failures significantly better than WER under the paired semantic-ID cluster-bootstrap analysis.

The fraction of bootstrap differences greater than zero was 0.9993 for the original ASR condition and 1.0000 for Vosk. These fractions are descriptive bootstrap diagnostics; the percentile confidence intervals are the reported inferential quantities.

## Suggested thesis wording

> Exact answer-span loss discriminated downstream propagation failures better than WER under both ASR conditions. For the original ASR system, the paired AUC improvement was ΔAUC = 0.131 (semantic-ID cluster-bootstrap 95% CI [0.048, 0.218]); for Vosk, the improvement was ΔAUC = 0.138 (95% CI [0.089, 0.189]). Thus, the advantage of task-relevant lexical preservation over aggregate transcription error was replicated across two ASR systems.

This supports a statement about comparative predictive discrimination. It does not by itself establish that answer-span loss causally produces downstream failure.

## Reproducibility artifacts

- Analysis script: `llm/analyze_paired_delta_auc.py`
- Machine-readable summary: `analysis/results/initial_evaluation_v1/comparisons/paired_delta_auc/paired_delta_auc_summary_v1.json`
- All 20,000 paired bootstrap replicates: `analysis/results/initial_evaluation_v1/comparisons/paired_delta_auc/paired_delta_auc_bootstrap_replicates_v1.csv`
- Historical manifest: `analysis/manifests/historical_pre_restructure/initial_evaluation_v1/FROZEN_PAIRED_DELTA_AUC_V1.sha256`
