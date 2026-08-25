# Original ASR and Vosk evaluation with NeMo-paper normalization

## Evaluation version

This analysis applies the supplied `nemo_paper_normalize`, alignment, and corpus aggregation code to both frozen 871-recording ASR checkpoints. Previous frozen results were preserved. New per-recording WER/CER values were joined to the already frozen Gemma predictions, after which only analyses involving WER or CER were recomputed.

The frozen QA questions, gold answers, answer-preservation definition, Oracle predictions, ASR-path Gemma predictions, EM, F1, retention, and propagation-failure outcomes were not changed.

`SKIP = {"ā", "š", "="}` is declared in the supplied code but is not referenced by `nemo_paper_normalize`. This evaluation therefore preserves that behavior exactly and does not independently remove those symbols.

## Corpus ASR metrics

### Full FLEURS test set: 871 recordings

| Condition | WER | CER | Word substitutions | Deletions | Insertions |
|---|---:|---:|---:|---:|---:|
| Original ASR | **43.30%** | **27.76%** | 4,421 | 4,054 | 226 |
| Vosk | **16.75%** | **6.28%** | 1,872 | 568 | 925 |

The common denominators were 20,093 reference words and 83,406 reference characters. Spaces were excluded from CER as specified.

### Frozen QA-matched subset: 769 recordings

| Condition | Corpus WER | Corpus CER | Mean recording WER | Mean recording CER |
|---|---:|---:|---:|---:|
| Original ASR | **44.05%** | **28.08%** | 43.11% | 26.78% |
| Vosk | **16.92%** | **6.23%** | 16.77% | 6.42% |

Vosk's mean recording WER was 26.34 percentage points lower than the original ASR's on the paired 769 recordings (semantic-ID cluster-bootstrap 95% CI: 24.57–28.12 points lower).

## Downstream results that remain unchanged

| Outcome | Original ASR | Vosk |
|---|---:|---:|
| Answer-span preservation, all 769 | 15.34% | 65.93% |
| Overall Gemma EM | 9.62% | 39.53% |
| Overall mean Gemma F1 | 0.3402 | 0.6952 |
| Primary retention | 14.68% | 62.98% |
| Primary propagation failure | 85.32% | 37.02% |

These quantities depend on the frozen transcripts and Gemma outputs, not on the WER/CER normalization convention.

## Revised predictor discrimination

The primary cohort contains 470 recordings from 174 semantic IDs with `oracle_em == 1`.

| Condition | AUC: WER | AUC: answer-span loss | Paired ΔAUC | Cluster-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Original ASR | 0.7711 | 0.8943 | **0.1233** | **0.0416–0.2112** |
| Vosk | 0.7620 | 0.9422 | **0.1802** | **0.1219–0.2411** |

Here `ΔAUC = AUC(answer-span loss) − AUC(WER)`. Both intervals were calculated directly from 10,000 paired semantic-ID cluster-bootstrap resamples and exclude zero. Thus, the conclusion that answer-span loss provides stronger discrimination of downstream failure than WER remains supported under the supplied evaluation normalization.

## Adjusted association

After controlling for the revised WER and gold-answer length, answer-span preservation remained associated with substantially lower odds of downstream failure:

- Original ASR: OR 0.0131, cluster-bootstrap 95% CI 0.0022–0.0321.
- Vosk: OR 0.00268, cluster-bootstrap 95% CI 0.00034–0.00639.

These are associations, not causal effects. The WER odds ratios represent a full one-unit increase in WER and should not be interpreted as effects per percentage point.

## Recommended thesis statement

> Using the predefined NeMo-paper text normalization and corpus scoring procedure, WER was 43.30% for the original ASR system and 16.75% for Vosk on all 871 Persian FLEURS recordings. Within the Oracle-correct downstream cohort, exact answer-span loss discriminated propagation failure better than WER for both systems: paired ΔAUC = 0.123 (semantic-ID cluster-bootstrap 95% CI [0.042, 0.211]) for the original ASR and ΔAUC = 0.180 (95% CI [0.122, 0.241]) for Vosk.

## Reproducibility artifacts

- Evaluation script: `asr/evaluate_checkpoints_nemo_paper.py`
- Corpus and recording summary: `analysis/results/nemo_paper_v1/asr_metrics/nemo_paper_metrics_summary_v1.json`
- Per-recording metrics and revised joined tables: `analysis/results/nemo_paper_v1/asr_metrics/`
- Revised original-ASR propagation analysis: `analysis/results/nemo_paper_v1/original_asr/`
- Revised Vosk propagation analysis: `analysis/results/nemo_paper_v1/vosk/`
- Paired condition comparison: `analysis/results/nemo_paper_v1/comparisons/asr_conditions/`
- Paired ΔAUC summary and 20,000 replicates: `analysis/results/nemo_paper_v1/comparisons/paired_delta_auc/`
- Current integrity manifest: `analysis/manifests/current/RESEARCH_ARTIFACTS_V1.sha256`
