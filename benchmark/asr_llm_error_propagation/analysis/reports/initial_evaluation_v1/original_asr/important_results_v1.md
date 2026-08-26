# Gemma 2 9B ASR Error-Propagation Results — Frozen V1

Status: **Frozen**  
Date: **2026-08-25**

## Experimental integrity

- Oracle calls: **286/286**, one per semantic ID
- ASR-path calls: **769/769**, one per matched recording-question pair
- Failed or empty calls: **0**
- Same Gemma 2 9B IT NF4 model, prompt, chat template, deterministic decoding,
  maximum output length, normalization, EM, and F1 in both conditions
- Only condition difference: Oracle used `reference_normalized`; ASR path used
  `asr_normalized`
- Primary inclusion criterion fixed before ASR-path analysis: `oracle_em == 1`
- Primary cohort: **174 semantic IDs and 470 recordings**
- All confidence intervals below use **10,000 semantic-ID cluster bootstrap
  resamples**, unless explicitly described otherwise

## Oracle baseline

| Measure | Result |
|---|---:|
| Oracle exact match | 174/286 = **60.84%** |
| Oracle mean token F1 | **0.8485** |

## Primary downstream outcome

Among recordings whose semantic QA item Gemma answered exactly correctly from
the clean reference:

| Outcome | Result | Cluster-bootstrap 95% CI |
|---|---:|---:|
| Retention, `P(ASR EM=1 | Oracle EM=1)` | 69/470 = **14.68%** | **10.19%–19.48%** |
| Propagation failure | 401/470 = **85.32%** | **80.52%–89.81%** |

Propagation failure is defined exactly as `1 - retention`; its interval is the
complement of the retention interval.

## Continuous F1 degradation

`delta_f1 = oracle_f1 - asr_f1`, so positive values mean ASR hurt performance.

| Cohort | Mean delta F1 | Cluster-bootstrap 95% CI | Median |
|---|---:|---:|---:|
| All 769 recordings | **0.5064** | 0.4646–0.5481 | 0.5000 |
| Primary 470 recordings | **0.6323** | 0.5821–0.6810 | 0.6667 |

Across all 769 recordings, ASR-path F1 was lower in **604**, unchanged in
**147**, and higher than Oracle in **18** cases.

## Exact answer-span preservation and downstream failure

Primary Oracle-correct cohort:

| Exact answer span | Recordings | Semantic IDs | ASR EM retention | Propagation failure | Mean WER |
|---|---:|---:|---:|---:|---:|
| Lost | 398 | 160 | **3.02%** | **96.98%** | 46.95% |
| Preserved | 72 | 40 | **79.17%** | **20.83%** | 31.27% |

Therefore:

- **12** recordings lost the literal answer span but Gemma still answered
  exactly correctly.
- **15** recordings preserved the literal answer span but Gemma nevertheless
  failed exact match.

These exceptions confirm that exact-span preservation is an intermediate
lexical variable—not the final semantic outcome—while the large group
difference shows that it is highly informative for downstream failure.

## Predictive comparison

Outcome: propagation failure within the 470-recording primary cohort.

| Predictor | ROC AUC | Cluster-bootstrap 95% CI |
|---|---:|---:|
| WER | **0.7633** | 0.6875–0.8346 |
| `1 - AnswerPreserved` | **0.8943** | 0.8291–0.9497 |

Exact answer-span loss discriminated downstream failures substantially better
than overall WER in this dataset.

## Adjusted association model

An unpenalized logistic model used:

`PropagationFailure ~ WER + AnswerPreserved + GoldAnswerTokenCount`

with coefficient intervals obtained by semantic-ID cluster bootstrap.

| Predictor | Odds ratio | Cluster-bootstrap 95% CI |
|---|---:|---:|
| WER, per full 1.0 increase | 20.89 | 0.93–1885.48 |
| Answer preserved | **0.0124** | **0.0021–0.0306** |
| Gold-answer token count | 1.084 | 0.779–2.437 |

The WER estimate is imprecise after adjustment. Exact answer-span preservation
retains a strong association with lower failure odds after accounting for WER
and answer length. This is an association model, not a causal effect estimate.

The exploratory `WER × AnswerPreserved` interaction was highly uncertain and
did not provide clear evidence of an interaction. Its result should remain
secondary.

WER and CER were strongly correlated in the primary cohort (`r = 0.7988`), so
they were not entered together in the primary adjusted model.

## Predefined WER bins

| WER bin | Recordings | Semantic IDs | Retention | Failure |
|---|---:|---:|---:|---:|
| 0–10% | 8 | 6 | 62.50% | 37.50% |
| 10–20% | 26 | 17 | 34.62% | 65.38% |
| 20–30% | 59 | 39 | 32.20% | 67.80% |
| 30–50% | 209 | 120 | 13.40% | 86.60% |
| 50–100% | 167 | 94 | 4.79% | 95.21% |
| >100% | 1 | 1 | 0.00% | 100.00% |

The first and final bins are small, but the predefined table shows a strong
overall decline in downstream retention as WER increases.

## Qualitative groups saved

| Group | Rows |
|---|---:|
| Low WER (`<=20%`) + propagation failure | 20 |
| High WER (`>=50%`) + downstream success | 9 |
| Answer span lost + Gemma still correct | 12 |
| Answer span preserved + Gemma fails | 15 |

## Main interpretation

The frozen experimental chain supports the following conclusion:

> ASR corruption caused substantial downstream degradation among questions
> Gemma could answer from clean text. WER was meaningfully associated with
> failure, but exact task-relevant answer-span preservation supplied strong
> additional predictive information after accounting for WER and gold-answer
> length.

This preserves the thesis structure:

`WER/CER → lexical task-information preservation → downstream LLM performance`

## Artifacts

- Oracle predictions: `llm/gemma2_9b_oracle_predictions_v1.csv`
- ASR-path predictions: `llm/gemma2_9b_asr_path_predictions_v1.csv`
- ASR-path metadata: `llm/gemma2_9b_asr_path_predictions_v1.metadata.json`
- Final joined table:
  `analysis/results/initial_evaluation_v1/original_asr/gemma2_9b_final_joined_v1.csv`
- Full analysis:
  `analysis/results/initial_evaluation_v1/original_asr/gemma2_9b_propagation_analysis_v1.json`
- Four qualitative CSVs: `analysis/results/initial_evaluation_v1/original_asr/qualitative_*.csv`
- Historical freeze manifest: `analysis/manifests/historical_pre_restructure/initial_evaluation_v1/FROZEN_GEMMA2_9B_ASR_PATH_V1.sha256`

## Primary hashes

- Oracle predictions:
  `286cc1990fc74d285524ab72c98dd356fff4f796ae61f8438693604eda49ee63`
- ASR-path predictions:
  `a2798b8b82fe72ecbd05203379d6ea8565d90b8e6a02a50a7aacc11c4c91bd30`
- Final joined table:
  `55a771a42a95bfbdc3277a945d697ccdba5847339ef2f9d1d24fd206c5906162`
- Full analysis JSON:
  `a373bc3dceee345613fb2ae5983211664f7d1d3a6ee78e7b026bf90ab1b1c9e1`
