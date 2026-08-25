# Gemma 2 9B propagation: Vosk versus the original ASR condition

## Frozen design

The Vosk condition reuses the same 769 recording-question pairs, 286 semantic IDs, frozen QA questions and gold answers, Gemma 2 9B model archive, prompt, chat template, deterministic decoding parameters, answer normalization, EM/F1 scoring, and 286 frozen Oracle predictions. The only experimental change is the context supplied to Gemma: the stored Vosk transcript replaces the original ASR transcript. No audio was retranscribed.

The primary cohort remains the previously defined 470 recordings from 174 semantic IDs for which Gemma achieved exact match under the clean Oracle condition. Confidence intervals resample semantic IDs and retain all recordings belonging to each sampled ID.

## Transcript-level comparison

| Metric on the 769 matched recordings | Original ASR | Vosk |
|---|---:|---:|
| Corpus WER | 45.52% | 12.74% |
| Corpus CER | 26.91% | 5.43% |
| Mean recording WER | 44.22% | 12.88% |
| Exact answer-span preservation | 15.34% | 65.93% |

For all 871 Vosk recordings, corpus WER was 12.79% and corpus CER was 5.48%. The downstream comparison uses only the 769 recordings attached to the frozen usable QA items.

## Gemma performance

| Outcome | Original ASR | Vosk | Vosk − original |
|---|---:|---:|---:|
| Overall ASR-path EM, all 769 | 9.62% | 39.53% | +29.91 pp |
| Mean ASR-path F1, all 769 | 0.3402 | 0.6952 | +0.3550 (95% CI 0.3173–0.3925) |
| Primary retention, 470 Oracle-correct recordings | 14.68% | 62.98% | +48.30 pp (95% CI 41.83–54.60 pp) |
| Primary propagation failure | 85.32% | 37.02% | −48.30 pp |
| Primary mean ASR-path F1 | 0.3677 | 0.7948 | +0.4270 (95% CI 0.3787–0.4755) |

Vosk primary retention was 296/470, with a semantic-ID cluster-bootstrap 95% CI of 56.73%–69.18%. Its propagation-failure rate was 174/470, 37.02% (95% CI 30.82%–43.27%). The improvement is therefore large, but Vosk transcription still caused more than one third of previously answerable recording-level cases to fail exact match.

## Paired primary outcomes

Because both ASR conditions use the same 470 recordings, their outcomes can be compared directly:

| Outcome pair | Recordings |
|---|---:|
| Both correct | 63 |
| Vosk correct, original ASR failed | 233 |
| Original ASR correct, Vosk failed | 6 |
| Both failed | 168 |

This pairing shows that the aggregate gain is not merely caused by different item composition: Vosk recovered 233 recording-level answers that failed under the original ASR context, while the reverse occurred in only 6 cases.

## Answer-span preservation within the Vosk condition

| Exact gold span in Vosk | Recordings | Gemma correct | Gemma failed | Retention |
|---|---:|---:|---:|---:|
| Lost | 165 | 7 | 158 | 4.24% |
| Preserved | 305 | 289 | 16 | 94.75% |
| Total | 470 | 296 | 174 | 62.98% |

Answer-span loss remained more predictive of propagation failure than WER:

- AUC, WER → failure: 0.8045 (95% CI 0.7540–0.8519)
- AUC, answer-span loss → failure: 0.9422 (95% CI 0.9096–0.9702)

After controlling for WER and gold-answer length, exact answer-span preservation remained strongly associated with lower odds of failure: OR 0.00291 (cluster-bootstrap 95% CI 0.00036–0.00706). This is an association, not a causal estimate.

## Continuous F1 degradation from Oracle

Across all 769 recordings, mean token F1 fell from 0.8466 under Oracle text to 0.6952 under Vosk, a mean drop of 0.1514 (95% CI 0.1209–0.1834). F1 decreased in 257/769 cases (33.42%), remained unchanged in 496/769 (64.50%), and increased in 16/769 (2.08%).

Within the primary cohort, mean F1 drop was 0.2052 (95% CI 0.1631–0.2497). When the answer span was lost, the primary mean drop was 0.5299; when preserved, it was only 0.0296.

## Interpretation

The Vosk transcripts produced substantially better downstream Gemma performance than the original ASR transcripts on the identical frozen benchmark. The paired retention gain was 48.30 percentage points, alongside a 32.78-point reduction in matched-set corpus WER and a 50.59-point increase in exact answer-span preservation.

The Vosk result also replicates the central finding from the original experiment: overall WER is informative, but whether task-relevant answer text survives is a stronger predictor of downstream failure.

## Reproducibility artifacts

- Stored Vosk input: `asr/vosk_fleurs_checkpoint.csv`
- Prepared Vosk/QA table: `analysis/inputs/vosk_fleurs_qa_eval_input_v1.csv`
- Raw Gemma predictions: `llm/gemma2_9b_vosk_path_predictions_v1.csv`
- Vosk joined table and analysis: `analysis/results/initial_evaluation_v1/vosk/`
- Paired comparison: `analysis/results/initial_evaluation_v1/comparisons/asr_conditions/`
- Vosk figures and F1 summary: `analysis/results/initial_evaluation_v1/vosk/supplementary/`
- Historical manifest: `analysis/manifests/historical_pre_restructure/initial_evaluation_v1/FROZEN_GEMMA2_9B_VOSK_PATH_V1.sha256`
