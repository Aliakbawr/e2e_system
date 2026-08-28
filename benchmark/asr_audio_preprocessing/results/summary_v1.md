# Low-level audio repair: paired Vosk ablation

The `low_level_gain` profile was evaluated on the same frozen 769 FLEURS
recordings used by the existing Vosk → Gemma propagation benchmark.

| Metric | Baseline | Enhanced | Change |
|---|---:|---:|---:|
| Corpus WER | 12.740% | 12.472% | -0.268 pp |
| Mean recording WER | 12.880% | 12.614% | -0.266 pp |
| Answer-span preservation | 65.930% | 66.320% | +0.390 pp |

- Preprocessing applied: 206/769 recordings
- Transcripts changed: 119
- Recording WER: 63 improved, 31 regressed, 675 unchanged
- Answer spans: 8 gained, 5 lost
- Semantic-ID cluster-bootstrap WER delta 95% CI: -0.445 to -0.100 pp
- Bootstrap probability of WER improvement: 99.9%
- Answer-preservation delta 95% CI: -0.632 to +1.448 pp

Decision: enable the narrowly gated profile for its statistically supported WER
gain. Do not claim a reliable downstream QA gain until the enhanced transcripts
complete the frozen Gemma replay.

## Frozen Gemma replay

All 769 enhanced contexts completed the same frozen Gemma 2 9B prompt and
scoring protocol with no failures.

| Downstream metric | Baseline Vosk | Enhanced audio | Change |
|---|---:|---:|---:|
| Exact match | 39.532% | 39.922% | +0.390 pp |
| Mean token F1 | 0.6952 | 0.7040 | +0.0088 |
| Primary-cohort retention | 62.979% | 63.404% | +0.426 pp |

- ASR contexts changed: 117/769
- LLM answers changed: 33/769
- Exact matches: 7 gained, 4 lost
- Cluster-bootstrap EM delta 95% CI: -0.394 to +1.289 pp
- Cluster-bootstrap mean-F1 delta 95% CI: +0.0026 to +0.0162
- Bootstrap probability of mean-F1 improvement: 99.72%

Updated conclusion: audio repair produces a reliable ASR WER improvement and a
reliable overall downstream mean-F1 improvement. The small exact-match and
primary-retention increases are directional only because their confidence
intervals cross zero.
