# Benchmark artifacts

```text
asr/                          Standalone Persian ASR evaluations
llm/                          Standalone Persian LLM evaluations
asr_llm_error_propagation/    End-to-end FLEURS ASR → Gemma QA experiment
```

Dataset-specific directories contain the raw checkpoints and notebooks, while
consolidated CSV files provide analysis-ready model rankings.

## Evaluation setups

| Setup | Dataset | Primary result |
|---|---|---:|
| ASR | Common Voice 25 Persian | NeMo WER 12.33% |
| ASR | FLEURS `fa_ir` | Vosk WER 16.75% |
| LLM reading comprehension | ParsiNLU RC | Gemma 2 9B EM 25.17%, F1 0.5811 |
| LLM multiple choice | ParsiNLU MCQ | Gemma 2 9B accuracy 46.86% |
| ASR → LLM propagation | FLEURS-derived frozen QA | Original-ASR retention 14.68%; Vosk retention 62.98% |

The standalone ASR/LLM rankings establish model performance. The propagation
setup is a separate experiment and must not be interpreted as another row in
either standalone ranking.
