# Benchmark artifacts

```text
asr/                          Standalone Persian ASR evaluations
llm/                          Standalone Persian LLM evaluations
tts/                          Standalone Persian TTS evaluations
asr_llm_error_propagation/    End-to-end FLEURS ASR → Gemma QA experiment
chatbot_multiturn/             Noisy multi-turn runtime regression/challenge suite
enhancement_tracking/          Paired stage snapshots and improvement reports
```

The ASR and LLM dataset directories contain raw checkpoints, notebooks, and
consolidated analysis-ready rankings. The TTS directory currently preserves its
Colab notebooks and supplied final tables; its referenced WAV and CSV exports
are external artifacts.

## Evaluation setups

| Setup | Dataset | Primary result |
|---|---|---:|
| ASR | Common Voice 25 Persian | NeMo WER 12.33% |
| ASR | FLEURS `fa_ir` | Vosk WER 16.75% |
| LLM reading comprehension | ParsiNLU RC | Gemma 2 9B EM 25.17%, F1 0.5811 |
| LLM multiple choice | ParsiNLU MCQ | Gemma 2 9B accuracy 46.86% |
| TTS | Random 10% of Common Voice 25 Persian (1,052 texts) | Piper MOS 4.534 and WER 0.107; VITS MMS has the fastest reported runtime |
| ASR → LLM propagation | FLEURS-derived frozen QA | Original-ASR retention 14.68%; Vosk retention 62.98% |

The standalone ASR, LLM, and TTS rankings establish component performance. The
propagation setup is a separate experiment and must not be interpreted as
another row in a standalone ranking. See [`tts/README.md`](tts/README.md) for
the TTS quality, intelligibility, runtime tables, and notebook inventory.

For chatbot enhancement work, use the staged protocol in
`enhancement_tracking/README.md`. It reuses the frozen propagation data after
each relevant change, reserves the expensive full LLM replay for milestones,
and keeps a separate held-out real-audio test for the final generalization
check.
