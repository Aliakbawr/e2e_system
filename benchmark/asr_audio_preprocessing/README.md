# ASR audio-preprocessing ablation

This experiment targets the 206 abnormally low-level recordings discovered in
the frozen 769-row Vosk propagation cohort. It applies gain only below
`-45 dBFS`, targets `-24 dBFS`, caps gain at `40 dB`, and preserves `1 dB` peak
headroom. All other recordings reuse their frozen Vosk transcript unchanged.

The current decoder must first reproduce a deterministic sample from the frozen
baseline. The evaluator then saves an append-only checkpoint and can resume:

```bash
/home/aliakbar/miniconda3/envs/persian_assistant/bin/python \
  -m benchmark.asr_audio_preprocessing.run_ablation
```

The completed paired experiment applies preprocessing to 206/769 recordings.
Corpus WER improves from 12.740% to 12.472%; the semantic-ID cluster-bootstrap
95% interval for the change is -0.445 to -0.100 percentage points. Answer-span
preservation changes from 65.93% to 66.32% (8 gained, 5 lost), but its confidence
interval crosses zero. The profile is therefore accepted for its reliable WER
gain; the answer-span change is treated as preliminary.

Production enables this profile by default. Set
`PERSIAN_ASSISTANT_ASR_AUDIO_PREPROCESSING=none` for rollback or an A/B control.

The full frozen Gemma replay also completed. Overall mean F1 improves from
0.6952 to 0.7040 with a semantic-ID cluster-bootstrap 95% interval of +0.0026
to +0.0162. Exact match improves by 0.39 percentage points, but its interval
crosses zero. See `results/summary_v1.md` and
`results/downstream_comparison_v1.json` for the paired results.
