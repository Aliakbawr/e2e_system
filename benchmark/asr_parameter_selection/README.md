# ASR parameter selection and showcase

This experiment separates parameter selection from final evaluation. Semantic
IDs—not recordings—define the existing development/held-out split, so repeated
recordings of one sentence cannot leak across partitions.

## 1. Audio-level parameters

The audio grid tunes the two performance parameters on development data:

- low-level threshold: `-55, -50, -45, -40, -35 dBFS`;
- target RMS: `-30, -27, -24, -21, -18 dBFS`.

Maximum gain (`40 dB`) and peak headroom (`-1 dBFS`) are held fixed as safety
constraints. After selection, only the selected configuration is decoded on
held-out data.

```bash
/home/aliakbar/miniconda3/envs/persian_assistant/bin/python \
  -m benchmark.asr_parameter_selection.run_audio_sweep
```

The decoder checkpoint is append-only and resumable. Use `--max-new N` for a
short smoke run and `--workers N` to control concurrent recognizers. Identical
gain transformations are reused across threshold settings. The final artifacts
are a development grid CSV, a selection JSON, and a WER heatmap under `results/`.

## 2. Confidence and N-best parameters

The uncertainty experiment first caches the normal Vosk pass because that pass
contains per-word confidence. It then evaluates this grid without further ASR
decoding:

- word-confidence threshold: `0.45, 0.55, 0.65, 0.75, 0.85`;
- alternative decoder-score gap: `0.25, 0.5, 1.0, 2.0, 3.0`.

```bash
/home/aliakbar/miniconda3/envs/persian_assistant/bin/python \
  -m benchmark.asr_parameter_selection.run_uncertainty_sweep
```

Selection uses development data only. By default, clarification must remain at
or below 10% and unnecessary clarification at or below 5%. Feasible settings
maximize oracle-assisted answer-span preservation, then minimize WER and
clarification burden. Held-out results are reported only after selection.

“Oracle-assisted” is deliberately explicit: when the policy asks, the analysis
assumes the user chooses the eligible hypothesis that preserves the answer span
and then minimizes WER. It measures the potential value of clarification, not
an autonomous transcript-selection algorithm.

The confidence pass is accepted only when its transcript exactly reproduces the
frozen primary transcript, ensuring word positions remain aligned. Any decoder-
drift rows are excluded and listed by filename and reason in the selection JSON.

## Presentation

Use the generated heatmaps together with these five columns:

| Configuration | Corpus WER | Answer-span rate | Clarification rate | Unnecessary clarification rate |
|---|---:|---:|---:|---:|

Report development results as parameter-selection evidence and the single
held-out result as the unbiased performance estimate. Do not describe the
existing `0.65`/`1.0` operating point as optimal unless this sweep selects it.

After both experiments complete, build the combined PNG/PDF showcase and its
Markdown results summary with:

```bash
/home/aliakbar/miniconda3/envs/persian_assistant/bin/python \
  -m benchmark.asr_parameter_selection.build_showcase
```
