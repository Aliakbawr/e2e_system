# FLEURS ASR model comparison v1

This directory preserves the original Colab exports for the five ASR
configurations evaluated on all 871 Persian FLEURS (`fa_ir`) test recordings.

## Layout

```text
frozen_outputs/
  wav2vec2_large_xlsr_persian_v3/
  whisper_large_v3/
  nemo_fastconformer_rnnt/
  nemo_fastconformer_ctc/
  vosk_model_fa_0_42/
source_archives/
  asr_comparison-20260825T224310Z-1-001.zip
```

Each model directory contains the checkpoint, final results, and exported
summary produced by the notebook. The CSV contents, including their original
Google Colab audio paths, are intentionally unchanged.

The Vosk checkpoint in `frozen_outputs/vosk_model_fa_0_42/` is byte-identical
to `asr/vosk_fleurs_checkpoint.csv` (SHA-256
`49b906e9838fd6b8c77635dfe97aeab7b82a5aeac73cff2bc0bee675f298f443`).
The duplicate is retained so the extracted archive remains complete.

The summary metrics are frozen notebook exports. Consult the relevant
normalization convention before comparing them with later thesis analyses.
