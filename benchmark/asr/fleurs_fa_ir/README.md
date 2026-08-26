# Persian FLEURS ASR benchmark

This directory mirrors the Common Voice 25 benchmark layout for the official
Persian FLEURS (`fa_ir`) test split.

## Contents

- `fleurs_asr_evaluation.ipynb`: preserved Colab evaluation workflow.
- `*_checkpoint.csv`: raw per-recording outputs and timings for all 871 test
  recordings.
- `fleurs_fa_ir_asr_summary.csv`: consolidated ASR metrics and runtime totals.

The NeMo hybrid model has separate RNNT and CTC rows because both decoding
strategies were evaluated. WER and CER come from the frozen notebook exports.
RTF is total inference time divided by total audio duration, and average
latency is total inference time divided by 871 recordings.

GPU-memory measurements were not stored by the FLEURS evaluation, so
`gpu_memory_gb` is intentionally empty rather than estimated. The checkpoint
CSV files retain their original Colab audio paths and prediction text.

## Results

| Model | Samples | WER ↓ | CER ↓ | RTF ↓ | Avg latency (s) ↓ | Inference time (s) ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Vosk `fa-0.42` | 871 | **0.1675** | **0.0628** | 0.2820 | 4.3131 | 3,756.71 |
| Wav2Vec2 XLSR Persian v3 | 871 | 0.2956 | 0.0904 | 0.0165 | 0.2523 | 219.77 |
| NeMo FastConformer CTC | 871 | 0.2972 | 0.1058 | **0.0076** | **0.1163** | **101.29** |
| Whisper large-v3 | 871 | 0.3130 | 0.0913 | 0.2545 | 3.8923 | 3,390.19 |
| NeMo FastConformer RNNT | 871 | 0.4332 | 0.2777 | 0.0102 | 0.1555 | 135.48 |

Total evaluated audio duration was 13,321.86 seconds for every configuration.
