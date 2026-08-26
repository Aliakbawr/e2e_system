# Persian Common Voice 25 ASR benchmark

This directory contains the evaluation notebook, four 10,519-row model
checkpoints, and the consolidated `cv25_asr_summary.csv` ranking.

## Results

| Model | Samples | WER ↓ | CER ↓ | RTF ↓ | Avg latency (s) ↓ | GPU memory (GB) | Inference time (s) ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| NeMo FastConformer hybrid large | 10,519 | **0.1233** | **0.0379** | 0.0275 | 0.1359 | 0.88 | 1,429.43 |
| Vosk `fa-0.42` | 10,519 | 0.2234 | 0.0593 | 0.3051 | 1.5098 | **0.00** | 15,881.76 |
| Wav2Vec2 XLSR Persian v3 | 10,519 | 0.2615 | 0.1205 | **0.0150** | **0.0744** | 1.64 | **782.60** |
| Whisper large-v3 | 10,519 | 0.3677 | 0.1064 | 0.2944 | 1.4568 | 6.70 | 15,324.19 |

Total evaluated audio duration was 52,047.28 seconds for every model.

`nemo_final_results_legacy.csv` was found among the uploaded LLM files and was
moved here because it is an ASR artifact. Its WER/CER values differ from the
current consolidated table, so it is retained as a legacy source export and is
not the primary Common Voice result.
