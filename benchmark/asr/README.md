# Persian ASR benchmarks

```text
common_voice_25/   Mozilla Common Voice 25 Persian test evaluation
fleurs_fa_ir/      Official FLEURS Persian (`fa_ir`) test evaluation
```

Each dataset directory contains its evaluation notebook, frozen model
checkpoints, and a consolidated numeric summary CSV.

## Best results

| Dataset | Best WER | WER | CER |
|---|---|---:|---:|
| Common Voice 25 Persian | NeMo FastConformer hybrid large | 12.33% | 3.79% |
| FLEURS `fa_ir` | Vosk `fa-0.42` | 16.75% | 6.28% |

Results are not directly pooled across datasets. See each dataset README for
all models, runtime metrics, and evaluation-specific notes.
