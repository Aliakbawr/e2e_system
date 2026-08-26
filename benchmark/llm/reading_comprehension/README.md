# ParsiNLU reading comprehension

This task contains 600 evaluation questions per model.

```text
checkpoints/       Raw model generations and token-F1 values
scored_outputs/    Outputs augmented with semantic and exact-match scoring
model_summaries/   Original one-row model summaries
parsinlu_reading_comprehension_ranking.csv
```

The consolidated ranking uses Exact Match, mean token F1, and mean semantic
similarity. The Qwen2.5 3B upload contains its complete checkpoint but no
separate scored-output or summary export; its ranking values are retained from
the supplied final evaluation table. Its checkpoint mean F1 is independently
consistent with the ranking value.

## Results

| Rank | Model | Exact Match ↑ | Mean F1 ↑ | Semantic similarity ↑ |
|---:|---|---:|---:|---:|
| 1 | **Gemma 2 9B** | **25.17%** | **0.5811** | **0.7839** |
| 2 | Gemma 2 2B | 15.17% | 0.4805 | 0.7123 |
| 3 | Qwen2.5 7B | 13.50% | 0.4409 | 0.7160 |
| 4 | Dorna2 8B | 9.67% | 0.3928 | 0.6636 |
| 5 | Qwen2.5 3B | 6.33% | 0.3650 | 0.6647 |
