# ParsiNLU multiple-choice evaluation

This task contains 1,050 questions per model: 350 each for common knowledge,
literature, and math/logic.

```text
checkpoints/       Per-question predictions and correctness
model_summaries/   Original category and overall summaries
parsinlu_mcq_ranking.csv
```

The file uploaded as the second Dorna `(1)` checkpoint is the Gemma 2 9B run:
its 1,050 predictions reproduce the supplied Gemma category accuracies, and its
paired summary names `google/gemma-2-9b-it`. The files were therefore renamed
to `gemma2-9b-it_*`. The original summary's incorrect Dorna `model_slug` field
is preserved as source metadata and should not be used as the model identity.

Qwen2.5 3B has a complete checkpoint but no separate summary export. Its
category and overall results in the consolidated ranking are reproduced
directly from the checkpoint.

## Results

| Rank | Model | Common knowledge ↑ | Literature ↑ | Math & logic ↑ | Overall ↑ |
|---:|---|---:|---:|---:|---:|
| 1 | **Gemma 2 9B** | **56.57%** | **47.71%** | 36.29% | **46.86%** |
| 2 | Qwen2.5 7B | 45.14% | 33.43% | **42.57%** | 40.38% |
| 3 | Dorna2 8B | 40.86% | 37.43% | 32.57% | 36.95% |
| 4 | Qwen2.5 3B | 33.71% | 26.86% | 28.00% | 29.52% |
| 5 | Gemma 2 2B | 34.29% | 25.43% | 17.43% | 25.71% |
