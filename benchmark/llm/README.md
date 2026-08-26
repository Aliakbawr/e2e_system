# Persian LLM benchmarks

This directory contains the experiment notebooks, checkpoints, scored outputs,
and summaries for two ParsiNLU evaluation tasks.

```text
datasets/                    Uploaded source dataset artifacts
notebooks/                   Colab evaluation and pilot notebooks
reading_comprehension/       ParsiNLU RC checkpoints, scores, and ranking
multiple_choice/             ParsiNLU MCQ checkpoints, summaries, and ranking
```

All consolidated ranking metrics are stored as proportions from 0 to 1 rather
than formatted percentages.

## Best results

| Task | Best model | Primary result |
|---|---|---:|
| Reading comprehension | Gemma 2 9B | EM 25.17%, mean F1 0.5811, semantic similarity 0.7839 |
| Multiple choice | Gemma 2 9B | Overall accuracy 46.86% |
