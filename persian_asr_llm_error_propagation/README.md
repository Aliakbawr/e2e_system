# Persian ASR → LLM Error Propagation

Research project studying how Persian ASR errors affect downstream Gemma 2 9B question answering on the official FLEURS `fa_ir` test set.

## Project layout

```text
data/                 Official FLEURS metadata, archive, and WAV files
qa/                   Frozen QA benchmark
asr/                  ASR runners, frozen checkpoints, and model comparisons
llm/                  Gemma inference runners and frozen raw predictions
analysis/
  inputs/             Frozen recording–QA evaluation tables
  results/            Versioned statistical outputs and figures
  reports/            Thesis-oriented result summaries
  manifests/          Historical and current integrity manifests
notebooks/            Experiment notebooks, organized by research stage
```

The multi-model Colab run is preserved under `asr/model_comparison_v1/`, with
its original notebook at
`notebooks/01_asr/01_fleurs_asr_model_comparison_v1.ipynb`.

The current primary metric convention is `analysis/results/nemo_paper_v1/`. Earlier results are retained under `analysis/results/initial_evaluation_v1/` for provenance and sensitivity comparison.

See [analysis/README.md](analysis/README.md) for the analysis map and [notebooks/README.md](notebooks/README.md) before adding notebooks.

## Local dataset files

The official FLEURS audio archive and extracted WAV files are intentionally
excluded from Git because of their size. Keep them locally at `data/test.tar.gz`
and `data/test/`. The current integrity manifest records the local archive hash,
so a full manifest verification requires restoring the dataset first.
