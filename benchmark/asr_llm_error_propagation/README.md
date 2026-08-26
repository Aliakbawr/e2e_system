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

The project remains self-contained for frozen-result verification. The
standalone five-model FLEURS ranking is presented separately in
`../asr/fleurs_fa_ir/`; duplicated comparison exports here are retained only as
frozen provenance.

The current primary metric convention is `analysis/results/nemo_paper_v1/`. Earlier results are retained under `analysis/results/initial_evaluation_v1/` for provenance and sensitivity comparison.

See [analysis/README.md](analysis/README.md) for the analysis map and [notebooks/README.md](notebooks/README.md) before adding notebooks.

## Local dataset files

The official FLEURS audio archive and extracted WAV files are intentionally
excluded from Git because of their size. Keep them locally at `data/test.tar.gz`
and `data/test/`. The current integrity manifest records the local archive hash,
so a full manifest verification requires restoring the dataset first.

## Primary evaluation results

The primary reporting convention is `nemo_paper_v1`. The frozen QA-matched
dataset contains 769 recordings from 286 semantic IDs. The primary downstream
cohort contains 470 recordings from 174 IDs for which Gemma 2 9B achieved exact
match under the clean-transcript Oracle condition.

| Outcome | Original ASR | Vosk |
|---|---:|---:|
| Corpus WER, all 871 recordings | 43.30% | 16.75% |
| Corpus CER, all 871 recordings | 27.76% | 6.28% |
| Exact answer-span preservation, matched 769 | 15.34% | 65.93% |
| Overall Gemma Exact Match | 9.62% | 39.53% |
| Overall mean Gemma F1 | 0.3402 | 0.6952 |
| Primary downstream retention | 14.68% | 62.98% |
| Primary propagation failure | 85.32% | 37.02% |

Primary retention used semantic-ID cluster-bootstrap confidence intervals:
10.19%–19.48% for the original ASR and 56.73%–69.18% for Vosk.

## Predictor comparison

| Condition | AUC: WER | AUC: answer-span loss | Paired ΔAUC | Cluster-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Original ASR | 0.7711 | 0.8943 | **0.1233** | **0.0416–0.2112** |
| Vosk | 0.7620 | 0.9422 | **0.1802** | **0.1219–0.2411** |

Here, `ΔAUC = AUC(answer-span loss) − AUC(WER)`. Both paired intervals exclude
zero, supporting the conclusion that exact task-relevant span loss discriminates
downstream failure better than overall WER in these frozen conditions.
