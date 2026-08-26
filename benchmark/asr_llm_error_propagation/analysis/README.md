# Analysis directory

This directory contains derived research inputs, versioned results, reports, and integrity records. Raw ASR and LLM checkpoints remain outside this directory so generated analyses are clearly separated from experimental observations.

## Structure

```text
inputs/
  fleurs_asr_qa_eval_input_v1.csv     Original-ASR × frozen-QA table
  vosk_fleurs_qa_eval_input_v1.csv    Vosk × frozen-QA table
  asr_model_comparison_v1/             Colab recording and test metadata

results/
  answer_span/                         Pre-LLM answer-span analysis
  initial_evaluation_v1/               Earlier normalization convention
    original_asr/
    vosk/
    comparisons/
  nemo_paper_v1/                       Current primary normalization
    asr_metrics/
    original_asr/
    vosk/
    comparisons/

reports/
  initial_evaluation_v1/               Historical thesis summaries
  nemo_paper_v1/                       Current primary results report

manifests/
  historical_pre_restructure/          Original manifests with original paths
  current/                             Post-restructure verification manifest
```

## Interpretation rule

Use `nemo_paper_v1` for primary WER/CER reporting. LLM predictions, EM/F1, and answer-span preservation are frozen and shared across evaluation conventions. Use `initial_evaluation_v1` only as provenance or normalization-sensitivity material.

Historical manifests are intentionally unchanged: they document hashes and paths at the time each stage was frozen. The current manifest verifies files at their reorganized locations.
