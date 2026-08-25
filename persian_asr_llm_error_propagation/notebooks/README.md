# Experiment notebooks

Place future notebooks in the stage matching their purpose:

```text
01_asr/                    ASR inspection and transcription experiments
02_qa_benchmark/           QA construction and validation notebooks
03_llm_inference/          Oracle and ASR-path LLM experiments
04_statistical_analysis/   Final statistical analyses and figures
99_sandbox/                Temporary exploration not used as evidence
```

Recommended naming convention:

`NN_short_description_vN.ipynb`

For example: `01_vosk_checkpoint_validation_v1.ipynb`.

Notebook outputs should be cleared before version control when they contain large tensors, audio, model logs, or duplicated tables. A notebook used for a thesis result should record its input hashes and write durable outputs into the corresponding versioned directory under `analysis/results/`.

