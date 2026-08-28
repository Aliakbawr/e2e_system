# Chatbot enhancement tracking

This directory makes each robustness enhancement a reproducible, paired
experiment. Do not rely on one final test: that hides regressions and makes it
impossible to attribute a gain to a particular change.

## Evaluation cadence

| When | Evaluation | Purpose |
|---|---|---|
| Every code change | Unit/regression tests | Catch local breakage quickly |
| Every enhancement stage | Noisy multi-turn control suite | Measure routing, clarification, resolution, and memory |
| Every ASR-text stage | Frozen 769-row propagation proxy | Audit answer-span gains/losses and movement toward the reference |
| Prompt/output stages | Noisy multi-turn local-LLM run | Check whether the delivered context produces the intended answer |
| Milestones and final | Full frozen 769-prompt Gemma replay | Measure actual ASR → LLM downstream QA improvement |
| Final only | Sealed real-audio conversations | Estimate generalization without tuning on the showcase set |

The propagation proxy is intentionally labeled as an intermediate metric. A
better answer-span rate or token error rate is encouraging, but it does not
prove that Gemma answered more questions correctly.

## Record the current baseline

The initial snapshot uses the identity processor, representing behavior before
new ASR transcript preprocessing is introduced:

```bash
python3 -m benchmark.enhancement_tracking.run_stage \
  --stage-id 00_preprocessing_baseline \
  --description "Before ASR and LLM pre/postprocessing enhancements" \
  --processor identity \
  --llm-result benchmark/chatbot_multiturn/results/gemma2_9b_v1.json
```

After an enhancement, expose the runtime function as
`src.asr.text.preprocess_asr_text`, then record the next stage:

```bash
python3 -m benchmark.enhancement_tracking.run_stage \
  --stage-id 01_persian_text_normalization \
  --description "Safe Persian ASR text canonicalization" \
  --processor runtime
```

Stage files are immutable by default: an existing stage ID is never
overwritten. Each artifact records dataset hashes, Git revision, dirty paths,
configuration, aggregate metrics, and paired per-example outcomes.

## Show the improvement

```bash
python3 -m benchmark.enhancement_tracking.compare_stages \
  benchmark/enhancement_tracking/results/00_preprocessing_baseline.json \
  benchmark/enhancement_tracking/results/01_persian_text_normalization.json \
  --output benchmark/enhancement_tracking/results/00_to_01.md
```

Use the same frozen datasets for all stage comparisons. Add difficult examples
to a new dataset version rather than silently editing the version used by an
earlier result.

## Final held-out set

Keep a separate set of roughly 20–30 real spoken multi-turn sessions sealed
until the enhancement sequence is complete. Balance it across critical names,
technical terms, numbers, negation, follow-up references, topic switches, and
ordinary clean questions. Report task success, incorrect-answer rate,
clarification precision/recall, recovery after correction, and end-to-end
latency. Never replace the frozen development suites with this set; they serve
different purposes.
