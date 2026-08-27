# Noisy multi-turn chatbot evaluation

This suite evaluates the deployed dialogue-control behavior before any ASR
correction fine-tuning or LLM replacement is considered. It uses hand-authored
Persian ASR hypotheses with word confidence and decoder scores, then sends them
through the same text-only state machine used by the runtime.

## Coverage

- Critical term, name, number, and negation corruption
- Direct and ordinal clarification replies
- Correction memory beyond recent dialogue history
- Pronoun-based follow-up questions
- Abandoning a clarification by switching topics
- Harmless Persian orthographic variants
- Non-critical paraphrases that should not cause clarification
- Overconfident ASR errors that a confidence-only gate can miss

`core` cases are regression expectations for implemented behavior. `challenge`
cases encode desired behavior that may expose limitations; they are reported
separately and do not fail `--strict`.

## Run

The deterministic control evaluation does not load ASR, TTS, or Gemma:

```bash
python -m benchmark.chatbot_multiturn.run_evaluation --strict
```

Run the same conversations through the configured local Gemma model:

```bash
python -m benchmark.chatbot_multiturn.run_evaluation \
  --with-llm \
  --output benchmark/chatbot_multiturn/results/llm_results.json
```

The LLM metric checks small groups of required Persian answer concepts. Review
the saved per-turn answers qualitatively as well; keyword matching is not a
complete semantic metric.

Console output shows the summary by default. Add `--show-rows` for all per-turn
records. `prefill_history` in a case can deliberately evict recent turns while
leaving structured session memory intact.

## Interpretation

Use the control metrics to improve routing, clarification, and memory logic.
Use the optional LLM keyword metric only after control behavior is acceptable.
Model fine-tuning or replacement should be justified by failures that remain
after the correct transcript, history, and confirmed memory reach the LLM.
