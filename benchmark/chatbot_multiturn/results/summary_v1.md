# Noisy multi-turn evaluation summary — V1

Date: 2026-08-27

## Core regression results

The deterministic control path passed all 12 core turns:

| Metric | Result |
|---|---:|
| Decision accuracy | 100% |
| Clarification precision | 100% |
| Clarification recall | 100% |
| Resolved-query accuracy | 100% |
| Correction-memory accuracy | 100% |

Gemma 2 9B passed all seven answer-concept checks after confirmed memory was
placed near the current question and critical corrections received a scoped
read-back instruction.

## Challenge results

Both deliberately unsupported challenge turns failed their desired routing
decision:

1. A low-confidence politeness-word paraphrase (`لطفا` / `خواهشا`) caused an
   unnecessary clarification even though task-relevant information survived.
2. A high-confidence proper-name disagreement (`شریعت` / `شریف`) did not cause
   clarification because the primary word confidence exceeded the threshold.

These failures isolate the next limitation to task-relevance and confidence
calibration in the dialogue-control gate.

## Model-change decision

This V1 suite does not justify ASR-correction fine-tuning or replacing Gemma.
The observed core LLM failures were corrected through prompt placement and a
turn-scoped instruction, after which all core answer checks passed. The next
engineering experiment should improve semantic importance detection and allow
strong N-best disagreement on names, numbers, and negation to override a high
one-best word confidence.

## Limitations

The suite is small and hand-authored. It is a regression and diagnostic suite,
not a population-level performance estimate. Future versions should add
anonymized failures sampled from runtime logs and replayed real audio before a
fine-tuning decision is reconsidered.
