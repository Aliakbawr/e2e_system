# Constrained N-best ASR enhancement

## Frozen decoder evidence

All 769 enhanced-audio recordings completed top-three Vosk decoding. The split
was assigned by semantic ID before policy analysis: 192 development IDs/517
recordings and 94 held-out IDs/252 recordings.

| Metric | Development | Held out | All |
|---|---:|---:|---:|
| Primary mean recording WER | 12.77% | 12.29% | 12.61% |
| Oracle N-best mean recording WER | 11.24% | 11.00% | 11.16% |
| Top-decoder mean recording WER | 14.98% | 14.04% | 14.67% |
| Primary answer-span rate | 67.70% | 63.49% | 66.32% |
| Oracle N-best answer-span rate | 74.08% | 69.84% | 72.69% |
| Recoverable answer spans | 33 | 16 | 49 |

Oracle N-best improves on the primary in 207 recordings, including 65 held-out
recordings. This is genuine headroom, but it cannot be used directly because it
requires the reference. Decoder score alone is unsafe: the top decoder choice
is better in only 5 recordings and worse in 342.

## Runtime policy

- Alternatives outside the configured decoder-score gap are ignored.
- A hypothesis is selected automatically only when an explicit user-confirmed
  correction matches both the primary wording and current context.
- Names after entity cues, numbers, and negation disagreements trigger targeted
  clarification even when primary word confidence is high.
- Known harmless politeness variation does not trigger clarification.
- The policy never generates a new transcript; it keeps the primary or uses an
  actual Vosk hypothesis.

## Noisy multi-turn result

| Metric | Stage 01 | Stage 02 |
|---|---:|---:|
| Core decision accuracy | 100% | 100% |
| Challenge decision accuracy | 0% | 100% |

Both challenge turns improved: harmless `لطفا/خواهشا` variation no longer asks
an unnecessary question, while high-confidence `شریعت/شریف` disagreement after
`دانشگاه` now asks for clarification. No evaluated turn regressed.

The local Gemma replay also passes: core keyword accuracy is 100% on 7 answer
turns, and challenge keyword accuracy is 100% on the newly routed answer turn.
