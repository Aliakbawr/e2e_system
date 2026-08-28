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

## Contextual recovery benchmark

Reference labels were used offline to identify in-gap, single-replacement
alternatives that are better than the primary. Runtime received only the actual
N-best evidence and a simulated prior explicit correction. Each positive case
also had no-memory and irrelevant-context controls.

| Metric | Development | Held out | All |
|---|---:|---:|---:|
| Positive audio cases | 51 | 27 | 78 |
| Target recovery rate | 72.55% | 81.48% | 75.64% |
| Conditional WER before | 15.92% | 13.01% | 14.91% |
| Conditional WER after | 11.88% | 8.39% | 10.67% |
| Answer-span rate before | 54.90% | 62.96% | 57.69% |
| Answer-span rate after | 68.63% | 77.78% | 71.79% |
| Wrong alternatives selected | 0 | 0 | 0 |
| No-memory control changes | 0 | 0 | 0 |
| Irrelevant-context control changes | 0 | 0 | 0 |

The held-out result covers 27 recordings from 22 semantic IDs. It demonstrates
recovery after an explicit correction, not unconditional ASR improvement across
all speech. Naturally recorded multi-turn audio remains necessary for external
validation.

## Contextual ASR-to-Gemma propagation

Only the 59 transcripts recovered by explicit correction memory were changed;
the other 710 frozen audio-enhanced Gemma predictions were reused exactly.

| Metric | Audio-enhanced baseline | Contextual recovery | Change |
|---|---:|---:|---:|
| Exact match, all 769 | 39.92% | 41.09% | +1.17 pp |
| Mean F1, all 769 | 0.7040 | 0.7075 | +0.0035 |
| Exact match, held-out 252 | 36.90% | 38.49% | +1.59 pp |
| Mean F1, held-out 252 | 0.6860 | 0.6922 | +0.0062 |
| Exact match, 59 changed contexts | 38.98% | 54.24% | +15.25 pp |
| Mean F1, 59 changed contexts | 0.7607 | 0.8058 | +0.0451 |

Across all 769 recordings, exact match gained 9 cases and lost none. A
semantic-ID clustered bootstrap gives a 95% interval of +0.40 to +2.08
percentage points for exact-match change and +0.0006 to +0.0071 for mean F1.
The held-out estimate is promising but less conclusive: only 22 held-out
contexts changed, so its exact-match interval starts at zero and its F1
interval crosses zero.

These results demonstrate downstream improvement for the semi-synthetic
explicit-correction condition. They do not establish the same effect for
naturally occurring conversations, which remains the next validation step.
