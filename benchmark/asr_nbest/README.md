# Vosk N-best evidence

This benchmark freezes the top three Vosk decoder hypotheses for the same 769
enhanced-audio recordings. Semantic IDs are assigned deterministically to a 70%
development split and a 30% held-out split before policy evaluation, preventing
recordings of the same sentence from leaking across partitions.

```bash
/home/aliakbar/miniconda3/envs/persian_assistant/bin/python \
  -m benchmark.asr_nbest.build_evidence
```

The append-only checkpoint is resumable. Oracle N-best WER and answer-span
coverage measure the maximum available headroom; they are diagnostic upper
bounds and must never be used as a runtime selection rule.

The completed split contains 192 development semantic IDs (517 recordings) and
94 held-out IDs (252 recordings). On held-out data, oracle N-best mean recording
WER is 11.00% versus 12.29% for the primary, and 16 missing answer spans are
recoverable. However, blindly choosing the top decoder hypothesis worsens WER
to 14.04%. Runtime therefore selects an alternative only with matching explicit
correction memory; otherwise critical disagreement causes clarification.
