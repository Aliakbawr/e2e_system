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

Evaluate explicit-correction recovery and its negative controls with:

```bash
/home/aliakbar/miniconda3/envs/persian_assistant/bin/python \
  -m benchmark.asr_nbest.evaluate_contextual_recovery
```

This is a semi-synthetic conversational benchmark: the speech and hypotheses
are real frozen audio outputs, while the preceding explicit correction is
constructed offline from the reference-labeled better alternative. It measures
the implemented recovery mechanism but does not replace evaluation on naturally
recorded multi-turn conversations.

Prepare and analyze the downstream Gemma comparison with:

```bash
/home/aliakbar/miniconda3/envs/persian_assistant/bin/python \
  -m benchmark.asr_nbest.prepare_contextual_downstream
/home/aliakbar/miniconda3/envs/persian_assistant/bin/python \
  -m benchmark.asr_nbest.analyze_contextual_downstream
```

In the complete 769-recording comparison, contextual recovery raises exact
match from 39.92% to 41.09% and mean F1 from 0.7040 to 0.7075. There are 9
exact-match gains and no losses. These figures apply to the semi-synthetic
explicit-correction setup described above; naturally recorded multi-turn audio
is still required for external validation.
