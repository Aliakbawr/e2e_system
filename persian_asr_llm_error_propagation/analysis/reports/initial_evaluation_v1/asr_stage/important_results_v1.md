# Persian FLEURS ASR Error-Propagation Results — Frozen ASR Stage V1

Status: **Frozen**  
Freeze date: **2026-08-25**  
Language/test split: **FLEURS `fa_ir` test**

This report consolidates the important results from the frozen ASR stage and
the pre-LLM answer-span analysis. The ASR model and QA set must not be tuned or
altered in response to these findings.

## 1. Dataset and QA coverage

- FLEURS recordings: **871**
- Unique semantic/text IDs: **324**
- Unique reference transcripts: **324**
- Total audio duration: **3.700517 hours**
- Frozen QA candidates: **324**
- Frozen usable QA items: **286**
- Rejected QA items: **38**
- Matched recording/QA rows: **769**
- Matched semantic IDs: **286**
- Gold answer found in normalized reference: **100%**

QA inclusion was determined only by the already-frozen `usable` field. No QA
question or gold answer was generated, edited, selected, or rejected after
observing ASR performance.

## 2. Frozen ASR configuration

- Model: `nvidia/stt_fa_fastconformer_hybrid_large`
- Local archive: `stt_fa_fastconformer_hybrid_large.nemo`
- NeMo model class: `EncDecHybridRNNTCTCBPEModel`
- Tokenizer: SentencePiece BPE, **1,024 tokens**
- Input sampling rate: **16 kHz**
- Acoustic features: **80-bin mel spectrogram**
- Window: **25 ms Hann**
- Window stride: **10 ms**
- Feature normalization: `per_feature`
- Decoder used by the existing model: RNNT `greedy_batch`
- Maximum symbols per step: **10**
- Device during benchmark: **CUDA GPU**
- Successful transcriptions: **871/871**
- Failed or empty transcriptions: **0**

The runner preserved the existing inference call:
`model.transcribe([audio_path])`, followed by extraction of the hypothesis
`.text`. Raw ASR text was retained separately from deterministic evaluation
normalization.

## 3. Corpus ASR performance

These are corpus-level Icefall/Kaldi-style rates: errors are pooled across the
dataset before division. CER excludes spaces.

| Metric | Result |
|---|---:|
| Corpus WER | **0.448590 (44.86%)** |
| Corpus CER | **0.265781 (26.58%)** |
| Word substitutions | 4,181 |
| Word deletions | 5,345 |
| Word insertions | 116 |
| Reference words | 21,494 |
| Character substitutions | 2,439 |
| Character deletions | 18,859 |
| Character insertions | 495 |
| Reference characters, excluding spaces | 81,996 |

The earlier **0.435013 WER** was the unweighted mean of per-recording WER and
must not be presented as the corpus WER. The primary overall WER is **44.86%**.

### Runtime

| Runtime measure | Result |
|---|---:|
| Mean latency per recording | 0.075665 s |
| Median latency | 0.072069 s |
| 95th-percentile latency | 0.095182 s |
| Mean RTF | 0.005270 |
| Aggregate RTF | 0.004947 |
| Total measured inference latency | 65.904198 s |

## 4. Exact answer-span preservation

Primary definition:

> `answer_preserved = True` when the normalized gold-answer string is an exact
> substring of the normalized ASR transcript.

This is an **exact lexical answer-span preservation measure**. It is not a
semantic-preservation measure and is not downstream QA accuracy.

| Measure | Result |
|---|---:|
| Exact spans preserved | **118/769** |
| Exact span preservation rate | **15.34%** |
| Exact spans lost | 651/769 |
| Cluster-bootstrap 95% CI | **11.81%–19.24%** |
| Naive Wilson 95% CI, ignores clustering | 12.97%–18.06% |
| Semantic IDs preserved in at least one recording | 63/286 |
| Semantic IDs preserved in every recording | 26/286 |

The cluster interval used **10,000 bootstrap resamples**, seed **42**, with
whole semantic IDs resampled so repeated recordings remained together.

### Substring sensitivity check

A stricter contiguous-token-span check preserved **114/769** rather than
118/769. Only **4 rows** disagreed. The protocol-defined `answer_preserved`
substring variable remains the primary result; the token-span result is a
sensitivity analysis.

## 5. Relationship between WER and answer-span survival

| Group | Recordings | Mean WER | Median WER | Mean CER | Median CER |
|---|---:|---:|---:|---:|---:|
| Answer span lost | 651 | 0.4644 | 0.4706 | 0.2759 | 0.2603 |
| Answer span preserved | 118 | 0.3199 | 0.3267 | 0.1608 | 0.1381 |

- ROC AUC when WER predicts answer-span loss: **0.7444**
- Pearson correlation between WER and preservation: **−0.3294**
- Low-WER (`WER <= 0.20`) critical answer losses: **29**
- High-WER (`WER >= 0.50`) answer survivals: **19**

WER therefore has meaningful predictive value, but it does not determine
whether task-relevant information survives.

### Preservation by WER band

| WER band | Recordings | Preserved | Preservation rate |
|---|---:|---:|---:|
| <=0.10 | 14 | 12 | 85.71% |
| 0.10–0.20 | 44 | 17 | 38.64% |
| 0.20–0.30 | 95 | 26 | 27.37% |
| 0.30–0.40 | 141 | 25 | 17.73% |
| 0.40–0.50 | 213 | 23 | 10.80% |
| 0.50–0.60 | 155 | 14 | 9.03% |
| 0.60–0.70 | 76 | 0 | 0.00% |
| >0.70 | 31 | 1 | 3.23% |

## 6. Gold-answer length as a confounder

Longer exact strings are mechanically easier for ASR to damage. Preservation
declined strongly with normalized gold-answer length.

| Gold-answer length | Recordings | Semantic IDs | Preserved | Rate | Cluster-bootstrap 95% CI | Mean WER |
|---|---:|---:|---:|---:|---:|---:|
| 1 token | 111 | 40 | 32 | **28.83%** | 15.74%–42.61% | 0.4113 |
| 2 tokens | 172 | 66 | 41 | **23.84%** | 15.34%–33.13% | 0.4337 |
| 3–4 tokens | 252 | 94 | 33 | **13.10%** | 7.56%–19.22% | 0.4560 |
| 5+ tokens | 234 | 86 | 12 | **5.13%** | 1.69%–9.44% | 0.4484 |

Gold-answer length must therefore be reported or controlled when interpreting
the relationship between WER and critical-information loss.

## 7. Key qualitative examples

### High WER, answer preserved — ID 1735

- WER: **54.55%**
- Question: `گونه‌های جدید تکامل‌یافته در چند نسل گزارش شدند؟`
- Gold answer: `دو نسل`
- ASR transcript still contains: `دو نسل`
- `answer_preserved = True`

The transcript is heavily corrupted overall, but the information required for
this question survives exactly.

### Low WER, critical answer destroyed — ID 1966

- WER: **6.67%**
- Question: `این کمیسیون در پاسخ به چه چیزی ایجاد شد؟`
- Gold answer: `اعتراضات گسترده ضدحکومتی`
- ASR retained only: `اعتراضات گسترده ضد`
- `answer_preserved = False`

The transcript has low overall error, but the missing part directly destroys
the exact information required by the question.

Together these examples demonstrate:

> **WER is useful but does not determine whether task-relevant information
> survives.**

## 8. Interpretation boundary and next experiment

The 15.34% result means that the exact normalized gold string survived in
15.34% of matched recordings. It does **not** imply that semantic information
survived in only 15.34%. An LLM may recover a correct answer from paraphrased,
inflected, or partially corrupted evidence even when the exact gold span is
absent.

The downstream LLM experiment should make exactly two calls per row:

1. **Oracle condition:** context = `reference_normalized`
2. **ASR condition:** context = `asr_normalized`

Both conditions must use the same frozen question, prompt, decoding settings,
and evaluation procedure. Future confidence intervals and condition contrasts
should resample by semantic `id`, not by independent recording.

## 9. Important artifact locations

- ASR checkpoint: `asr/fleurs_fa_ir_asr_checkpoint.csv`
- Corpus metrics: `asr/fleurs_fa_ir_asr_metrics.json`
- Frozen QA: `qa/fleurs_fa_ir_generated_qa_candidates_v1.csv`
- Downstream table: `analysis/inputs/fleurs_asr_qa_eval_input_v1.csv`
- Answer-preservation summary: `analysis/results/answer_span/original_asr_v1/answer_preservation_summary_v1.json`
- Cluster-aware span analysis: `analysis/results/answer_span/original_asr_v1/exact_answer_span_analysis_v1.json`
- Low-WER answer losses: `analysis/results/answer_span/original_asr_v1/critical_low_wer_answer_loss_v1.csv`
- High-WER answer survivals: `analysis/results/answer_span/original_asr_v1/robust_high_wer_answer_survival_v1.csv`
- Historical manifest: `analysis/manifests/historical_pre_restructure/initial_evaluation_v1/FROZEN_ASR_STAGE_V1.sha256`

## 10. Frozen hashes

| Artifact | SHA-256 |
|---|---|
| NeMo model | `03fd08c2a5b40e0b18dc899c080f3ace6c7193ab699c0d6cc61c672c935d49ee` |
| FLEURS TSV | `2f96b4df444a63964e689db45ce9445c9759b41578d010a5f758dbd29b004579` |
| FLEURS audio archive | `f787ae225da693ed28c734c18aac1932c0f271473c3fb49a98db5e9b222fa11d` |
| ASR runner | `6c39ea07536ef09b4d52bcf90128a28eeab69e5308f264ef0275bf090e42b6d0` |
| ASR checkpoint | `b41211ac8614bfd083777b8d82a272269251ce8865b11b2c7c44e8dd314c1c00` |
| Frozen QA | `cc83e1a2fd6a9c8e291b02d6f0d08f57f026b2d67e6f309ea6db15fbe3b74635` |
| Downstream table | `264440b8dcffa3c2fa9c2b48b157ab5291aa39f1b8d7f170bce4e69b339537d3` |
| Cluster-aware analysis JSON | `b0407ec59901e63bd30f2e0ae90bc800ece11ea01285ed69978e7b9ac4c5f459` |
