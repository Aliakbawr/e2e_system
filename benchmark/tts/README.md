# Persian TTS benchmark

This directory contains the preserved Colab workflows used to compare three
Persian text-to-speech systems on the same 1,052-text evaluation set. The set
was formed by randomly selecting 10% of the Persian Common Voice 25 dataset:

- `Kamtera_VITS`
- `VITS_MMS_FAS`
- `PIPER_TTS_FAS`

Model labels are normalized in this README. In particular, `PIPER_TTS`,
`PIPER_TTS_FAS`, and notebook-internal Piper aliases refer to the same evaluated
Persian Piper configuration.

The comparison covers predicted speech quality, intelligibility through ASR
back-transcription, and synthesis runtime. The supplied final result tables are
the canonical summary in this directory; several notebooks retain exploratory
or intermediate output cells and machine-specific Google Drive paths.

## Notebook inventory

| Notebook | Role |
|---|---|
| `Kamtera_persian_tts_female_vits.ipynb` | Loads the Kamtera Persian female VITS checkpoint, generates audio, and contains TTS/ASR evaluation cells. |
| `nisqa_mos_measure.ipynb` | Runs NISQA scoring for Kamtera and MMS artifacts; it also retains exploratory F5-TTS cells. |
| `piper_asr_eval.ipynb` | Generates Piper audio and measures synthesis runtime. |
| `Copy of piper_asr_eval.ipynb` | A more heavily documented variant of the Piper generation/runtime notebook. |
| `piper_generations.ipynb` | Back-transcribes Piper generations with Persian NeMo FastConformer and computes WER, CER, and ASR runtime. |
| `PIPER_TTS_nisqa.ipynb` | Runs NISQA quality scoring over the Piper generations. |
| `f5tts.ipynb` | Exploratory F5-TTS generation workflow; F5-TTS is not included in the supplied final three-model comparison. |

## Evaluation protocol

The evaluation follows the same pipeline for every reported TTS model:

1. randomly select 10% of the Persian Common Voice 25 dataset, producing the
   1,052-sample evaluation subset;
2. use the Common Voice reference transcript of each selected sample as input
   to the TTS model;
3. synthesize one utterance from every reference transcript;
4. transcribe each synthesized utterance with the selected Persian NeMo
   FastConformer ASR model (`nvidia/stt_fa_fastconformer_hybrid_large`); and
5. calculate corpus WER and CER by comparing the ASR predictions with the same
   Common Voice reference transcripts used for synthesis.

The NeMo FastConformer model was selected because it produced the best WER and
CER in the project's [Persian Common Voice 25 ASR benchmark](../asr/common_voice_25/README.md).
Vosk was evaluated in that benchmark as a separate ASR system, but the
preserved TTS evaluation notebook uses NeMo FastConformer for
back-transcription.

In addition to this intelligibility pipeline:

- **Quality:** NISQA predicts MOS and the noise, discontinuity, coloration, and
  loudness dimensions for every generated utterance.
- **Intelligibility:** WER and CER measure how accurately the selected ASR model
  recovers the original Common Voice transcript from synthesized speech.
- **Runtime:** TTS RTF is synthesis time divided by generated-audio duration.
  Average latency is mean synthesis time per sample. Aggregate RTF uses total
  synthesis time divided by total generated-audio duration.
- **Speedup:** the reported approximation is `1 / aggregate RTF`.
- **ASR RTF:** recognition time divided by generated-audio duration; it measures
  the evaluator, not TTS synthesis speed.

All error-rate and RTF values are proportions rather than percentages.

## Overall results

| Model | MOS mean ↑ | MOS std ↓ | Noise ↑ | Discontinuity ↑ | Coloration ↑ | Loudness ↑ | WER ↓ | CER ↓ | Reported TTS RTF ↓ | ASR RTF ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Kamtera VITS | 3.380 | 0.611 | 3.632 | 4.077 | 3.814 | 4.098 | 0.144 | 0.068 | 0.043 | 0.034 |
| VITS MMS FAS | 3.913 | 0.683 | 4.011 | 4.048 | 3.633 | 3.887 | 0.253 | 0.114 | **0.032** | **0.026** |
| Piper TTS FAS | **4.534** | **0.362** | **4.070** | **4.656** | **4.168** | **4.282** | **0.107** | **0.045** | 0.192 | 0.059 |

Piper has the best predicted quality and the lowest ASR-based error rates.
VITS MMS is the fastest system in the supplied runtime summary, while Piper is
the slowest of the three synthesis paths.

## NISQA quality

| Model | Samples | MOS mean ↑ | MOS std ↓ | Noise ↑ | Discontinuity ↑ | Coloration ↑ | Loudness ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kamtera VITS | 1,052 | 3.380 | 0.611 | 3.632 | 4.077 | 3.814 | 4.098 |
| VITS MMS FAS | 1,052 | 3.913 | 0.683 | 4.011 | 4.048 | 3.633 | 3.887 |
| Piper TTS FAS | 1,052 | **4.534** | **0.362** | **4.070** | **4.656** | **4.168** | **4.282** |

## Intelligibility and per-sample runtime

| Model | Samples | WER ↓ | CER ↓ | TTS RTF ↓ | Avg TTS latency (s) ↓ |
|---|---:|---:|---:|---:|---:|
| Kamtera VITS | 1,052 | 0.144 | 0.068 | 0.044 | 0.175 |
| VITS MMS FAS | 1,052 | 0.253 | 0.114 | **0.032** | **0.087** |
| Piper TTS FAS | 1,052 | **0.107** | **0.045** | 0.192 | 0.434 |

## Aggregate runtime

| Model | Samples | Aggregate RTF ↓ | Avg TTS latency (s) ↓ | Speedup vs real time ↑ |
|---|---:|---:|---:|---:|
| Kamtera VITS | 1,052 | 0.043 | 0.175 | 23.27× |
| VITS MMS FAS | 1,052 | **0.026** | **0.087** | **39.03×** |
| Piper TTS FAS | 1,052 | 0.192 | 0.434 | 5.20× |

The supplied summaries contain two differently labeled VITS MMS runtime
values: `0.032` is reported as TTS RTF in the overall and per-sample tables,
while `0.026` is reported as aggregate RTF in the dedicated runtime table and
as ASR RTF in the overall table. Both are retained here rather than silently
merged. The original exported summary should be checked before treating either
value as the single canonical VITS MMS aggregate statistic. The small Kamtera
difference (`0.043` aggregate versus `0.044` in the per-sample table) is also
preserved as reported.

## Reproduction notes

The notebooks were developed in Colab and refer to files under `/content` and
mounted Google Drive directories. Before rerunning them:

1. reproduce or load the fixed random 10% Common Voice 25 subset and verify that
   it contains the expected 1,052 reference transcripts;
2. update the dataset, model, output, and metadata paths for the active
   environment;
3. generate all 1,052 utterances for one model and retain per-sample synthesis
   time and audio duration;
4. run NISQA over exactly those generated files;
5. back-transcribe the same files with
   `nvidia/stt_fa_fastconformer_hybrid_large`;
6. compare the predicted transcripts with the corresponding Common Voice
   reference transcripts to compute corpus WER and CER;
7. compute aggregate runtime from the complete set; and
8. export the per-sample results and one-row model summary before comparing
   models.

Generated WAV files and several CSV paths referenced by the notebooks are not
stored in this directory. Reruns therefore require the source text CSV, model
weights, and the corresponding external artifacts.

## Interpretation limits

- NISQA MOS is a non-intrusive model prediction, not a human listening-test MOS.
- WER and CER are intelligibility proxies and depend on the chosen ASR model and
  text normalization.
- A rerun is exactly comparable only when it reuses the same sampled Common
  Voice rows. The sampling seed or a frozen list of sample identifiers should
  accompany future exported results.
- Runtime values are hardware- and environment-dependent and should only be
  compared within the same recorded protocol.
- Notebook output cells may reflect exploratory runs; use the tables above as
  the supplied final comparison unless newer frozen exports are added.
