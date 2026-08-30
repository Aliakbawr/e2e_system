# Persian Speech Chatbot

Local Persian speech assistant with an end-to-end runtime pipeline:

```text
microphone -> Silero VAD -> Vosk Persian ASR -> Gemma 2 9B -> Piper Persian TTS -> audio playback
```

Research evaluations are isolated under [`benchmark/`](benchmark/README.md). The rest of this repository is the deployable chatbot application.

## Runtime layout

| Path | Responsibility |
| --- | --- |
| `main.py` | Primary local-chat entry point |
| `config/` | Environment-aware runtime settings |
| `src/asr/` | Vosk speech-recognition adapter and audio conversion |
| `src/llm/` | Gemma response generation adapter |
| `src/tts/` | Piper speech synthesis adapter |
| `src/audio/` | Microphone capture and playback |
| `src/core/` | End-to-end application orchestration |
| `scripts/` | Alternate and diagnostic entry points |
| `models/` | Local model installation locations; weights are not versioned |
| `data/` | Runtime input/output locations; generated audio is not versioned |
| `report/` | XeLaTeX sources for the bachelor final-project report |

## Setup

Use Python 3.10 or 3.11 in a virtual environment. Install PyTorch for your CUDA version first, then install the application dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the local models as described in [`models/README.md`](models/README.md), or override their paths with environment variables.

## Run

From the repository root:

```bash
python main.py
```

The microphone waits for speech automatically. After you finish speaking and the
configured silence interval passes, processing starts. Listening resumes after
the bot finishes playing its answer. Use Ctrl+C while waiting to exit.

Alternative utilities:

```bash
python -m scripts.push_to_talk_chat
python -m scripts.record_mic
```

## Configuration

The defaults live in `config/settings.py`. These environment variables can override machine-specific paths and devices:

- `PERSIAN_ASSISTANT_ASR_MODEL_PATH`
- `PERSIAN_ASSISTANT_VOSK_LOG_LEVEL`
- `PERSIAN_ASSISTANT_ASR_MAX_ALTERNATIVES` (default: `3`)
- `PERSIAN_ASSISTANT_ASR_WORD_CONFIDENCE_THRESHOLD` (default: `0.65`)
- `PERSIAN_ASSISTANT_ASR_ALTERNATIVE_SCORE_GAP` (default: `3.0`)
- `PERSIAN_ASSISTANT_ASR_CLARIFICATION_MAX_OPTIONS` (default: `2`)
- `PERSIAN_ASSISTANT_LLM_MODEL_PATH`
- `PERSIAN_ASSISTANT_TTS_MODEL_PATH`
- `PERSIAN_ASSISTANT_TTS_CONFIG_PATH`
- `PERSIAN_ASSISTANT_TTS_STREAM_SOFT_MIN_CHARS` (default: `30`)
- `PERSIAN_ASSISTANT_TTS_STREAM_MAX_CHARS` (default: `60`)
- `PERSIAN_ASSISTANT_MIC_DEVICE`
- `PERSIAN_ASSISTANT_AUDIO_PLAYER`
- `PERSIAN_ASSISTANT_VAD_MODEL_PATH`
- `PERSIAN_ASSISTANT_VAD_THRESHOLD` (default: `0.5`)
- `PERSIAN_ASSISTANT_VAD_MIN_SPEECH_MS` (default: `150`)
- `PERSIAN_ASSISTANT_VAD_MIN_SILENCE_MS` (default: `900`)
- `PERSIAN_ASSISTANT_VAD_PRE_ROLL_MS` (default: `300`)
- `PERSIAN_ASSISTANT_VAD_POST_ROLL_MS` (default: `200`)
- `PERSIAN_ASSISTANT_VAD_MAX_UTTERANCE_SEC` (default: `30`)
- `PERSIAN_ASSISTANT_MAX_LLM_TOKENS`
- `PERSIAN_ASSISTANT_MAX_LLM_INPUT_TOKENS`
- `PERSIAN_ASSISTANT_LOG_LEVEL` (default: `INFO`)
- `PERSIAN_ASSISTANT_LOG_FILE` (default: `data/logs/chatbot.log`)
- `PERSIAN_ASSISTANT_LOG_MAX_BYTES` (default: `5242880`)
- `PERSIAN_ASSISTANT_LOG_BACKUP_COUNT` (default: `3`)

The interactive chat commands keep the six most recent completed conversation
turns in memory. This enables basic follow-up questions and pronoun references.
The session is in-memory only and is cleared when the process exits.

ASR results include Vosk word confidence and up to three transcript hypotheses.
Vosk exposes those through two different decoding modes, so requesting more than
one alternative performs a second decoding pass. Set
`PERSIAN_ASSISTANT_ASR_MAX_ALTERNATIVES=1` when lower latency is preferred.

When a word is below the confidence threshold and plausible alternatives differ
at that position, the assistant asks a targeted clarification before invoking
the LLM. Orthographic variants such as `آدرس` and `ادرس` are treated as equal.
The pending choices are retained for one following turn, allowing replies such
as `شفت`, `منظورم شفت بود`, or `دومی` to reconstruct the original question.
Confirmed choices and contextual corrections are then kept in a separate,
bounded session memory even after their original dialogue turns leave the
six-turn history. This memory is supplied to the LLM with instructions not to
treat contextual corrections as global replacements, and is cleared when the
process exits or the session is cleared.

Runtime events are printed to the console and written to a rotating UTF-8 log.
Normal answers stream from Gemma into phrase-sized Piper requests and then into
audio playback. Metrics include time to the first generated token, first phrase,
and first audio, along with phrase count and total synthesis time. Clarification
and retry prompts continue to use the simpler single-utterance TTS path. To
monitor ASR confidence, alternatives, session size, stage latency, and errors:

```bash
tail -f data/logs/chatbot.log
```

Logs contain recognized speech and generated answers. Treat them as local
conversation data and remove or relocate the log when privacy requires it.

Large model files, generated WAV files, caches, and IDE metadata are deliberately excluded from version control.
