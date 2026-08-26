# Persian Speech Chatbot

Local Persian speech assistant with an end-to-end runtime pipeline:

```text
microphone/WAV -> Vosk Persian ASR -> Gemma 2 9B -> Piper Persian TTS -> audio playback
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

Press Enter to begin recording and Enter again to stop. Use Ctrl+C to exit.

Alternative utilities:

```bash
python -m scripts.push_to_talk_chat
python -m scripts.record_mic
```

## Configuration

The defaults live in `config/settings.py`. These environment variables can override machine-specific paths and devices:

- `PERSIAN_ASSISTANT_ASR_MODEL_PATH`
- `PERSIAN_ASSISTANT_VOSK_LOG_LEVEL`
- `PERSIAN_ASSISTANT_LLM_MODEL_PATH`
- `PERSIAN_ASSISTANT_TTS_MODEL_PATH`
- `PERSIAN_ASSISTANT_TTS_CONFIG_PATH`
- `PERSIAN_ASSISTANT_MIC_DEVICE`
- `PERSIAN_ASSISTANT_AUDIO_PLAYER`
- `PERSIAN_ASSISTANT_MAX_LLM_TOKENS`
- `PERSIAN_ASSISTANT_MAX_LLM_INPUT_TOKENS`

The interactive chat commands keep the six most recent completed conversation
turns in memory. This enables basic follow-up questions and pronoun references.
The session is in-memory only and is cleared when the process exits.

Large model files, generated WAV files, caches, and IDE metadata are deliberately excluded from version control.
