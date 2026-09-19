# Persian Speech Chatbot

Local Persian speech assistant with an end-to-end runtime pipeline:

```text
microphone -> Silero VAD -> Vosk Persian ASR -> Gemma 2 9B -> Piper Persian TTS -> audio playback
```

Research evaluations are isolated under [`benchmark/`](benchmark/README.md). The rest of this repository is the deployable chatbot application.

The standalone Persian TTS quality, intelligibility, and runtime comparison is
documented in [`benchmark/tts/README.md`](benchmark/tts/README.md).

## Main pipeline

The deployable application is an offline-first, component-based speech
pipeline. The main runtime path is:

```text
Microphone (16 kHz mono)
    |
    v
Silero VAD + endpoint detection
    |
    v
ASR audio preparation (mono/resample + optional low-level gain)
    |
    v
Vosk Persian ASR (text, word confidence, and N-best hypotheses)
    |
    v
Persian text normalization + memory-aware N-best reranking
    |
    v
Dialogue risk decision
    |---------------- retry/clarify ----------------> Piper TTS -> playback
    |
    `---------------- answer -----------------------> Gemma 2 9B
                                                         |
                                                         v
                                              streamed phrase buffer
                                                         |
                                                         v
                                              Piper TTS -> playback + WAV
```

For each normal turn, `scripts/run_chat.py` records one utterance and passes its
path to `src/core/pipeline.py`. The pipeline performs the following operations:

1. **Capture and endpointing:** `src/audio/microphone.py` reads 16 kHz mono
   audio. Silero VAD detects speech and stops after the configured silence,
   while retaining pre-roll and post-roll audio.
2. **Audio preparation:** `src/asr/audio.py` converts the recording to mono
   16 kHz PCM16. The default `low_level_gain` profile only amplifies recordings
   below its configured RMS threshold.
3. **Speech recognition:** `src/asr/transcriber.py` runs Vosk and returns the
   primary Persian transcript, word-level confidence, and up to three N-best
   hypotheses.
4. **Dialogue control:** `src/asr/text.py`, `src/asr/rerank.py`, and
   `src/asr/risk.py` normalize the transcript, apply confirmed session context,
   and decide whether to answer, request a retry, or ask a targeted
   clarification.
5. **Response generation:** `src/llm/generator.py` prompts Gemma with the recent
   conversation and confirmed session memory. Generation is deterministic and
   streamed as text fragments.
6. **Speech synthesis and playback:** `src/tts/text_stream.py` groups streamed
   text into speakable phrases. `src/tts/synthesizer.py` normalizes Persian
   text, verbalizes numbers, synthesizes it with Piper, plays audio as it
   arrives, and saves the combined response to `data/audio_out/response.wav`.
7. **Observability:** stage latency, ASR confidence and alternatives, dialogue
   decisions, time to first token/audio, and TTS runtime are written to the
   console and rotating application log.

`ChatSession` retains at most six completed turns and twelve explicitly
confirmed memory items. All session state remains in memory and is discarded
when the process exits.

## Runtime models

These are the models used by the main application, as opposed to additional
models evaluated under `benchmark/`:

| Stage | Runtime model | Local artifact | Purpose |
| --- | --- | --- | --- |
| VAD | Silero VAD v6.2.1 | `models/vad/silero/silero_vad.onnx` | Frame-level speech probability and automatic utterance endpointing |
| ASR | Vosk Persian `vosk-model-fa-0.42` | `models/asr/vosk/vosk-model-fa-0.42/` | Offline Persian transcription, word confidence, and N-best hypotheses |
| LLM | Gemma 2 9B Instruct, local `gemma-2-9b-it-4bit` export | `models/llm/gemma-2-9b-it-4bit/` | Persian response generation with recent-turn and confirmed-memory context |
| TTS | Piper Persian `fa_IR-gyro-medium` | `models/tts/piper/fa_IR-gyro-medium.onnx` | Single-speaker Persian synthesis at 22,050 Hz |

Model paths are centralized in `config/settings.py` and can be overridden by
environment variables. See [`models/README.md`](models/README.md) for the model
directory contract and installation notes. Benchmark-only models such as NeMo
FastConformer are not loaded by the main runtime.

## Project structure

```text
.
├── main.py                       Main hands-free chatbot entry point
├── config/
│   └── settings.py               Paths, thresholds, limits, and environment overrides
├── src/
│   ├── audio/                    Microphone input, Silero VAD, and playback
│   ├── asr/                      Audio preparation, Vosk decoding, N-best, and risk logic
│   ├── core/                     End-to-end pipeline, dialogue decisions, and session state
│   ├── llm/                      Gemma prompt construction and streaming generation
│   ├── tts/                      Persian normalization, phrase buffering, and Piper synthesis
│   └── utils/                    Shared audio and logging utilities
├── scripts/
│   ├── run_chat.py               Hands-free conversation loop used by main.py
│   ├── push_to_talk_chat.py      Press-Enter recording alternative
│   └── record_mic.py             Microphone diagnostic utility
├── models/                       Machine-local runtime model artifacts
├── data/
│   ├── audio_in/                 Captured user utterances
│   ├── audio_out/                Synthesized chatbot responses
│   └── logs/                     Rotating runtime logs
├── benchmark/                    ASR, LLM, TTS, and end-to-end evaluations
├── report/                       XeLaTeX final-project sources
└── requirements.txt              Python runtime dependencies
```

## Setup

### Docker (Linux and WSL)

The image contains the application and Python/system dependencies. Model weights
are deliberately not copied into it: keep the runtime artifacts in the
documented `models/` layout and Compose mounts that directory read-only. Runtime
audio and logs are persisted in `data/`.

Copy the deployment defaults once and adjust the host IDs or model location if
needed:

```bash
cp .env.example .env
```

For native Linux with CPU inference:

```bash
docker compose -f compose.yaml -f compose.linux.yaml build
docker compose -f compose.yaml -f compose.linux.yaml run --rm assistant
```

For WSL 2 with WSLg audio and CPU inference, run these commands from the WSL
shell (Docker Desktop must have WSL integration enabled):

```bash
docker compose -f compose.yaml -f compose.wsl.yaml build
docker compose -f compose.yaml -f compose.wsl.yaml run --rm assistant
```

The current four-bit Gemma model is intended for an NVIDIA GPU. Add the GPU
overlay to either command after installing a recent NVIDIA driver and the
NVIDIA Container Toolkit (native Linux), or enabling GPU support in Docker
Desktop (WSL). For example, on WSL:

```bash
docker compose -f compose.yaml -f compose.wsl.yaml -f compose.gpu.yaml build
docker compose -f compose.yaml -f compose.wsl.yaml -f compose.gpu.yaml run --rm assistant
```

The GPU image defaults to PyTorch's CUDA 12.8 wheel index. Set
`PYTORCH_INDEX_URL` in `.env` when the target driver requires another supported
PyTorch CUDA build. The CPU image remains useful for tests and can run inference
when the installed model/backend supports CPU execution, but Gemma 2 9B will be
slow without a GPU.

Before starting the full model pipeline, check the model mounts, Python
packages, CUDA visibility, and audio devices:

```bash
# Use compose.linux.yaml here on native Linux.
docker compose -f compose.yaml -f compose.wsl.yaml run --rm assistant \
  python -m scripts.docker_check
```

Useful diagnostics:

```bash
# Verify that Docker can see the NVIDIA GPU.
docker compose -f compose.yaml -f compose.wsl.yaml -f compose.gpu.yaml run --rm \
  assistant python -c "import torch; print(torch.cuda.is_available())"

# List the audio devices visible inside the container.
docker compose -f compose.yaml -f compose.wsl.yaml run --rm \
  assistant python -c "import sounddevice as sd; print(sd.query_devices())"
```

On a Linux host whose user/group IDs differ from `1000`, set `HOST_UID` and
`HOST_GID` in `.env`; set `AUDIO_GID` to the numeric host `audio` group ID. To
keep models elsewhere, set `MODEL_DIR` to an absolute host path. Any application
setting can be passed for one run with Compose's `-e NAME=value` option.

To build and run the isolated unit-test image:

```bash
docker build --target test -t persian-speech-chatbot:test .
docker run --rm persian-speech-chatbot:test
```

To transfer a prebuilt image instead of rebuilding it on the destination, save
the CPU or GPU variant after building it:

```bash
docker save -o persian-speech-chatbot.tar persian-speech-chatbot:local
```

Copy that tar file, the Compose files, `.env`, and the required `models/`
artifacts to the other machine. Then import and run it there:

```bash
docker load -i persian-speech-chatbot.tar
# Select compose.linux.yaml or compose.wsl.yaml, and add compose.gpu.yaml
# only when the saved image is the GPU variant.
docker compose -f compose.yaml -f compose.wsl.yaml run --rm assistant
```

### Local Python

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

Reproducible development-grid tuning and held-out evaluation for the ASR audio
and uncertainty parameters is documented in
[`benchmark/asr_parameter_selection/README.md`](benchmark/asr_parameter_selection/README.md).

## Configuration

The defaults live in `config/settings.py`. These environment variables can override machine-specific paths and devices:

- `PERSIAN_ASSISTANT_ASR_MODEL_PATH`
- `PERSIAN_ASSISTANT_VOSK_LOG_LEVEL`
- `PERSIAN_ASSISTANT_ASR_MAX_ALTERNATIVES` (default: `3`)
- `PERSIAN_ASSISTANT_ASR_WORD_CONFIDENCE_THRESHOLD` (default: `0.65`)
- `PERSIAN_ASSISTANT_ASR_ALTERNATIVE_SCORE_GAP` (default: `1.0`)
- `PERSIAN_ASSISTANT_ASR_CLARIFICATION_MAX_OPTIONS` (default: `2`)
- `PERSIAN_ASSISTANT_ASR_AUDIO_PREPROCESSING` (default: `low_level_gain`)
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
