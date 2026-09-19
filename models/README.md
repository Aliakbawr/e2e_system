# Runtime models

The main Persian speech-chatbot pipeline uses four local model artifacts. Model
weights are machine-local deployment assets and are intentionally ignored by
Git.

## Models used by the main pipeline

| Stage | Model | Runtime/backend | Default location | Notes |
| --- | --- | --- | --- | --- |
| Voice activity detection | Silero VAD v6.2.1 | ONNX Runtime on CPU | `models/vad/silero/silero_vad.onnx` | Processes 512-sample frames at 16 kHz and supplies speech probabilities to the endpoint detector. |
| Speech recognition | Vosk Persian `vosk-model-fa-0.42` | Vosk/Kaldi recognizer | `models/asr/vosk/vosk-model-fa-0.42/` | Produces the primary transcript, word confidences, and optional N-best alternatives. |
| Language model | Gemma 2 9B Instruct | Transformers/PyTorch | `models/llm/gemma-2-9b-it-4bit/` | Local export based on `google/gemma-2-9b-it`; runs on CUDA when available and otherwise on CPU. |
| Speech synthesis | Piper Persian `fa_IR-gyro-medium` | Piper ONNX voice | `models/tts/piper/fa_IR-gyro-medium.onnx` | Single-speaker `gyro` voice, medium quality, with a 22,050 Hz output sample rate. |

These are deployment choices. Other checkpoints stored locally or evaluated in
`benchmark/`—including NeMo FastConformer and alternative ASR, LLM, or TTS
models—are research artifacts and are not loaded by `main.py`.

## Directory layout

Default runtime locations:

```text
models/
├── asr/vosk/vosk-model-fa-0.42/
├── llm/gemma-2-9b-it-4bit/
├── vad/silero/silero_vad.onnx
└── tts/piper/
    ├── fa_IR-gyro-medium.onnx
    └── fa_IR-gyro-medium.onnx.json
```

The model paths can be overridden with the environment variables documented in
the [root README](../README.md#configuration). Tokenizer and model configuration
files remain beside their corresponding local weights. Piper requires both its
`.onnx` voice and matching `.onnx.json` configuration file.

## Silero VAD installation

The VAD model currently used by the runtime is Silero VAD v6.2.1. To install
the same pinned model manually:

```bash
mkdir -p models/vad/silero
curl -fL https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.1/src/silero_vad/data/silero_vad.onnx \
  -o models/vad/silero/silero_vad.onnx
```
