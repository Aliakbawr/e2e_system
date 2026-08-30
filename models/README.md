# Local models

Model weights are machine-local deployment artifacts and are intentionally ignored by Git.

Default locations:

```text
models/
├── asr/vosk/vosk-model-fa-0.42/
├── llm/gemma-2-9b-it-4bit/
├── vad/silero/silero_vad.onnx
└── tts/piper/
    ├── fa_IR-gyro-medium.onnx
    └── fa_IR-gyro-medium.onnx.json
```

The model paths can be overridden with the environment variables documented in the root README. Tokenizer and model configuration files may remain beside their corresponding local weights.

The VAD model currently used by the runtime is Silero VAD v6.2.1. To install
the same pinned model manually:

```bash
mkdir -p models/vad/silero
curl -fL https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.1/src/silero_vad/data/silero_vad.onnx \
  -o models/vad/silero/silero_vad.onnx
```
