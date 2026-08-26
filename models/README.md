# Local models

Model weights are machine-local deployment artifacts and are intentionally ignored by Git.

Default locations:

```text
models/
├── asr/vosk/vosk-model-fa-0.42/
├── llm/gemma-2-9b-it-4bit/
└── tts/piper/
    ├── fa_IR-gyro-medium.onnx
    └── fa_IR-gyro-medium.onnx.json
```

The model paths can be overridden with the environment variables documented in the root README. Tokenizer and model configuration files may remain beside their corresponding local weights.
