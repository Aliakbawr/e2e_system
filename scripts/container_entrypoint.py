"""Validate mounted runtime assets before starting the containerized app."""

from __future__ import annotations

import os
import sys
from pathlib import Path


DEFAULT_ASSETS = {
    "VAD model": (
        "PERSIAN_ASSISTANT_VAD_MODEL_PATH",
        "/app/models/vad/silero/silero_vad.onnx",
    ),
    "ASR model": (
        "PERSIAN_ASSISTANT_ASR_MODEL_PATH",
        "/app/models/asr/vosk/vosk-model-fa-0.42",
    ),
    "LLM model": (
        "PERSIAN_ASSISTANT_LLM_MODEL_PATH",
        "/app/models/llm/gemma-2-9b-it-4bit",
    ),
    "TTS model": (
        "PERSIAN_ASSISTANT_TTS_MODEL_PATH",
        "/app/models/tts/piper/fa_IR-gyro-medium.onnx",
    ),
    "TTS config": (
        "PERSIAN_ASSISTANT_TTS_CONFIG_PATH",
        "/app/models/tts/piper/fa_IR-gyro-medium.onnx.json",
    ),
}


def missing_assets() -> list[tuple[str, Path]]:
    """Return required model paths that do not exist."""
    missing = []
    for label, (environment_name, default) in DEFAULT_ASSETS.items():
        path = Path(os.getenv(environment_name, default)).expanduser()
        if not path.exists():
            missing.append((label, path))
    return missing


def main() -> int:
    command = sys.argv[1:] or ["python", "main.py"]
    starts_app = command in (["python", "main.py"], ["python3", "main.py"])

    if starts_app:
        missing = missing_assets()
        if missing:
            print("Container startup failed: required model assets are missing:", file=sys.stderr)
            for label, path in missing:
                print(f"  - {label}: {path}", file=sys.stderr)
            print(
                "Mount a complete models directory at /app/models; "
                "see models/README.md.",
                file=sys.stderr,
            )
            return 2

    os.execvp(command[0], command)
    return 127


if __name__ == "__main__":
    raise SystemExit(main())

