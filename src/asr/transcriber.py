"""Vosk-based Persian speech recognition for the chatbot runtime."""

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from config.settings import ASR_LOG_LEVEL, ASR_MODEL_PATH, SAMPLE_RATE


@lru_cache(maxsize=1)
def _load_model():
    """Load the local Vosk model once, on the first transcription request."""
    model_path = Path(ASR_MODEL_PATH)

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Vosk model directory not found: {model_path}. "
            "Extract vosk-model-fa-0.42.zip into models/asr/vosk/."
        )

    try:
        from vosk import Model, SetLogLevel
    except ImportError as exc:
        raise RuntimeError(
            "The 'vosk' Python package is not installed. "
            "Install application dependencies with: pip install -r requirements.txt"
        ) from exc

    SetLogLevel(ASR_LOG_LEVEL)
    return Model(str(model_path))


def _read_pcm16(audio_path: str) -> bytes:
    """Read an audio file and return 16 kHz mono signed PCM16 bytes."""
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, source_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True,
    )

    if audio.size == 0:
        return b""

    audio = audio.mean(axis=1)

    if source_rate != SAMPLE_RATE:
        divisor = math.gcd(int(source_rate), SAMPLE_RATE)
        audio = resample_poly(
            audio,
            SAMPLE_RATE // divisor,
            int(source_rate) // divisor,
        )

    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    audio = np.clip(audio, -1.0, 1.0)

    return (audio * 32767.0).astype("<i2").tobytes()


def _result_text(payload: str) -> str:
    """Extract recognized text from a Vosk JSON result."""
    return str(json.loads(payload).get("text", "")).strip()


def transcribe_audio(audio_path: str) -> str:
    """Transcribe one audio file with the local Persian Vosk model."""
    from vosk import KaldiRecognizer

    pcm = _read_pcm16(audio_path)
    if not pcm:
        return ""

    recognizer = KaldiRecognizer(_load_model(), SAMPLE_RATE)
    recognizer.SetWords(False)

    segments = []
    chunk_size = 8000

    for offset in range(0, len(pcm), chunk_size):
        chunk = pcm[offset : offset + chunk_size]
        if recognizer.AcceptWaveform(chunk):
            text = _result_text(recognizer.Result())
            if text:
                segments.append(text)

    final_text = _result_text(recognizer.FinalResult())
    if final_text:
        segments.append(final_text)

    return " ".join(segments).strip()
