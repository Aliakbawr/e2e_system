"""Hands-free microphone capture using streaming voice activity detection."""

import os
import warnings
from pathlib import Path

import sounddevice as sd
import soundfile as sf

from config.settings import (
    AUDIO_INPUT_DIR,
    MIC_DEVICE,
    SAMPLE_RATE,
    VAD_MAX_UTTERANCE_SEC,
    VAD_MIN_SILENCE_MS,
    VAD_MIN_SPEECH_MS,
    VAD_MODEL_PATH,
    VAD_POST_ROLL_MS,
    VAD_PRE_ROLL_MS,
    VAD_THRESHOLD,
)
from src.audio.vad import SileroVAD, UtteranceEndpointDetector


OUTPUT_PATH = AUDIO_INPUT_DIR / "input.wav"

warnings.filterwarnings("ignore")
os.environ["ALSA_CARD"] = "default"
os.environ["PULSE_LATENCY_MSEC"] = "10"


def get_available_devices() -> None:
    """List all available audio input devices."""
    print("Available audio devices:")
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"] > 0:
            print(f"  {index}: {device['name']} (inputs: {device['max_input_channels']})")


def _capture_from_device(device: int | None, output_path: Path) -> str:
    vad = SileroVAD(VAD_MODEL_PATH)
    endpoint = UtteranceEndpointDetector(
        sample_rate=SAMPLE_RATE,
        chunk_samples=vad.chunk_samples,
        threshold=VAD_THRESHOLD,
        min_speech_ms=VAD_MIN_SPEECH_MS,
        min_silence_ms=VAD_MIN_SILENCE_MS,
        pre_roll_ms=VAD_PRE_ROLL_MS,
        post_roll_ms=VAD_POST_ROLL_MS,
        max_utterance_sec=VAD_MAX_UTTERANCE_SEC,
    )

    print("\n🎤 Listening... Speak when you are ready (Ctrl+C to exit).")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
        blocksize=vad.chunk_samples,
    ) as stream:
        while True:
            chunk, overflowed = stream.read(vad.chunk_samples)
            if overflowed:
                print("Warning: microphone input overflowed; continuing...")

            mono = chunk[:, 0]
            event = endpoint.process(mono, vad.probability(mono))
            if event == "start":
                print("🗣️  Speech detected...")
            elif event == "end":
                break

    audio = endpoint.audio()
    if audio.size == 0:
        raise RuntimeError("No audio captured")

    sf.write(str(output_path), audio, SAMPLE_RATE, subtype="PCM_16")
    print(f"🛑 Speech ended. Saved: {output_path}")
    return str(output_path)


def record_utterance(output_path: str | Path | None = None) -> str:
    """Wait for speech and record until the configured silence endpoint."""
    path = Path(output_path) if output_path else OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        return _capture_from_device(MIC_DEVICE, path)
    except (OSError, sd.PortAudioError):
        print(f"Warning: Could not open device {MIC_DEVICE}. Trying default device...")
        get_available_devices()
        return _capture_from_device(None, path)


def record_until_enter(output_path: str | Path | None = None) -> str:
    """Backward-compatible alias for the previous microphone API."""
    return record_utterance(output_path)
