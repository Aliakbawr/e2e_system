import queue
import re
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import hazm
import sounddevice as sd
import soundfile as sf

from piper.voice import PiperVoice
from config.settings import TTS_MODEL_PATH, TTS_CONFIG_PATH

voice = PiperVoice.load(TTS_MODEL_PATH, TTS_CONFIG_PATH)
normalizer = hazm.Normalizer()

PERSIAN_MAP = {
    "ك": "ک",
    "ي": "ی",
    "ى": "ی",
    "ؤ": "و",
    "ئ": "ی",
    "ة": "ه",
}


def normalize_text(text: str) -> str:
    text = normalizer.normalize(str(text))

    for k, v in PERSIAN_MAP.items():
        text = text.replace(k, v)

    text = re.sub(r"[^\w\s.!?؟،؛]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def synthesize_speech(text: str, output_path: str):
    text = normalize_text(text)

    start = time.time()

    with wave.open(output_path, "wb") as f:
        voice.synthesize_wav(text, f)

    runtime = time.time() - start

    audio, sr = sf.read(output_path)
    duration = len(audio) / sr

    return {
        "audio_path": output_path,
        "tts_time": runtime,
        "audio_duration": duration,
        "tts_rtf": runtime / duration
    }


@dataclass(frozen=True)
class _PCMChunk:
    audio: bytes
    sample_rate: int
    sample_width: int
    channels: int


_END = object()


class StreamingSpeechOutput:
    """Synthesize submitted phrases and play PCM in ordered worker stages."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._text_queue = queue.Queue()
        self._audio_queue = queue.Queue()
        self._errors = []
        self._tts_time = 0.0
        self._audio_samples = 0
        self._sample_rate = 0
        self._started_at = time.monotonic()
        self._first_audio_at = None
        self._closed = False
        self._tts_thread = threading.Thread(target=self._synthesize, daemon=True)
        self._play_thread = threading.Thread(target=self._play, daemon=True)
        self._tts_thread.start()
        self._play_thread.start()

    def submit(self, phrase: str) -> None:
        if self._closed:
            raise RuntimeError("Streaming speech output is already closed")
        text = normalize_text(phrase)
        if text:
            self._text_queue.put(text)

    def finish(self) -> dict:
        if not self._closed:
            self._closed = True
            self._text_queue.put(_END)
        self._tts_thread.join()
        self._play_thread.join()
        if self._errors:
            raise RuntimeError("Streaming TTS/playback failed") from self._errors[0]

        audio_duration = (
            self._audio_samples / self._sample_rate if self._sample_rate else 0.0
        )
        return {
            "audio_path": str(self.output_path),
            "tts_time": self._tts_time,
            "audio_duration": audio_duration,
            "tts_rtf": self._tts_time / audio_duration if audio_duration else 0.0,
            "time_to_first_audio": (
                self._first_audio_at - self._started_at
                if self._first_audio_at is not None
                else None
            ),
        }

    def _synthesize(self) -> None:
        try:
            while True:
                phrase = self._text_queue.get()
                if phrase is _END:
                    break

                started = time.monotonic()
                for chunk in voice.synthesize(phrase):
                    self._audio_queue.put(
                        _PCMChunk(
                            audio=chunk.audio_int16_bytes,
                            sample_rate=chunk.sample_rate,
                            sample_width=chunk.sample_width,
                            channels=chunk.sample_channels,
                        )
                    )
                self._tts_time += time.monotonic() - started
        except BaseException as exc:
            self._errors.append(exc)
        finally:
            self._audio_queue.put(_END)

    def _play(self) -> None:
        wav_file = None
        audio_stream = None
        audio_format = None
        try:
            while True:
                chunk = self._audio_queue.get()
                if chunk is _END:
                    break

                chunk_format = (
                    chunk.sample_rate,
                    chunk.sample_width,
                    chunk.channels,
                )
                if audio_format is None:
                    if chunk.sample_width != 2:
                        raise RuntimeError(
                            f"Streaming playback requires PCM16, got "
                            f"{chunk.sample_width * 8}-bit audio"
                        )
                    audio_format = chunk_format
                    self._sample_rate = chunk.sample_rate
                    wav_file = wave.open(str(self.output_path), "wb")
                    wav_file.setframerate(chunk.sample_rate)
                    wav_file.setsampwidth(chunk.sample_width)
                    wav_file.setnchannels(chunk.channels)
                    audio_stream = sd.RawOutputStream(
                        samplerate=chunk.sample_rate,
                        channels=chunk.channels,
                        dtype="int16",
                    )
                    audio_stream.start()
                elif chunk_format != audio_format:
                    raise RuntimeError("Piper changed audio format during streaming")

                if self._first_audio_at is None:
                    self._first_audio_at = time.monotonic()
                wav_file.writeframesraw(chunk.audio)
                audio_stream.write(chunk.audio)
                self._audio_samples += len(chunk.audio) // (
                    chunk.sample_width * chunk.channels
                )
        except BaseException as exc:
            self._errors.append(exc)
        finally:
            if audio_stream is not None:
                audio_stream.stop()
                audio_stream.close()
            if wav_file is not None:
                wav_file.close()
