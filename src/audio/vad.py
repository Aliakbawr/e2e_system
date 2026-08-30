"""Streaming Silero VAD inference and utterance endpoint detection."""

from collections import deque
from math import ceil
from pathlib import Path

import numpy as np


class SileroVAD:
    """Run the official Silero ONNX model over 16 kHz mono audio chunks."""

    sample_rate = 16000
    chunk_samples = 512
    context_samples = 64

    def __init__(self, model_path: str | Path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "Silero VAD requires onnxruntime; install the project requirements"
            ) from exc

        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Silero VAD model not found: {model_path}. "
                "See models/README.md for installation instructions."
            )

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.context_samples), dtype=np.float32)

    def probability(self, chunk: np.ndarray) -> float:
        audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if audio.size != self.chunk_samples:
            raise ValueError(
                f"Silero VAD expects {self.chunk_samples} samples, got {audio.size}"
            )

        current = audio.reshape(1, -1)
        model_input = np.concatenate((self._context, current), axis=1)
        output, self._state = self._session.run(
            None,
            {
                "input": model_input,
                "state": self._state,
                "sr": np.asarray(self.sample_rate, dtype=np.int64),
            },
        )
        self._context = model_input[:, -self.context_samples :]
        return float(output.item())


class UtteranceEndpointDetector:
    """Convert frame-level VAD probabilities into one complete utterance."""

    def __init__(
        self,
        *,
        sample_rate: int,
        chunk_samples: int,
        threshold: float,
        min_speech_ms: int,
        min_silence_ms: int,
        pre_roll_ms: int,
        post_roll_ms: int,
        max_utterance_sec: float,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("VAD threshold must be between 0 and 1")
        if min_speech_ms <= 0 or min_silence_ms <= 0:
            raise ValueError("VAD speech and silence durations must be positive")
        if pre_roll_ms < 0 or post_roll_ms < 0 or max_utterance_sec <= 0:
            raise ValueError("VAD buffer durations must not be negative")

        frame_ms = chunk_samples * 1000.0 / sample_rate
        self.threshold = threshold
        self.start_frames = max(1, ceil(min_speech_ms / frame_ms))
        self.stop_frames = max(1, ceil(min_silence_ms / frame_ms))
        self.post_roll_frames = max(0, ceil(post_roll_ms / frame_ms))
        self.max_frames = max(1, ceil(max_utterance_sec * sample_rate / chunk_samples))
        pre_roll_frames = max(self.start_frames, ceil(pre_roll_ms / frame_ms))

        self._pre_roll: deque[np.ndarray] = deque(maxlen=pre_roll_frames)
        self._utterance: list[np.ndarray] = []
        self._speech_run = 0
        self._silence_run = 0
        self.speaking = False

    def process(self, chunk: np.ndarray, probability: float) -> str | None:
        """Consume a frame and return ``start`` or ``end`` on transitions."""
        frame = np.asarray(chunk, dtype=np.float32).reshape(-1).copy()
        voiced = probability >= self.threshold

        if not self.speaking:
            self._pre_roll.append(frame)
            self._speech_run = self._speech_run + 1 if voiced else 0
            if self._speech_run < self.start_frames:
                return None

            self.speaking = True
            self._utterance = list(self._pre_roll)
            self._silence_run = 0
            return "start"

        self._utterance.append(frame)
        self._silence_run = 0 if voiced else self._silence_run + 1

        if len(self._utterance) >= self.max_frames:
            return "end"
        if self._silence_run < self.stop_frames:
            return None

        excess_silence = max(0, self._silence_run - self.post_roll_frames)
        if excess_silence:
            del self._utterance[-excess_silence:]
        return "end"

    def audio(self) -> np.ndarray:
        if not self._utterance:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self._utterance).astype(np.float32, copy=False)
