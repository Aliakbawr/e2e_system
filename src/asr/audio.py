"""Audio preparation profiles for Persian ASR."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


SUPPORTED_PROFILES = ("none", "low_level_gain")
LOW_LEVEL_THRESHOLD_DBFS = -45.0
TARGET_RMS_DBFS = -24.0
MAX_GAIN_DB = 40.0
PEAK_HEADROOM_DBFS = -1.0


@dataclass(frozen=True)
class PreparedAudio:
    pcm16: bytes
    duration_sec: float
    source_rate: int
    target_rate: int
    profile: str
    applied: bool
    input_rms_dbfs: float
    output_rms_dbfs: float
    gain_db: float

    def metrics(self) -> dict:
        return {
            "profile": self.profile,
            "applied": self.applied,
            "source_rate": self.source_rate,
            "target_rate": self.target_rate,
            "duration_sec": self.duration_sec,
            "input_rms_dbfs": self.input_rms_dbfs,
            "output_rms_dbfs": self.output_rms_dbfs,
            "gain_db": self.gain_db,
        }


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))


def _low_level_gain(audio: np.ndarray) -> tuple[np.ndarray, float]:
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    rms_dbfs = _dbfs(rms)
    if rms_dbfs >= LOW_LEVEL_THRESHOLD_DBFS or rms == 0.0:
        return audio, 0.0

    peak = float(np.max(np.abs(audio)))
    peak_dbfs = _dbfs(peak)
    gain_db = min(
        TARGET_RMS_DBFS - rms_dbfs,
        MAX_GAIN_DB,
        PEAK_HEADROOM_DBFS - peak_dbfs,
    )
    if gain_db <= 0.0:
        return audio, 0.0
    return audio * (10.0 ** (gain_db / 20.0)), gain_db


def prepare_audio_file(
    audio_path: str | Path,
    target_rate: int,
    profile: str = "none",
) -> PreparedAudio:
    """Read a file and create mono PCM16 for one explicit ablation profile."""
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(
            f"Unknown ASR audio preprocessing profile {profile!r}; "
            f"choose one of {SUPPORTED_PROFILES}"
        )
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.size == 0:
        return PreparedAudio(
            pcm16=b"",
            duration_sec=0.0,
            source_rate=int(source_rate),
            target_rate=target_rate,
            profile=profile,
            applied=False,
            input_rms_dbfs=-240.0,
            output_rms_dbfs=-240.0,
            gain_db=0.0,
        )

    audio = audio.mean(axis=1)
    if source_rate != target_rate:
        divisor = math.gcd(int(source_rate), target_rate)
        audio = resample_poly(
            audio,
            target_rate // divisor,
            int(source_rate) // divisor,
        )

    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    input_rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    gain_db = 0.0
    if profile == "low_level_gain":
        audio, gain_db = _low_level_gain(audio)

    audio = np.clip(audio, -1.0, 1.0)
    output_rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return PreparedAudio(
        pcm16=(audio * 32767.0).astype("<i2").tobytes(),
        duration_sec=len(audio) / target_rate,
        source_rate=int(source_rate),
        target_rate=target_rate,
        profile=profile,
        applied=gain_db > 0.0,
        input_rms_dbfs=_dbfs(input_rms),
        output_rms_dbfs=_dbfs(output_rms),
        gain_db=gain_db,
    )
