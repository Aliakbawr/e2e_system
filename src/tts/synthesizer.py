import time
import wave
import soundfile as sf
import hazm
import re

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

    text = re.sub(r"[^\w\s]", " ", text)
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