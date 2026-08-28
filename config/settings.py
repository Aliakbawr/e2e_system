import os
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
AUDIO_INPUT_DIR = DATA_DIR / "audio_in"
AUDIO_OUTPUT_DIR = DATA_DIR / "audio_out"
LOG_FILE = Path(
    os.getenv(
        "PERSIAN_ASSISTANT_LOG_FILE",
        str(DATA_DIR / "logs/chatbot.log"),
    )
).expanduser()
LOG_LEVEL = os.getenv("PERSIAN_ASSISTANT_LOG_LEVEL", "INFO")
LOG_MAX_BYTES = int(os.getenv("PERSIAN_ASSISTANT_LOG_MAX_BYTES", "5242880"))
LOG_BACKUP_COUNT = int(os.getenv("PERSIAN_ASSISTANT_LOG_BACKUP_COUNT", "3"))

# Numba (used by librosa) and Matplotlib try to cache compiled/config
# files during import.  Keep those caches in a writable location so model
# loading also works when the Python environment itself is read-only.
CACHE_DIR = Path(tempfile.gettempdir()) / "persian_assistant_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE_DIR / "numba"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))

MIC_DEVICE = int(os.getenv("PERSIAN_ASSISTANT_MIC_DEVICE", "0"))
SAMPLE_RATE = 16000
AUDIO_PLAYER = os.getenv("PERSIAN_ASSISTANT_AUDIO_PLAYER", "paplay")
# ASR (Vosk)
ASR_MODEL_NAME = "vosk-model-fa-0.42"
ASR_LOG_LEVEL = int(os.getenv("PERSIAN_ASSISTANT_VOSK_LOG_LEVEL", "-1"))
ASR_MAX_ALTERNATIVES = max(
    1,
    int(os.getenv("PERSIAN_ASSISTANT_ASR_MAX_ALTERNATIVES", "3")),
)
ASR_WORD_CONFIDENCE_THRESHOLD = float(
    os.getenv("PERSIAN_ASSISTANT_ASR_WORD_CONFIDENCE_THRESHOLD", "0.65")
)
ASR_ALTERNATIVE_SCORE_GAP = float(
    os.getenv("PERSIAN_ASSISTANT_ASR_ALTERNATIVE_SCORE_GAP", "1.0")
)
ASR_CLARIFICATION_MAX_OPTIONS = max(
    2,
    int(os.getenv("PERSIAN_ASSISTANT_ASR_CLARIFICATION_MAX_OPTIONS", "2")),
)
ASR_AUDIO_PREPROCESSING = os.getenv(
    "PERSIAN_ASSISTANT_ASR_AUDIO_PREPROCESSING",
    "low_level_gain",
)

ASR_MODEL_PATH = os.getenv(
    "PERSIAN_ASSISTANT_ASR_MODEL_PATH",
    str(MODELS_DIR / "asr/vosk/vosk-model-fa-0.42"),
)

# LLM
LLM_MODEL_PATH = os.getenv(
    "PERSIAN_ASSISTANT_LLM_MODEL_PATH",
    str(MODELS_DIR / "llm/gemma-2-9b-it-4bit"),
)

# TTS (Piper)
TTS_MODEL_PATH = os.getenv(
    "PERSIAN_ASSISTANT_TTS_MODEL_PATH",
    str(MODELS_DIR / "tts/piper/fa_IR-gyro-medium.onnx"),
)

TTS_CONFIG_PATH = os.getenv(
    "PERSIAN_ASSISTANT_TTS_CONFIG_PATH",
    str(MODELS_DIR / "tts/piper/fa_IR-gyro-medium.onnx.json"),
)

MAX_LLM_TOKENS = int(os.getenv("PERSIAN_ASSISTANT_MAX_LLM_TOKENS", "150"))
MAX_LLM_INPUT_TOKENS = int(
    os.getenv("PERSIAN_ASSISTANT_MAX_LLM_INPUT_TOKENS", "2048")
)
