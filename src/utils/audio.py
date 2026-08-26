import librosa
import soundfile as sf
import numpy as np
from config.settings import SAMPLE_RATE


def prepare_audio(input_path: str, output_path: str = "temp.wav") -> str:
    audio, _ = librosa.load(
        input_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    sf.write(output_path, audio.astype(np.float32), SAMPLE_RATE)
    return output_path