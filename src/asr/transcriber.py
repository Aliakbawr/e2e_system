import torch

# Import settings first: it configures writable cache directories needed by
# NeMo's librosa/Numba dependencies during import and model restoration.
from config.settings import ASR_MODEL_PATH
import nemo.collections.asr as nemo_asr

device = "cuda" if torch.cuda.is_available() else "cpu"

model = nemo_asr.models.ASRModel.restore_from(ASR_MODEL_PATH)
model = model.to(device)

def transcribe_audio(audio_path: str) -> str:
    result = model.transcribe([audio_path])
    hyp = result[0]
    return hyp.text if hasattr(hyp, "text") else str(hyp)
