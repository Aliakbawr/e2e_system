import numpy as np
import queue
import threading
from pathlib import Path

import scipy.io.wavfile as wav
import sounddevice as sd

from config.settings import AUDIO_INPUT_DIR, MIC_DEVICE, SAMPLE_RATE


audio_queue = queue.Queue()
recording = True


def callback(indata, frames, time, status):
    if recording:
        audio_queue.put(indata.copy())


def record_until_enter(output_path=None):
    global recording
    recording = True
    output_path = Path(output_path) if output_path else AUDIO_INPUT_DIR / "mic.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    while not audio_queue.empty():
        audio_queue.get_nowait()

    print("\n🎤 Recording started...")
    print("👉 Press ENTER to stop...\n")

    chunks = []

    def wait_for_enter():
        input()
        global recording
        recording = False

    # thread that waits for ENTER
    t = threading.Thread(target=wait_for_enter)
    t.start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=MIC_DEVICE,
        callback=callback,
    ):
        while recording:
            try:
                chunk = audio_queue.get(timeout=0.1)
                chunks.append(chunk)
            except queue.Empty:
                pass

    t.join()

    print("🛑 Recording stopped.")

    if len(chunks) == 0:
        raise RuntimeError("No audio captured")

    audio = np.concatenate(chunks, axis=0)

    wav.write(str(output_path), SAMPLE_RATE, audio)

    return str(output_path)
