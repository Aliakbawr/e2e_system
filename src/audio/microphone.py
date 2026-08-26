import numpy as np
import sounddevice as sd
import soundfile as sf

from config.settings import AUDIO_INPUT_DIR, MIC_DEVICE, SAMPLE_RATE


OUTPUT_PATH = AUDIO_INPUT_DIR / "input.wav"


def record_until_enter():

    AUDIO_INPUT_DIR.mkdir(parents=True, exist_ok=True)


    input(
        "\nPress Enter to start recording..."
    )


    frames = []


    def callback(
        indata,
        frames_count,
        time,
        status
    ):

        frames.append(
            indata.copy()
        )


    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        device=MIC_DEVICE,
        callback=callback
    )


    print("\nListening... Speak now")
    print("Press Enter to stop\n")


    stream.start()


    input()


    stream.stop()
    stream.close()


    if len(frames) == 0:
        print("No audio captured")
        return None

    audio = np.concatenate(frames)

    sf.write(
        str(OUTPUT_PATH),
        audio,
        SAMPLE_RATE
    )


    print(
        f"Saved: {OUTPUT_PATH}"
    )


    return str(OUTPUT_PATH)
