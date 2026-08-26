import sounddevice as sd
import soundfile as sf

from config.settings import AUDIO_INPUT_DIR, MIC_DEVICE, SAMPLE_RATE

OUTPUT_PATH = AUDIO_INPUT_DIR / "mic.wav"
DURATION = 5


def main():
    AUDIO_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("🎤 Recording... Speak now")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=MIC_DEVICE,
    )

    sd.wait()
    sf.write(str(OUTPUT_PATH), audio, SAMPLE_RATE)
    print(f"✅ Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
