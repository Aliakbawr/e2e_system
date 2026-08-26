import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import warnings
import threading

from config.settings import AUDIO_INPUT_DIR, MIC_DEVICE, SAMPLE_RATE


OUTPUT_PATH = AUDIO_INPUT_DIR / "input.wav"

# Suppress ALSA warnings
warnings.filterwarnings('ignore')
os.environ['ALSA_CARD'] = 'default'
os.environ['PULSE_LATENCY_MSEC'] = '10'


def get_available_devices():
    """List all available audio input devices."""
    devices = sd.query_devices()
    print("Available audio devices:")
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"  {i}: {device['name']} (inputs: {device['max_input_channels']})")


def record_until_enter():

    AUDIO_INPUT_DIR.mkdir(parents=True, exist_ok=True)


    input(
        "\nPress Enter to start recording..."
    )


    print("\nListening... Speak now")
    print("Press Enter to stop\n")

    # Use blocking read instead of callback to avoid PortAudio threading issues
    frames = []
    stop_recording = threading.Event()
    
    def recording_thread():
        """Run recording in a separate thread to handle blocking reads."""
        try:
            # Try with the configured device first
            device = MIC_DEVICE
            try:
                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    device=device,
                    blocksize=4096
                ) as stream:
                    while not stop_recording.is_set():
                        try:
                            data = stream.read(4096)
                            frames.append(data[0])
                        except Exception as e:
                            print(f"Error reading from stream: {e}")
                            break
            except (OSError, sd.PortAudioError) as e:
                print(f"Warning: Could not open device {device}. Trying default device...")
                get_available_devices()
                # Fall back to default device
                with sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=1,
                    device=None,  # Use default device
                    blocksize=4096
                ) as stream:
                    while not stop_recording.is_set():
                        try:
                            data = stream.read(4096)
                            frames.append(data[0])
                        except Exception as e:
                            print(f"Error reading from stream: {e}")
                            break
        except Exception as e:
            print(f"Fatal error in recording: {e}")
            raise
    
    # Start recording thread
    rec_thread = threading.Thread(target=recording_thread, daemon=True)
    rec_thread.start()
    
    # Wait for user to press Enter to stop
    input()
    
    # Stop recording
    stop_recording.set()
    rec_thread.join(timeout=1.0)


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
