import subprocess
from pathlib import Path

from config.settings import AUDIO_PLAYER


def play_audio(path: str) -> None:

    print("PLAY PATH:", path)

    if not Path(path).exists():
        print("FILE DOES NOT EXIST")
        return

    print("Playing...")

    try:
        subprocess.run([AUDIO_PLAYER, str(path)], check=False)
    except FileNotFoundError:
        print(
            f"Audio player '{AUDIO_PLAYER}' was not found. "
            "Set PERSIAN_ASSISTANT_AUDIO_PLAYER to an installed command."
        )
