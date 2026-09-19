"""Report whether a container has the models, accelerator, and audio it needs."""

from __future__ import annotations

import importlib.metadata
import os

from scripts.container_entrypoint import DEFAULT_ASSETS, missing_assets


PACKAGES = (
    "torch",
    "transformers",
    "bitsandbytes",
    "vosk",
    "piper-tts",
    "onnxruntime",
    "sounddevice",
)


def main() -> int:
    failures = 0
    print("Persian Speech Chatbot container check")
    print("\nPackages:")
    for package in PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            print(f"  FAIL {package}: not installed")
            failures += 1
        else:
            print(f"  OK   {package} {version}")

    print("\nModels:")
    missing = dict(missing_assets())
    failures += len(missing)
    for label, (environment_name, default) in DEFAULT_ASSETS.items():
        path = os.getenv(environment_name, default)
        state = "FAIL" if label in missing else "OK  "
        print(f"  {state} {label}: {path}")

    print("\nCompute:")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"  OK   CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("  INFO CUDA is unavailable; PyTorch will use the CPU")
    except Exception as error:  # pragma: no cover - diagnostic boundary
        print(f"  FAIL PyTorch probe: {error}")
        failures += 1

    print("\nAudio:")
    try:
        import sounddevice

        devices = sounddevice.query_devices()
        inputs = [device for device in devices if device["max_input_channels"] > 0]
        outputs = [device for device in devices if device["max_output_channels"] > 0]
        if inputs and outputs:
            print(f"  OK   {len(inputs)} input device(s), {len(outputs)} output device(s)")
        else:
            print(
                f"  WARN {len(inputs)} input device(s), {len(outputs)} output device(s); "
                "check the selected Linux/WSL Compose overlay"
            )
    except Exception as error:  # pragma: no cover - host hardware dependent
        print(f"  WARN audio probe failed: {error}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
