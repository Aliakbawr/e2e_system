#!/usr/bin/env python3
"""Run the existing NeMo Persian ASR once over the FLEURS fa_ir test set.

The output CSV is an append-only, recording-level checkpoint. On restart, every
file_name already present in the checkpoint (including failed recordings) is
skipped so that a deliberate rerun cannot silently change the frozen results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import re
import shutil
import sys
import tarfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPOSITORY_DIR = PROJECT_DIR.parents[1]

DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_MODEL_PATH = (
    REPOSITORY_DIR
    / "models/asr/nemo_stt_fa/stt_fa_fastconformer_hybrid_large.nemo"
)
DEFAULT_CHECKPOINT_PATH = SCRIPT_DIR / "fleurs_fa_ir_asr_checkpoint.csv"
DEFAULT_QA_PATH = (
    PROJECT_DIR / "qa/fleurs_fa_ir_generated_qa_candidates_v1.csv"
)
SAMPLE_RATE = 16_000

TSV_COLUMNS = (
    "id",
    "file_name",
    "raw_transcription",
    "transcription",
    "character_transcription",
    "num_samples",
    "gender",
)
OUTPUT_COLUMNS = (
    "id",
    "file_name",
    "audio_path",
    "gender",
    "duration_sec",
    "reference_raw",
    "reference_normalized",
    "asr_raw",
    "asr_normalized",
    "wer",
    "cer",
    "latency_sec",
    "rtf",
    "status",
    "error",
)

_PERSIAN_TRANSLATION = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ـ": "",
        "\u200c": " ",  # ZWNJ: make joined and whitespace forms comparable.
        "\u200d": "",
        "\ufeff": "",
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
)


def normalize_persian_for_evaluation(text: str) -> str:
    """Return deterministic, punctuation-free text for WER/CER evaluation."""
    text = unicodedata.normalize("NFKC", text or "").translate(_PERSIAN_TRANSLATION)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = "".join(
        " " if unicodedata.category(ch)[0] in {"P", "S", "Z", "C"} else ch
        for ch in text
    )
    return re.sub(r"\s+", " ", text).strip().lower()


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Compute Levenshtein distance using O(len(hypothesis)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row_index, reference_item in enumerate(reference, start=1):
        current = [row_index]
        for column_index, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column_index] + 1,
                    previous[column_index - 1]
                    + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


def error_rate(reference: Sequence[str], hypothesis: Sequence[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else float(len(hypothesis))
    return edit_distance(reference, hypothesis) / len(reference)


def calculate_wer_cer(reference: str, hypothesis: str) -> tuple[float, float]:
    """Calculate WER by whitespace tokens and CER including internal spaces."""
    return (
        error_rate(reference.split(), hypothesis.split()),
        error_rate(list(reference), list(hypothesis)),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(tsv_path: Path) -> list[dict[str, str]]:
    if not tsv_path.is_file():
        raise FileNotFoundError(f"FLEURS manifest not found: {tsv_path}")
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, fieldnames=TSV_COLUMNS, delimiter="\t"))

    filenames = [row["file_name"] for row in rows]
    if len(rows) != 871:
        raise ValueError(f"Expected 871 FLEURS rows, found {len(rows)}")
    if len(set(filenames)) != 871:
        raise ValueError("The FLEURS manifest does not contain 871 unique filenames")
    if len({row["id"] for row in rows}) != 324:
        raise ValueError("The FLEURS manifest does not contain 324 unique IDs")
    if len({row["transcription"] for row in rows}) != 324:
        raise ValueError("The FLEURS manifest does not contain 324 unique transcripts")
    return rows


def validate_frozen_qa(qa_path: Path) -> str:
    if not qa_path.is_file():
        raise FileNotFoundError(f"Frozen QA benchmark not found: {qa_path}")
    before_hash = sha256_file(qa_path)
    with qa_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 324:
        raise ValueError(f"Expected 324 frozen QA candidate rows, found {len(rows)}")
    return before_hash


def audio_paths_for(rows: Iterable[dict[str, str]], data_dir: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for row in rows:
        filename = row["file_name"]
        candidates = (data_dir / "test" / filename, data_dir / filename)
        match = next((path.resolve() for path in candidates if path.is_file()), None)
        if match is not None:
            paths[filename] = match
    return paths


def extract_expected_wavs(
    archive_path: Path, data_dir: Path, expected_filenames: set[str]
) -> None:
    """Extract only manifest-listed regular WAV files; ignore every other member."""
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"No extracted WAVs and FLEURS archive not found: {archive_path}"
        )
    destination = data_dir / "test"
    destination.mkdir(parents=True, exist_ok=True)
    extracted: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            filename = Path(member.name).name
            if not member.isfile() or filename not in expected_filenames:
                continue
            source = archive.extractfile(member)
            if source is None:
                raise OSError(f"Could not read archive member: {member.name}")
            target = destination / filename
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.add(filename)
    missing = expected_filenames - extracted
    if missing:
        raise ValueError(f"Archive is missing {len(missing)} manifest WAV files")


def validate_archive_members(archive_path: Path, expected_filenames: set[str]) -> None:
    if not archive_path.is_file():
        raise FileNotFoundError(
            f"WAVs are not extracted and FLEURS archive was not found: {archive_path}"
        )
    found: list[str] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            filename = Path(member.name).name
            if member.isfile() and filename in expected_filenames:
                found.append(filename)
    if len(found) != 871 or set(found) != expected_filenames:
        raise ValueError(
            "test.tar.gz does not contain exactly the 871 manifest-listed WAV files"
        )


def read_wav_duration(path: Path) -> tuple[float, int]:
    # Official FLEURS WAVs use IEEE float PCM (format tag 3), unsupported by
    # Python's standard-library wave module but supported by libsndfile.
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Reading FLEURS WAV metadata requires soundfile") from exc
    info = sf.info(str(path))
    return info.frames / info.samplerate, info.samplerate


def completed_filenames(checkpoint_path: Path) -> set[str]:
    if not checkpoint_path.exists():
        return set()
    with checkpoint_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != OUTPUT_COLUMNS:
            raise ValueError(
                f"Checkpoint schema mismatch in {checkpoint_path}; refusing to overwrite it"
            )
        rows = list(reader)
    filenames = [row["file_name"] for row in rows]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Checkpoint contains duplicate file_name values")
    return set(filenames)


def load_existing_asr(model_path: Path) -> tuple[Any, Any, str]:
    """Restore the same model and device behavior used by src/asr/transcriber.py."""
    if not model_path.is_file():
        raise FileNotFoundError(f"NeMo model archive not found: {model_path}")

    # Match config/settings.py's writable-cache setup without importing the chat
    # pipeline (which would instantiate a second ASR model).
    cache_dir = Path(os.getenv("TMPDIR", "/tmp")) / "persian_assistant_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir / "numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "matplotlib"))

    try:
        import torch
        import nemo.collections.asr as nemo_asr
    except ImportError as exc:
        raise RuntimeError(
            "This runner requires the same PyTorch and NVIDIA NeMo environment "
            "as the existing ASR implementation."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nemo_asr.models.ASRModel.restore_from(str(model_path))
    model = model.to(device)
    model.eval()
    return model, torch, device


def transcribe_once(model: Any, audio_path: Path) -> str:
    """Preserve the existing one-file NeMo transcription and decoding logic."""
    result = model.transcribe([str(audio_path)])
    hypothesis = result[0]
    return hypothesis.text if hasattr(hypothesis, "text") else str(hypothesis)


def format_float(value: float) -> str:
    return "" if not math.isfinite(value) else f"{value:.9f}"


def append_checkpoint_row(
    checkpoint_path: Path, row: dict[str, str], write_header: bool
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def run(args: argparse.Namespace) -> int:
    data_dir = args.data_dir.resolve()
    rows = read_manifest(data_dir / "test.tsv")
    qa_hash = validate_frozen_qa(args.qa_path.resolve())
    expected_filenames = {row["file_name"] for row in rows}

    paths = audio_paths_for(rows, data_dir)
    if len(paths) != len(rows) and not args.validate_only:
        print("Extracting the 871 manifest-listed WAVs from test.tar.gz ...")
        extract_expected_wavs(data_dir / "test.tar.gz", data_dir, expected_filenames)
        paths = audio_paths_for(rows, data_dir)
    if not args.validate_only and len(paths) != len(rows):
        raise FileNotFoundError(
            f"Found {len(paths)}/871 WAVs after extraction in {data_dir}"
        )

    if args.validate_only:
        if len(paths) == 871:
            archive_note = "extracted"
        else:
            validate_archive_members(data_dir / "test.tar.gz", expected_filenames)
            archive_note = "validated in test.tar.gz"
        print(
            "Validation passed: 871 recordings, 324 IDs, 324 transcripts; "
            f"audio {archive_note}; frozen QA SHA-256={qa_hash}"
        )
        return 0

    checkpoint_path = args.checkpoint.resolve()
    already_recorded = completed_filenames(checkpoint_path)
    unknown = already_recorded - expected_filenames
    if unknown:
        raise ValueError(
            f"Checkpoint contains {len(unknown)} filenames absent from this manifest"
        )
    remaining = [row for row in rows if row["file_name"] not in already_recorded]
    if not remaining:
        print(f"Checkpoint is already complete: {checkpoint_path}")
        return 0

    model, torch, device = load_existing_asr(args.model_path.resolve())
    print(
        f"Model: nvidia/stt_fa_fastconformer_hybrid_large | device: {device} | "
        f"remaining: {len(remaining)}/{len(rows)}"
    )

    write_header = not checkpoint_path.exists()
    for index, manifest_row in enumerate(remaining, start=1):
        filename = manifest_row["file_name"]
        audio_path = paths[filename]
        duration, actual_sample_rate = read_wav_duration(audio_path)
        if actual_sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Unexpected sample rate for {filename}: {actual_sample_rate} Hz"
            )

        asr_raw = ""
        asr_normalized = ""
        wer = cer = latency = rtf = float("nan")
        status = "ok"
        error = ""
        try:
            if device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            asr_raw = transcribe_once(model, audio_path)
            if device == "cuda":
                torch.cuda.synchronize()
            latency = time.perf_counter() - started

            asr_normalized = normalize_persian_for_evaluation(asr_raw)
            evaluation_reference = normalize_persian_for_evaluation(
                manifest_row["transcription"]
            )
            wer, cer = calculate_wer_cer(evaluation_reference, asr_normalized)
            rtf = latency / duration if duration > 0 else float("nan")
        except Exception as exc:  # Preserve failures as frozen recording results.
            if device == "cuda":
                torch.cuda.synchronize()
            status = "error"
            error = f"{type(exc).__name__}: {exc}"

        output_row = {
            "id": manifest_row["id"],
            "file_name": filename,
            "audio_path": str(audio_path),
            "gender": manifest_row["gender"],
            "duration_sec": format_float(duration),
            "reference_raw": manifest_row["raw_transcription"],
            # Keep this byte-for-byte equal to the official FLEURS field.
            "reference_normalized": manifest_row["transcription"],
            "asr_raw": asr_raw,
            "asr_normalized": asr_normalized,
            "wer": format_float(wer),
            "cer": format_float(cer),
            "latency_sec": format_float(latency),
            "rtf": format_float(rtf),
            "status": status,
            "error": error,
        }
        append_checkpoint_row(checkpoint_path, output_row, write_header)
        write_header = False
        print(
            f"[{len(already_recorded) + index:03d}/871] {filename}: {status}",
            flush=True,
        )

    if sha256_file(args.qa_path.resolve()) != qa_hash:
        raise RuntimeError("Frozen QA benchmark changed during the ASR run")
    final_count = len(completed_filenames(checkpoint_path))
    if final_count != 871:
        raise RuntimeError(f"Checkpoint is incomplete: {final_count}/871 rows")
    print(f"Frozen ASR checkpoint complete: {checkpoint_path}")
    print(f"Frozen QA benchmark unchanged: SHA-256={qa_hash}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate manifests/QA/archive without extracting audio or loading ASR.",
    )
    return parser.parse_args(argv)


def main() -> int:
    try:
        return run(parse_args())
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
