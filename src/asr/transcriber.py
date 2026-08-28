"""Vosk-based Persian speech recognition for the chatbot runtime."""

import json
import logging
from functools import lru_cache
from pathlib import Path

from config.settings import (
    ASR_AUDIO_PREPROCESSING,
    ASR_LOG_LEVEL,
    ASR_MAX_ALTERNATIVES,
    ASR_MODEL_PATH,
    SAMPLE_RATE,
)
from src.asr.audio import prepare_audio_file
from src.asr.types import RecognizedWord, TranscriptAlternative, TranscriptionResult


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_model():
    """Load the local Vosk model once, on the first transcription request."""
    model_path = Path(ASR_MODEL_PATH)

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Vosk model directory not found: {model_path}. "
            "Extract vosk-model-fa-0.42.zip into models/asr/vosk/."
        )

    try:
        from vosk import Model, SetLogLevel
    except ImportError as exc:
        raise RuntimeError(
            "The 'vosk' Python package is not installed. "
            "Install application dependencies with: pip install -r requirements.txt"
        ) from exc

    SetLogLevel(ASR_LOG_LEVEL)
    return Model(str(model_path))


def _read_pcm16(audio_path: str, preprocessing_profile: str | None = None) -> bytes:
    """Read an audio file and return 16 kHz mono signed PCM16 bytes."""
    profile = preprocessing_profile or ASR_AUDIO_PREPROCESSING
    prepared = prepare_audio_file(audio_path, SAMPLE_RATE, profile)
    if prepared.applied:
        logger.info(
            "asr_audio_preprocessed profile=%s input_rms_dbfs=%.2f "
            "output_rms_dbfs=%.2f gain_db=%.2f duration_sec=%.3f",
            prepared.profile,
            prepared.input_rms_dbfs,
            prepared.output_rms_dbfs,
            prepared.gain_db,
            prepared.duration_sec,
        )
    return prepared.pcm16


def _optional_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_words(items) -> tuple[RecognizedWord, ...]:
    words = []
    for item in items or []:
        text = str(item.get("word", "")).strip()
        if not text:
            continue
        words.append(
            RecognizedWord(
                text=text,
                confidence=_optional_float(item.get("conf")),
                start=_optional_float(item.get("start")),
                end=_optional_float(item.get("end")),
            )
        )
    return tuple(words)


def _parse_result(payload: str) -> tuple[TranscriptAlternative, ...]:
    """Parse both standard and N-best Vosk result shapes."""
    data = json.loads(payload)
    raw_alternatives = data.get("alternatives")
    if not raw_alternatives:
        raw_alternatives = [data]

    alternatives = []
    for item in raw_alternatives:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        alternatives.append(
            TranscriptAlternative(
                text=text,
                words=_parse_words(item.get("result")),
                decoder_score=_optional_float(item.get("confidence")),
            )
        )
    return tuple(alternatives)


def _combine_segments(
    segments: list[tuple[TranscriptAlternative, ...]],
) -> TranscriptionResult:
    """Combine Vosk speech segments into utterance-level ranked hypotheses."""
    non_empty = [segment for segment in segments if segment]
    if not non_empty:
        return TranscriptionResult(text="")

    combined = []
    seen_texts = set()
    rank_count = min(
        ASR_MAX_ALTERNATIVES,
        max(len(segment) for segment in non_empty),
    )

    for rank in range(rank_count):
        choices = [
            segment[rank] if rank < len(segment) else segment[0]
            for segment in non_empty
        ]
        text = " ".join(choice.text for choice in choices).strip()
        if not text or text in seen_texts:
            continue

        scores = [choice.decoder_score for choice in choices]
        combined.append(
            TranscriptAlternative(
                text=text,
                words=tuple(
                    word for choice in choices for word in choice.words
                ),
                decoder_score=(
                    sum(scores) if all(score is not None for score in scores) else None
                ),
            )
        )
        seen_texts.add(text)

    primary = combined[0]
    return TranscriptionResult(
        text=primary.text,
        words=primary.words,
        alternatives=tuple(combined),
    )


def _recognize_pcm(pcm: bytes, max_alternatives: int = 0) -> TranscriptionResult:
    """Run one Vosk decoding pass over PCM audio."""
    from vosk import KaldiRecognizer

    recognizer = KaldiRecognizer(_load_model(), SAMPLE_RATE)
    recognizer.SetWords(True)
    if max_alternatives > 0:
        recognizer.SetMaxAlternatives(max_alternatives)

    segments = []
    chunk_size = 8000

    for offset in range(0, len(pcm), chunk_size):
        chunk = pcm[offset : offset + chunk_size]
        if recognizer.AcceptWaveform(chunk):
            segments.append(_parse_result(recognizer.Result()))

    segments.append(_parse_result(recognizer.FinalResult()))
    return _combine_segments(segments)


def _merge_results(
    primary: TranscriptionResult,
    nbest: TranscriptionResult,
) -> TranscriptionResult:
    """Attach N-best hypotheses while preserving MBR text and word confidence."""
    primary_decoder_score = next(
        (
            alternative.decoder_score
            for alternative in nbest.alternatives
            if alternative.text == primary.text
        ),
        None,
    )
    alternatives = [
        TranscriptAlternative(
            text=primary.text,
            words=primary.words,
            decoder_score=primary_decoder_score,
        )
    ]
    seen_texts = {primary.text}

    for alternative in nbest.alternatives:
        if alternative.text in seen_texts:
            continue
        alternatives.append(alternative)
        seen_texts.add(alternative.text)
        if len(alternatives) >= ASR_MAX_ALTERNATIVES:
            break

    return TranscriptionResult(
        text=primary.text,
        words=primary.words,
        alternatives=tuple(alternatives),
    )


def transcribe_audio_detailed(
    audio_path: str,
    preprocessing_profile: str | None = None,
) -> TranscriptionResult:
    """Transcribe audio and retain Vosk word confidence and alternatives.

    Vosk emits per-word confidence in normal MBR mode, while N-best mode emits
    alternative decoder scores without word confidence. When alternatives are
    enabled, two recognizers therefore decode the same PCM using one shared
    cached acoustic model, and their complementary outputs are merged.
    """
    pcm = _read_pcm16(audio_path, preprocessing_profile)
    if not pcm:
        return TranscriptionResult(text="")

    primary = _recognize_pcm(pcm)
    if not primary.text or ASR_MAX_ALTERNATIVES <= 1:
        return TranscriptionResult(
            text=primary.text,
            words=primary.words,
            alternatives=primary.alternatives,
        )

    nbest = _recognize_pcm(pcm, max_alternatives=ASR_MAX_ALTERNATIVES)
    return _merge_results(primary, nbest)


def transcribe_audio(audio_path: str) -> str:
    """Return only the primary transcript for backward compatibility."""
    return transcribe_audio_detailed(audio_path).text
