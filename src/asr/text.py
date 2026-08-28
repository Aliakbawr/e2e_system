"""Conservative text normalization for structured Persian ASR output."""

from __future__ import annotations

import re
import unicodedata

from src.asr.types import RecognizedWord, TranscriptAlternative, TranscriptionResult


_CHARACTER_TRANSLATION = str.maketrans(
    {
        # Arabic code points commonly emitted in otherwise Persian text.
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        # One representation makes numeric comparisons and memory stable.
        **dict(zip("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")),
        # Directional controls and a stray BOM carry no spoken meaning.
        "\ufeff": None,
        "\u202a": None,
        "\u202b": None,
        "\u202c": None,
        "\u202d": None,
        "\u202e": None,
        "\u2066": None,
        "\u2067": None,
        "\u2068": None,
        "\u2069": None,
    }
)


def preprocess_asr_text(text: str) -> str:
    """Canonicalize representation without guessing or correcting words.

    ZWNJ is preserved because it is meaningful Persian orthography. Punctuation
    is also preserved; removing it here could alter named entities or numbers.
    """
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    # Some Persian corpora use directional marks where a ZWNJ was intended.
    # Preserve the word boundary only when the mark is embedded in a word.
    normalized = re.sub(r"(?<=\w)[\u200e\u200f](?=\w)", "\u200c", normalized)
    normalized = normalized.replace("\u200e", "").replace("\u200f", "")
    normalized = normalized.translate(_CHARACTER_TRANSLATION)
    return re.sub(r"\s+", " ", normalized).strip()


def _preprocess_word(word: RecognizedWord) -> RecognizedWord:
    return RecognizedWord(
        text=preprocess_asr_text(word.text),
        confidence=word.confidence,
        start=word.start,
        end=word.end,
    )


def preprocess_transcription(
    transcription: TranscriptionResult,
) -> TranscriptionResult:
    """Normalize every textual view while retaining ASR evidence metadata."""
    alternatives = tuple(
        TranscriptAlternative(
            text=preprocess_asr_text(alternative.text),
            words=tuple(_preprocess_word(word) for word in alternative.words),
            decoder_score=alternative.decoder_score,
        )
        for alternative in transcription.alternatives
    )

    return TranscriptionResult(
        text=preprocess_asr_text(transcription.text),
        words=tuple(_preprocess_word(word) for word in transcription.words),
        alternatives=alternatives,
    )
