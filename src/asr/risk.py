"""Risk assessment for deciding when an ASR transcript needs clarification."""

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from config.settings import (
    ASR_ALTERNATIVE_SCORE_GAP,
    ASR_CLARIFICATION_MAX_OPTIONS,
    ASR_WORD_CONFIDENCE_THRESHOLD,
)
from src.asr.transcriber import TranscriptionResult


_PERSIAN_TRANSLATION = str.maketrans(
    {
        "آ": "ا",
        "أ": "ا",
        "إ": "ا",
        "ك": "ک",
        "ي": "ی",
        "ى": "ی",
        "ؤ": "و",
        "ئ": "ی",
        "ة": "ه",
        "ۀ": "ه",
        "\u200c": "",
    }
)


@dataclass(frozen=True)
class ASRRiskAssessment:
    """Result of checking whether task-relevant ASR ambiguity is visible."""

    requires_clarification: bool
    reason: str | None = None
    low_confidence_words: tuple[str, ...] = ()
    options: tuple[str, ...] = ()
    clarification: str | None = None

    def to_dict(self) -> dict:
        return {
            "requires_clarification": self.requires_clarification,
            "reason": self.reason,
            "low_confidence_words": list(self.low_confidence_words),
            "options": list(self.options),
            "clarification": self.clarification,
        }


def _normalize_token(token: str) -> str:
    token = token.translate(_PERSIAN_TRANSLATION).lower()
    return re.sub(r"[^\w]", "", token)


def _eligible_alternatives(transcription: TranscriptionResult):
    scored = [
        alternative.decoder_score
        for alternative in transcription.alternatives
        if alternative.decoder_score is not None
    ]
    best_score = max(scored) if scored else None

    for alternative in transcription.alternatives:
        if alternative.text == transcription.text:
            continue
        if (
            best_score is not None
            and alternative.decoder_score is not None
            and best_score - alternative.decoder_score > ASR_ALTERNATIVE_SCORE_GAP
        ):
            continue
        yield alternative


def _variants_at_low_confidence_words(
    transcription: TranscriptionResult,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    primary_tokens = transcription.text.split()
    normalized_primary = [_normalize_token(token) for token in primary_tokens]
    low_indices = {
        index
        for index, word in enumerate(transcription.words)
        if word.confidence is not None
        and word.confidence < ASR_WORD_CONFIDENCE_THRESHOLD
        and index < len(primary_tokens)
    }
    low_words = tuple(primary_tokens[index] for index in sorted(low_indices))
    if not low_indices:
        return low_words, ()

    variants = []
    for alternative in _eligible_alternatives(transcription):
        alternative_tokens = alternative.text.split()
        normalized_alternative = [
            _normalize_token(token) for token in alternative_tokens
        ]
        matcher = SequenceMatcher(
            None,
            normalized_primary,
            normalized_alternative,
            autojunk=False,
        )

        for operation, primary_start, primary_end, alt_start, alt_end in matcher.get_opcodes():
            if operation == "equal":
                continue
            affected = set(range(primary_start, primary_end))
            if not affected.intersection(low_indices):
                continue

            primary_phrase = " ".join(primary_tokens[primary_start:primary_end]).strip()
            alternative_phrase = " ".join(
                alternative_tokens[alt_start:alt_end]
            ).strip()
            for phrase in (primary_phrase, alternative_phrase):
                normalized_phrase = _normalize_token(phrase)
                if (
                    phrase
                    and normalized_phrase
                    and all(_normalize_token(item) != normalized_phrase for item in variants)
                ):
                    variants.append(phrase)

    return low_words, tuple(variants[:ASR_CLARIFICATION_MAX_OPTIONS])


def assess_asr_risk(transcription: TranscriptionResult) -> ASRRiskAssessment:
    """Request clarification when plausible hypotheses disagree on a weak word."""
    low_words, options = _variants_at_low_confidence_words(transcription)
    if len(options) < 2:
        return ASRRiskAssessment(
            requires_clarification=False,
            low_confidence_words=low_words,
        )

    quoted_options = " یا ".join(f"«{option}»" for option in options)
    return ASRRiskAssessment(
        requires_clarification=True,
        reason="low_confidence_alternative_disagreement",
        low_confidence_words=low_words,
        options=options,
        clarification=f"منظورتان {quoted_options} است؟",
    )
