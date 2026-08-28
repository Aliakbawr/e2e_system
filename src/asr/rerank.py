"""Constrained N-best selection using explicit conversation corrections."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from config.settings import ASR_ALTERNATIVE_SCORE_GAP
from src.asr.types import TranscriptionResult

if TYPE_CHECKING:
    from src.core.session import ChatSession


_TRANSLATION = str.maketrans(
    {"آ": "ا", "أ": "ا", "إ": "ا", "ك": "ک", "ي": "ی", "ى": "ی", "\u200c": ""}
)
_STOPWORDS = {
    "از",
    "به",
    "در",
    "را",
    "رو",
    "و",
    "یا",
    "این",
    "آن",
    "یک",
    "چه",
    "چیه",
    "است",
    "هست",
    "بگو",
    "درباره",
}


@dataclass(frozen=True)
class NBestReranking:
    transcription: TranscriptionResult
    changed: bool = False
    reason: str | None = None
    original_text: str | None = None
    selected_text: str | None = None

    def to_dict(self) -> dict:
        return {
            "changed": self.changed,
            "reason": self.reason,
            "original_text": self.original_text,
            "selected_text": self.selected_text,
        }


def _tokens(text: str) -> set[str]:
    normalized = str(text).translate(_TRANSLATION).lower()
    return {
        token
        for token in re.findall(r"\w+", normalized)
        if token and token not in _STOPWORDS
    }


def _context_matches(question: str, context: str, key: str, value: str) -> bool:
    question_tokens = _tokens(question) - _tokens(key) - _tokens(value)
    context_tokens = _tokens(context) - _tokens(key) - _tokens(value)
    return bool(question_tokens & context_tokens)


def _eligible_alternatives(transcription: TranscriptionResult):
    scores = [
        item.decoder_score
        for item in transcription.alternatives
        if item.decoder_score is not None
    ]
    best_score = max(scores) if scores else None
    for item in transcription.alternatives:
        if item.text == transcription.text:
            continue
        if (
            best_score is not None
            and item.decoder_score is not None
            and best_score - item.decoder_score > ASR_ALTERNATIVE_SCORE_GAP
        ):
            continue
        yield item


def rerank_with_session_memory(
    transcription: TranscriptionResult,
    session: ChatSession | None,
) -> NBestReranking:
    """Select an actual hypothesis only when explicit correction memory supports it."""
    if session is None or not transcription.text:
        return NBestReranking(transcription=transcription)

    primary_tokens = _tokens(transcription.text)
    candidates = list(_eligible_alternatives(transcription))
    for memory in reversed(session.memory):
        if memory.kind != "correction" or memory.source != "explicit_user_clarification":
            continue
        key_tokens = _tokens(memory.key)
        value_tokens = _tokens(memory.value)
        if not key_tokens or not value_tokens or not key_tokens <= primary_tokens:
            continue
        if not _context_matches(
            transcription.text, memory.context, memory.key, memory.value
        ):
            continue
        supported = [
            item
            for item in candidates
            if value_tokens <= _tokens(item.text)
            and not key_tokens <= _tokens(item.text)
        ]
        if not supported:
            continue
        selected = max(
            supported,
            key=lambda item: (
                item.decoder_score is not None,
                item.decoder_score if item.decoder_score is not None else float("-inf"),
            ),
        )
        return NBestReranking(
            transcription=TranscriptionResult(
                text=selected.text,
                words=selected.words,
                # Explicitly confirmed memory resolves this ambiguity. Retaining
                # competing alternatives here would immediately ask the same
                # clarification again for critical names or numbers.
                alternatives=(selected,),
            ),
            changed=True,
            reason="explicit_correction_memory",
            original_text=transcription.text,
            selected_text=selected.text,
        )
    return NBestReranking(transcription=transcription)
