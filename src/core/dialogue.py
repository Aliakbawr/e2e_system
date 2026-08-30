"""Shared, text-only dialogue control for runtime and evaluation."""

from dataclasses import dataclass

from src.asr.risk import (
    ASRRiskAssessment,
    ClarificationResolution,
    assess_asr_risk,
    resolve_clarification_transcription,
)
from src.asr.rerank import NBestReranking, rerank_with_session_memory
from src.asr.text import preprocess_transcription
from src.asr.types import TranscriptionResult
from src.core.session import ChatSession


@dataclass(frozen=True)
class DialogueDecision:
    """Control decision made before LLM generation."""

    action: str
    raw_question: str
    effective_question: str
    risk: ASRRiskAssessment
    resolution: ClarificationResolution | None = None
    memory_updated: bool = False
    reranking: NBestReranking | None = None


def interpret_transcription(
    transcription: TranscriptionResult,
    session: ChatSession | None = None,
) -> DialogueDecision:
    """Apply clarification, correction, and memory rules to one transcript."""
    transcription = preprocess_transcription(transcription)
    reranking = rerank_with_session_memory(transcription, session)
    transcription = reranking.transcription
    question = transcription.text.strip()
    risk = assess_asr_risk(transcription)
    resolution = None
    memory_updated = False

    if session is not None and session.pending_clarification is not None:
        pending = session.pending_clarification
        if question:
            resolution = resolve_clarification_transcription(
                pending.original_text,
                pending.options,
                transcription,
            )
            if resolution.resolved:
                session.remember_correction(
                    original=pending.options[0],
                    corrected=resolution.selected_option,
                    context=resolution.resolved_question,
                )
                memory_updated = True
            session.clear_pending_clarification()

    if not question:
        action = "retry"
        effective_question = ""
    elif resolution is not None and resolution.resolved:
        action = "answer"
        effective_question = resolution.resolved_question
    elif risk.requires_clarification:
        action = "clarify"
        effective_question = question
        if session is not None:
            session.add_turn(question, risk.clarification)
            session.set_pending_clarification(question, risk.options)
    else:
        action = "answer"
        effective_question = question

    return DialogueDecision(
        action=action,
        raw_question=question,
        effective_question=effective_question,
        risk=risk,
        resolution=resolution,
        memory_updated=memory_updated,
        reranking=reranking,
    )
