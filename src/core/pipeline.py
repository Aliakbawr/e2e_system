import logging
import time

from config.settings import AUDIO_OUTPUT_DIR
from src.asr.text import preprocess_transcription
from src.asr.transcriber import transcribe_audio_detailed
from src.audio.player import play_audio
from src.core.dialogue import interpret_transcription
from src.llm.generator import generate_answer
from src.tts.synthesizer import synthesize_speech


logger = logging.getLogger(__name__)


def _log_asr_result(transcription, elapsed):
    scored_words = [
        (word.text, word.confidence)
        for word in transcription.words
        if word.confidence is not None
    ]
    lowest_confidence_words = sorted(scored_words, key=lambda item: item[1])[:5]
    alternatives = [
        {
            "text": alternative.text,
            "decoder_score": alternative.decoder_score,
        }
        for alternative in transcription.alternatives
    ]

    logger.info(
        "asr_completed duration_sec=%.3f mean_word_confidence=%s "
        "word_count=%d alternative_count=%d transcript=%r",
        elapsed,
        (
            f"{transcription.mean_word_confidence:.4f}"
            if transcription.mean_word_confidence is not None
            else "unavailable"
        ),
        len(transcription.words),
        len(transcription.alternatives),
        transcription.text,
    )
    logger.info("asr_lowest_confidence_words values=%s", lowest_confidence_words)
    logger.info("asr_alternatives values=%s", alternatives)


def chat(audio_path, session=None):

    metrics = {}

    logger.info(
        "chat_turn_started audio_path=%s retained_session_turns=%d "
        "pending_clarification=%s",
        audio_path,
        len(session) if session is not None else 0,
        bool(session is not None and session.pending_clarification is not None),
    )

    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    # =====================
    # ASR
    # =====================

    start = time.time()

    raw_transcription = transcribe_audio_detailed(
        audio_path
    )
    transcription = preprocess_transcription(raw_transcription)
    dialogue = interpret_transcription(transcription, session)
    question = dialogue.raw_question
    risk = dialogue.risk
    resolution = dialogue.resolution
    reranking = dialogue.reranking
    asr_details = raw_transcription.to_dict()
    asr_details["preprocessed"] = transcription.to_dict()
    asr_details["preprocessing_changed"] = raw_transcription != transcription
    asr_details["reranking"] = reranking.to_dict() if reranking is not None else None
    asr_details["risk"] = risk.to_dict()

    asr_details["clarification_resolution"] = (
        resolution.to_dict() if resolution is not None else None
    )

    if dialogue.memory_updated and session is not None:
        logger.info(
            "session_memory_updated item_count=%d values=%s",
            len(session.memory),
            session.memory_snapshot(),
        )
    if resolution is not None:
        logger.info(
            "clarification_reply_processed resolved=%s method=%s "
            "selected_option=%r raw_reply=%r resolved_question=%r",
            resolution.resolved,
            resolution.method,
            resolution.selected_option,
            question,
            resolution.resolved_question,
        )

    metrics["asr_time"] = time.time() - start
    metrics["asr_mean_word_confidence"] = transcription.mean_word_confidence
    metrics["asr_alternative_count"] = len(transcription.alternatives)
    metrics["asr_preprocessing_changed"] = raw_transcription != transcription
    metrics["asr_nbest_reranked"] = bool(reranking is not None and reranking.changed)
    metrics["asr_requires_clarification"] = risk.requires_clarification
    metrics["clarification_resolved"] = bool(
        resolution is not None and resolution.resolved
    )
    metrics["session_memory_items"] = len(session.memory) if session is not None else 0
    _log_asr_result(transcription, metrics["asr_time"])
    if raw_transcription != transcription:
        logger.info(
            "asr_preprocessed raw_transcript=%r processed_transcript=%r "
            "raw_alternatives=%s processed_alternatives=%s",
            raw_transcription.text,
            transcription.text,
            [item.text for item in raw_transcription.alternatives],
            [item.text for item in transcription.alternatives],
        )
    logger.info(
        "asr_risk_assessed requires_clarification=%s reason=%s "
        "low_confidence_words=%s options=%s",
        risk.requires_clarification,
        risk.reason,
        risk.low_confidence_words,
        risk.options,
    )
    if reranking is not None and reranking.changed:
        logger.info(
            "asr_nbest_reranked reason=%s original_transcript=%r selected_transcript=%r",
            reranking.reason,
            reranking.original_text,
            reranking.selected_text,
        )


    print("\nUSER:")
    print(question)


    # =====================
    # Empty input handling
    # =====================

    if dialogue.action == "retry":

        logger.warning("asr_empty_transcript action=request_retry")

        answer = (
            "I did not receive any speech. "
            "Please try again."
        )

        output_audio = AUDIO_OUTPUT_DIR / "response.wav"

        tts_result = synthesize_speech(
            answer,
            str(output_audio)
        )

        metrics["tts_time"] = (
            tts_result["tts_time"]
        )

        logger.info(
            "tts_completed duration_sec=%.3f audio_duration_sec=%.3f rtf=%.3f",
            metrics["tts_time"],
            tts_result["audio_duration"],
            tts_result["tts_rtf"],
        )

        play_audio(
            str(output_audio)
        )


        return {
            "question": question,
            "asr": asr_details,
            "decision": "retry",
            "memory": session.memory_snapshot() if session is not None else [],
            "answer": answer,
            "audio": str(output_audio),
            "metrics": metrics
        }


    # =====================
    # Targeted clarification
    # =====================

    if dialogue.action == "clarify":
        answer = risk.clarification
        logger.warning(
            "asr_clarification_requested reason=%s low_confidence_words=%s "
            "options=%s",
            risk.reason,
            risk.low_confidence_words,
            risk.options,
        )

        output_audio = AUDIO_OUTPUT_DIR / "response.wav"
        tts_result = synthesize_speech(answer, str(output_audio))
        metrics["tts_time"] = tts_result["tts_time"]
        logger.info(
            "tts_completed duration_sec=%.3f audio_duration_sec=%.3f rtf=%.3f",
            metrics["tts_time"],
            tts_result["audio_duration"],
            tts_result["tts_rtf"],
        )
        play_audio(str(output_audio))
        logger.info("chat_turn_completed decision=clarify metrics=%s", metrics)

        return {
            "question": question,
            "asr": asr_details,
            "decision": "clarify",
            "memory": session.memory_snapshot() if session is not None else [],
            "answer": answer,
            "audio": str(output_audio),
            "metrics": metrics,
        }


    # =====================
    # LLM
    # =====================

    llm_question = dialogue.effective_question

    llm_result = generate_answer(
        llm_question,
        history=session.messages() if session is not None else None,
        memory=session.memory_prompt() if session is not None else None,
        turn_context=(
            "این پیام از پاسخ کاربر به رفع ابهام بازسازی شده است. "
            "در پاسخ، واژه، عدد یا نفی تأییدشده را صریحاً تکرار کن."
            if resolution is not None and resolution.resolved
            else None
        ),
    )

    answer = llm_result["answer"]

    if session is not None and answer:
        session.add_turn(llm_question, answer)


    print("\nBOT:")
    print(answer)


    metrics["llm_time"] = (
        llm_result["latency"]
    )

    logger.info(
        "llm_completed duration_sec=%.3f output_tokens=%d "
        "retained_session_turns=%d query=%r answer=%r",
        metrics["llm_time"],
        llm_result["tokens"],
        len(session) if session is not None else 0,
        llm_question,
        answer,
    )


    # =====================
    # TTS
    # =====================

    output_audio = AUDIO_OUTPUT_DIR / "response.wav"


    tts_result = synthesize_speech(
        answer,
        str(output_audio)
    )


    metrics["tts_time"] = (
        tts_result["tts_time"]
    )

    logger.info(
        "tts_completed duration_sec=%.3f audio_duration_sec=%.3f rtf=%.3f",
        metrics["tts_time"],
        tts_result["audio_duration"],
        tts_result["tts_rtf"],
    )


    # =====================
    # Audio playback
    # =====================

    play_audio(
        str(output_audio)
    )

    logger.info("chat_turn_completed metrics=%s", metrics)


    return {

        "question": question,

        "asr": asr_details,

        "decision": "answer",

        "resolved_question": llm_question,

        "memory": session.memory_snapshot() if session is not None else [],

        "answer": answer,

        "audio": str(output_audio),

        "metrics": metrics

    }
