import logging
import time

from config.settings import AUDIO_OUTPUT_DIR
from src.asr.risk import assess_asr_risk
from src.asr.transcriber import transcribe_audio_detailed
from src.audio.player import play_audio
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
        "chat_turn_started audio_path=%s retained_session_turns=%d",
        audio_path,
        len(session) if session is not None else 0,
    )

    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    # =====================
    # ASR
    # =====================

    start = time.time()

    transcription = transcribe_audio_detailed(
        audio_path
    )
    question = transcription.text
    asr_details = transcription.to_dict()
    risk = assess_asr_risk(transcription)
    asr_details["risk"] = risk.to_dict()

    metrics["asr_time"] = time.time() - start
    metrics["asr_mean_word_confidence"] = transcription.mean_word_confidence
    metrics["asr_alternative_count"] = len(transcription.alternatives)
    metrics["asr_requires_clarification"] = risk.requires_clarification
    _log_asr_result(transcription, metrics["asr_time"])
    logger.info(
        "asr_risk_assessed requires_clarification=%s reason=%s "
        "low_confidence_words=%s options=%s",
        risk.requires_clarification,
        risk.reason,
        risk.low_confidence_words,
        risk.options,
    )


    print("\nUSER:")
    print(question)


    # =====================
    # Empty input handling
    # =====================

    if not question.strip():

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
            "answer": answer,
            "audio": str(output_audio),
            "metrics": metrics
        }


    # =====================
    # Targeted clarification
    # =====================

    if risk.requires_clarification:
        answer = risk.clarification
        logger.warning(
            "asr_clarification_requested reason=%s low_confidence_words=%s "
            "options=%s",
            risk.reason,
            risk.low_confidence_words,
            risk.options,
        )

        if session is not None:
            session.add_turn(question, answer)

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
            "answer": answer,
            "audio": str(output_audio),
            "metrics": metrics,
        }


    # =====================
    # LLM
    # =====================

    llm_result = generate_answer(
        question,
        history=session.messages() if session is not None else None,
    )

    answer = llm_result["answer"]

    if session is not None and answer:
        session.add_turn(question, answer)


    print("\nBOT:")
    print(answer)


    metrics["llm_time"] = (
        llm_result["latency"]
    )

    logger.info(
        "llm_completed duration_sec=%.3f output_tokens=%d "
        "retained_session_turns=%d answer=%r",
        metrics["llm_time"],
        llm_result["tokens"],
        len(session) if session is not None else 0,
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

        "answer": answer,

        "audio": str(output_audio),

        "metrics": metrics

    }
