import time

from config.settings import AUDIO_OUTPUT_DIR
from src.asr.transcriber import transcribe_audio
from src.audio.player import play_audio
from src.llm.generator import generate_answer
from src.tts.synthesizer import synthesize_speech


def chat(audio_path, session=None):

    metrics = {}

    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    # =====================
    # ASR
    # =====================

    start = time.time()

    question = transcribe_audio(
        audio_path
    )

    metrics["asr_time"] = time.time() - start


    print("\nUSER:")
    print(question)


    # =====================
    # Empty input handling
    # =====================

    if not question.strip():

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

        play_audio(
            str(output_audio)
        )


        return {
            "question": question,
            "answer": answer,
            "audio": str(output_audio),
            "metrics": metrics
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


    # =====================
    # Audio playback
    # =====================

    play_audio(
        str(output_audio)
    )


    return {

        "question": question,

        "answer": answer,

        "audio": str(output_audio),

        "metrics": metrics

    }
