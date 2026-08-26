import logging

from src.core.pipeline import chat
from src.core.session import ChatSession
from src.audio.push_to_talk import record_until_enter
from src.utils.logging import setup_logging


logger = logging.getLogger(__name__)


def main():
    setup_logging()
    session = ChatSession()

    while True:
        try:
            audio_path = record_until_enter()

            result = chat(audio_path, session=session)

            print("\n🧠 You said:", result["question"])
            print("🤖 Answer:", result["answer"])
            print("⏱", result["metrics"])

        except (KeyboardInterrupt, EOFError):
            logger.info("chat_session_stopped retained_turns=%d", len(session))
            print("\n👋 Exiting...")
            break

        except Exception:
            logger.exception("chat_turn_failed")
            print("\n⚠️ An error occurred. Check the log and try again.")


if __name__ == "__main__":
    main()
