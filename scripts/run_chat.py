import logging

from src.audio.microphone import record_until_enter
from src.core.pipeline import chat
from src.core.session import ChatSession
from src.utils.logging import setup_logging


logger = logging.getLogger(__name__)


def main():
    setup_logging()
    session = ChatSession()

    while True:

        try:

            audio_path = record_until_enter()
            if audio_path is None:
                continue

            result = chat(
                audio_path,
                session=session,
            )


            print("\n====================")

            print("Question:")
            print(result["question"])


            print("\nAnswer:")
            print(result["answer"])


            print("\nMetrics:")
            print(result["metrics"])

            print("====================")


        except (KeyboardInterrupt, EOFError):

            logger.info("chat_session_stopped retained_turns=%d", len(session))
            print("\nExiting assistant...")
            break

        except Exception:
            logger.exception("chat_turn_failed")
            print("\nAn error occurred. Please check the log and try again.")


if __name__ == "__main__":
    main()
