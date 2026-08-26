from src.core.pipeline import chat
from src.core.session import ChatSession
from src.audio.push_to_talk import record_until_enter


def main():
    session = ChatSession()

    while True:
        try:
            audio_path = record_until_enter()

            result = chat(audio_path, session=session)

            print("\n🧠 You said:", result["question"])
            print("🤖 Answer:", result["answer"])
            print("⏱", result["metrics"])

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting...")
            break


if __name__ == "__main__":
    main()
