from src.core.pipeline import chat
from src.audio.push_to_talk import record_until_enter


def main():
    while True:
        try:
            audio_path = record_until_enter()

            result = chat(audio_path)

            print("\n🧠 You said:", result["question"])
            print("🤖 Answer:", result["answer"])
            print("⏱", result["metrics"])

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Exiting...")
            break


if __name__ == "__main__":
    main()
