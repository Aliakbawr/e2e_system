from src.audio.microphone import record_until_enter
from src.core.pipeline import chat


def main():
    while True:

        try:

            audio_path = record_until_enter()
            if audio_path is None:
                continue

            result = chat(
                audio_path
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

            print("\nExiting assistant...")
            break


if __name__ == "__main__":
    main()
