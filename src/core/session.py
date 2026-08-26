"""Small in-memory conversation session for the chatbot runtime."""

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatTurn:
    """One completed user/assistant exchange."""

    user: str
    assistant: str


class ChatSession:
    """Keep a bounded number of completed chat turns in memory."""

    def __init__(self, max_turns: int = 6):
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")

        self.max_turns = max_turns
        self._turns = deque(maxlen=max_turns)

    @property
    def turns(self) -> tuple[ChatTurn, ...]:
        """Return an immutable snapshot of the retained turns."""
        return tuple(self._turns)

    def add_turn(self, user: str, assistant: str) -> None:
        """Store a completed, non-empty exchange."""
        user = str(user).strip()
        assistant = str(assistant).strip()

        if not user or not assistant:
            raise ValueError("Both user and assistant messages must be non-empty")

        self._turns.append(ChatTurn(user=user, assistant=assistant))

    def messages(self) -> list[dict[str, str]]:
        """Return retained turns in the format expected by chat templates."""
        messages = []
        for turn in self._turns:
            messages.extend(
                [
                    {"role": "user", "content": turn.user},
                    {"role": "assistant", "content": turn.assistant},
                ]
            )
        return messages

    def clear(self) -> None:
        """Forget all retained turns."""
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
