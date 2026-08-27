"""Small in-memory conversation session for the chatbot runtime."""

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatTurn:
    """One completed user/assistant exchange."""

    user: str
    assistant: str


@dataclass(frozen=True)
class PendingClarification:
    """An ambiguous utterance waiting for the user's selection."""

    original_text: str
    options: tuple[str, ...]


class ChatSession:
    """Keep a bounded number of completed chat turns in memory."""

    def __init__(self, max_turns: int = 6):
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")

        self.max_turns = max_turns
        self._turns = deque(maxlen=max_turns)
        self._pending_clarification = None

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

    @property
    def pending_clarification(self) -> PendingClarification | None:
        return self._pending_clarification

    def set_pending_clarification(
        self,
        original_text: str,
        options: tuple[str, ...],
    ) -> None:
        original_text = str(original_text).strip()
        cleaned_options = tuple(
            str(option).strip() for option in options if str(option).strip()
        )
        if not original_text or len(cleaned_options) < 2:
            raise ValueError("A pending clarification requires text and two options")
        self._pending_clarification = PendingClarification(
            original_text=original_text,
            options=cleaned_options,
        )

    def clear_pending_clarification(self) -> None:
        self._pending_clarification = None

    def clear(self) -> None:
        """Forget all retained turns."""
        self._turns.clear()
        self.clear_pending_clarification()

    def __len__(self) -> int:
        return len(self._turns)
