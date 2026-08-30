"""Incremental text cleanup and phrase segmentation for streaming TTS."""

import re


class PhraseBuffer:
    """Turn arbitrary streamed text fragments into speakable phrases."""

    HARD_BOUNDARIES = frozenset(".!?؟؛")
    SOFT_BOUNDARIES = frozenset("،,")

    def __init__(self, soft_min_chars: int = 40, max_chars: int = 80):
        if soft_min_chars <= 0 or max_chars < soft_min_chars:
            raise ValueError("Invalid streaming phrase lengths")
        self.soft_min_chars = soft_min_chars
        self.max_chars = max_chars
        self._buffer = ""
        self._finished = False

    def feed(self, fragment: str) -> list[str]:
        if self._finished or not fragment:
            return []

        text = str(fragment).replace("*", "").replace("#", "")
        if "\n" in text:
            text = text.split("\n", 1)[0]
            self._finished = True
        self._buffer += text
        return self._extract_ready()

    def finish(self) -> list[str]:
        self._finished = True
        phrases = self._extract_ready()
        remainder = self._clean(self._buffer)
        self._buffer = ""
        if remainder:
            phrases.append(remainder)
        return phrases

    def _extract_ready(self) -> list[str]:
        phrases = []
        while self._buffer:
            boundary = self._find_boundary()
            if boundary is None:
                break

            phrase = self._clean(self._buffer[:boundary])
            self._buffer = self._buffer[boundary:].lstrip()
            if phrase:
                phrases.append(phrase)
        return phrases

    def _find_boundary(self) -> int | None:
        for index, char in enumerate(self._buffer):
            position = index + 1
            if char in self.HARD_BOUNDARIES:
                return position
            if char in self.SOFT_BOUNDARIES and position >= self.soft_min_chars:
                return position

        if len(self._buffer) < self.max_chars:
            return None

        split_at = self._buffer.rfind(" ", self.soft_min_chars, self.max_chars + 1)
        return split_at + 1 if split_at >= 0 else self.max_chars

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()
