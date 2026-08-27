"""Dependency-free structured ASR result types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RecognizedWord:
    """One recognized word and its confidence/timing metadata."""

    text: str
    confidence: float | None = None
    start: float | None = None
    end: float | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class TranscriptAlternative:
    """One ASR decoding hypothesis."""

    text: str
    words: tuple[RecognizedWord, ...] = ()
    decoder_score: float | None = None

    def to_dict(self, include_words: bool = False) -> dict:
        result = {
            "text": self.text,
            "decoder_score": self.decoder_score,
        }
        if include_words:
            result["words"] = [word.to_dict() for word in self.words]
        return result


@dataclass(frozen=True)
class TranscriptionResult:
    """Primary transcript plus confidence evidence and N-best hypotheses."""

    text: str
    words: tuple[RecognizedWord, ...] = ()
    alternatives: tuple[TranscriptAlternative, ...] = ()

    @property
    def mean_word_confidence(self) -> float | None:
        confidences = [
            word.confidence for word in self.words if word.confidence is not None
        ]
        if not confidences:
            return None
        return sum(confidences) / len(confidences)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "mean_word_confidence": self.mean_word_confidence,
            "words": [word.to_dict() for word in self.words],
            "alternatives": [
                alternative.to_dict() for alternative in self.alternatives
            ],
        }
