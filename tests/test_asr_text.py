import unittest

from src.asr.text import preprocess_asr_text, preprocess_transcription
from src.asr.types import RecognizedWord, TranscriptAlternative, TranscriptionResult


class ASRTextPreprocessingTests(unittest.TestCase):
    def test_normalizes_safe_persian_variants_digits_and_whitespace(self):
        self.assertEqual(
            preprocess_asr_text("  دانشگاه\u200f شريف\tساعت ۱۵ و ١٦  "),
            "دانشگاه شریف ساعت 15 و 16",
        )

    def test_preserves_zwnj_and_punctuation(self):
        self.assertEqual(preprocess_asr_text("می\u200cروم؟"), "می\u200cروم؟")

    def test_interprets_embedded_direction_mark_as_word_boundary(self):
        self.assertEqual(
            preprocess_asr_text("قوانین بین\u200fالمللی"),
            "قوانین بین\u200cالمللی",
        )

    def test_structured_result_keeps_confidence_and_scores(self):
        result = TranscriptionResult(
            text="شريف ۱۵",
            words=(
                RecognizedWord("شريف", confidence=0.4, start=0.1, end=0.4),
                RecognizedWord("۱۵", confidence=0.9, start=0.5, end=0.7),
            ),
            alternatives=(
                TranscriptAlternative("شريف ۱۵", decoder_score=10.0),
                TranscriptAlternative("شریف 15", decoder_score=9.5),
            ),
        )

        processed = preprocess_transcription(result)

        self.assertEqual(processed.text, "شریف 15")
        self.assertEqual([word.text for word in processed.words], ["شریف", "15"])
        self.assertEqual(processed.words[0].confidence, 0.4)
        self.assertEqual(processed.words[0].start, 0.1)
        self.assertEqual(processed.words[0].end, 0.4)
        # Keep every decoder hypothesis and score even when normalization makes
        # two textual forms identical; evidence should not be silently dropped.
        self.assertEqual(len(processed.alternatives), 2)
        self.assertEqual(processed.alternatives[0].decoder_score, 10.0)
        self.assertEqual(processed.alternatives[1].decoder_score, 9.5)

    def test_preprocessing_is_idempotent(self):
        once = preprocess_asr_text("دانشگاه شريف ۱۵")
        self.assertEqual(preprocess_asr_text(once), once)


if __name__ == "__main__":
    unittest.main()
