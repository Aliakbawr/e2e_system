import unittest

from src.tts.text_stream import PhraseBuffer


class PhraseBufferTests(unittest.TestCase):
    def test_emits_complete_persian_sentences_across_fragments(self):
        buffer = PhraseBuffer()

        self.assertEqual(buffer.feed("این یک پاسخ"), [])
        self.assertEqual(buffer.feed(" کوتاه است. جمله"), ["این یک پاسخ کوتاه است."])
        self.assertEqual(buffer.finish(), ["جمله"])

    def test_uses_a_long_comma_as_a_soft_boundary(self):
        buffer = PhraseBuffer(soft_min_chars=10, max_chars=30)

        self.assertEqual(buffer.feed("این بخش به اندازه کافی بلند است، ادامه"), [
            "این بخش به اندازه کافی بلند است،"
        ])
        self.assertEqual(buffer.finish(), ["ادامه"])

    def test_forces_a_word_boundary_when_punctuation_is_missing(self):
        buffer = PhraseBuffer(soft_min_chars=10, max_chars=20)

        phrases = buffer.feed("یک عبارت نسبتا طولانی بدون هیچ نشانه")

        self.assertEqual(phrases, ["یک عبارت نسبتا", "طولانی بدون هیچ"])
        self.assertEqual(buffer.finish(), ["نشانه"])

    def test_stops_at_first_newline_to_match_existing_answer_cleanup(self):
        buffer = PhraseBuffer()

        self.assertEqual(buffer.feed("پاسخ اصلی\nمتن اضافی."), [])
        self.assertEqual(buffer.finish(), ["پاسخ اصلی"])
        self.assertEqual(buffer.feed("نادیده گرفته شود."), [])


if __name__ == "__main__":
    unittest.main()
