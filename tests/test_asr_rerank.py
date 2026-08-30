import unittest

from src.asr.rerank import rerank_with_session_memory
from src.asr.risk import assess_asr_risk
from src.asr.types import RecognizedWord, TranscriptAlternative, TranscriptionResult
from src.core.dialogue import interpret_transcription
from src.core.session import ChatSession


def _shaft_transcription() -> TranscriptionResult:
    return TranscriptionResult(
        text="شرط ماشین چیه",
        words=(
            RecognizedWord("شرط", 0.39),
            RecognizedWord("ماشین", 0.99),
            RecognizedWord("چیه", 0.95),
        ),
        alternatives=(
            TranscriptAlternative("شرط ماشین چیه", decoder_score=100.0),
            TranscriptAlternative("شفت ماشین چیه", decoder_score=99.4),
        ),
    )


class ASRNBestRerankingTests(unittest.TestCase):
    def test_confirmed_contextual_correction_selects_actual_alternative(self):
        session = ChatSession()
        session.remember_correction("شرط", "شفت", "شفت ماشین چیه")

        result = rerank_with_session_memory(_shaft_transcription(), session)

        self.assertTrue(result.changed)
        self.assertEqual(result.transcription.text, "شفت ماشین چیه")
        self.assertEqual(result.reason, "explicit_correction_memory")
        self.assertFalse(assess_asr_risk(result.transcription).requires_clarification)

        decision = interpret_transcription(_shaft_transcription(), session)
        self.assertEqual(decision.action, "answer")
        self.assertEqual(decision.effective_question, "شفت ماشین چیه")

    def test_no_memory_keeps_primary(self):
        result = rerank_with_session_memory(_shaft_transcription(), ChatSession())
        self.assertFalse(result.changed)
        self.assertEqual(result.transcription.text, "شرط ماشین چیه")

    def test_irrelevant_correction_context_does_not_leak(self):
        session = ChatSession()
        session.remember_correction("شرط", "شفت", "قطعه صنعتی کارخانه")
        result = rerank_with_session_memory(_shaft_transcription(), session)
        self.assertFalse(result.changed)

    def test_high_confidence_name_disagreement_is_critical(self):
        transcription = TranscriptionResult(
            text="درباره دانشگاه شریعت بگو",
            words=tuple(
                RecognizedWord(text, confidence)
                for text, confidence in (
                    ("درباره", 0.99),
                    ("دانشگاه", 0.99),
                    ("شریعت", 0.91),
                    ("بگو", 0.98),
                )
            ),
            alternatives=(
                TranscriptAlternative("درباره دانشگاه شریعت بگو", decoder_score=65.0),
                TranscriptAlternative("درباره دانشگاه شریف بگو", decoder_score=64.3),
            ),
        )
        risk = assess_asr_risk(transcription)
        self.assertTrue(risk.requires_clarification)
        self.assertEqual(risk.reason, "critical_slot_alternative_disagreement")
        self.assertEqual(risk.options, ("شریعت", "شریف"))

    def test_politeness_disagreement_is_not_critical(self):
        transcription = TranscriptionResult(
            text="لطفا آدرس دانشگاه شریف رو بگو",
            words=(RecognizedWord("لطفا", 0.35),),
            alternatives=(
                TranscriptAlternative("لطفا آدرس دانشگاه شریف رو بگو", decoder_score=70.0),
                TranscriptAlternative("خواهشا آدرس دانشگاه شریف رو بگو", decoder_score=69.4),
            ),
        )
        risk = assess_asr_risk(transcription)
        self.assertFalse(risk.requires_clarification)

    def test_noisy_nbest_reply_resolves_original_question_without_looping(self):
        session = ChatSession()
        session.set_pending_clarification(
            "اگر من دو سیب داشته باشم و یکی را بخورم چند سیب دارم",
            ("سیب", "سیم"),
        )
        transcription = TranscriptionResult(
            text="سیبه",
            words=(RecognizedWord("سیبه", 0.51),),
            alternatives=(
                TranscriptAlternative("سیبه", decoder_score=56.32),
                TranscriptAlternative("سیب", decoder_score=56.08),
                TranscriptAlternative("سیبل", decoder_score=53.99),
            ),
        )

        decision = interpret_transcription(transcription, session)

        self.assertEqual(decision.action, "answer")
        self.assertEqual(
            decision.effective_question,
            "اگر من دو سیب داشته باشم و یکی را بخورم چند سیب دارم",
        )
        self.assertTrue(decision.resolution.resolved)
        self.assertEqual(decision.resolution.selected_option, "سیب")
        self.assertEqual(decision.resolution.method, "nbest_option_text")
        self.assertIsNone(session.pending_clarification)

    def test_repeated_noncritical_word_suppresses_spurious_clarification(self):
        transcription = TranscriptionResult(
            text="اگر دو سیب داشته باشم و یکی از سیب‌ها را بخورم",
            words=(
                RecognizedWord("اگر", 0.99),
                RecognizedWord("دو", 0.99),
                RecognizedWord("سیب", 0.50),
                RecognizedWord("داشته", 0.99),
                RecognizedWord("باشم", 0.99),
                RecognizedWord("و", 0.99),
                RecognizedWord("یکی", 0.99),
                RecognizedWord("از", 0.99),
                RecognizedWord("سیب‌ها", 0.99),
                RecognizedWord("را", 0.99),
                RecognizedWord("بخورم", 0.99),
            ),
            alternatives=(
                TranscriptAlternative(
                    "اگر دو سیب داشته باشم و یکی از سیب‌ها را بخورم",
                    decoder_score=100.0,
                ),
                TranscriptAlternative(
                    "اگر دو سیم داشته باشم و یکی از سیب‌ها را بخورم",
                    decoder_score=99.5,
                ),
            ),
        )

        self.assertFalse(assess_asr_risk(transcription).requires_clarification)


if __name__ == "__main__":
    unittest.main()
