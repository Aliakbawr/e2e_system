import unittest

import numpy as np

from benchmark.asr_parameter_selection.common import choose_development_configuration
from src.asr.audio import _low_level_gain


class ASRParameterSelectionTests(unittest.TestCase):
    def test_custom_audio_parameters_control_gain(self):
        audio = np.full(160, 0.001, dtype=np.float32)
        unchanged, gain = _low_level_gain(audio, threshold_dbfs=-70.0)
        self.assertEqual(gain, 0.0)
        np.testing.assert_array_equal(unchanged, audio)

        enhanced, gain = _low_level_gain(
            audio, threshold_dbfs=-50.0, target_rms_dbfs=-30.0,
            max_gain_db=20.0, peak_headroom_dbfs=-1.0,
        )
        self.assertGreater(gain, 0.0)
        self.assertGreater(float(np.max(enhanced)), float(np.max(audio)))

    def test_selection_uses_only_development_and_constraints(self):
        rows = [
            {"split": "development", "word_confidence_threshold": 0.55, "alternative_score_gap": 0.5, "oracle_answer_span_rate": 0.8, "oracle_corpus_wer": 0.2, "clarification_rate": 0.08, "unnecessary_clarification_rate": 0.02},
            {"split": "development", "word_confidence_threshold": 0.85, "alternative_score_gap": 3.0, "oracle_answer_span_rate": 0.99, "oracle_corpus_wer": 0.1, "clarification_rate": 0.50, "unnecessary_clarification_rate": 0.20},
            {"split": "held_out", "word_confidence_threshold": 0.75, "alternative_score_gap": 2.0, "oracle_answer_span_rate": 1.0, "oracle_corpus_wer": 0.0, "clarification_rate": 0.01, "unnecessary_clarification_rate": 0.0},
        ]
        selected = choose_development_configuration(
            rows, max_clarification_rate=0.10, max_unnecessary_rate=0.05
        )
        self.assertEqual(selected["word_confidence_threshold"], 0.55)


if __name__ == "__main__":
    unittest.main()
