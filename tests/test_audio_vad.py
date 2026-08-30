import unittest

import numpy as np

from src.audio.vad import UtteranceEndpointDetector


class UtteranceEndpointDetectorTests(unittest.TestCase):
    def _detector(self):
        return UtteranceEndpointDetector(
            sample_rate=16000,
            chunk_samples=512,
            threshold=0.5,
            min_speech_ms=64,
            min_silence_ms=96,
            pre_roll_ms=64,
            post_roll_ms=32,
            max_utterance_sec=10,
        )

    def test_preserves_pre_roll_and_trims_excess_endpoint_silence(self):
        detector = self._detector()
        frame = np.ones(512, dtype=np.float32)

        self.assertIsNone(detector.process(frame * 0, 0.0))
        self.assertIsNone(detector.process(frame * 1, 0.9))
        self.assertEqual(detector.process(frame * 2, 0.9), "start")
        self.assertIsNone(detector.process(frame * 3, 0.9))
        self.assertIsNone(detector.process(frame * 4, 0.1))
        self.assertIsNone(detector.process(frame * 5, 0.1))
        self.assertEqual(detector.process(frame * 6, 0.1), "end")

        captured = detector.audio().reshape(-1, 512)
        self.assertEqual(captured.shape[0], 4)
        np.testing.assert_array_equal(captured[:, 0], [1, 2, 3, 4])

    def test_short_noise_does_not_start_an_utterance(self):
        detector = self._detector()
        frame = np.zeros(512, dtype=np.float32)

        self.assertIsNone(detector.process(frame, 0.9))
        self.assertIsNone(detector.process(frame, 0.1))
        self.assertFalse(detector.speaking)
        self.assertEqual(detector.audio().size, 0)


if __name__ == "__main__":
    unittest.main()
