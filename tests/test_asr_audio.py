import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.asr.audio import prepare_audio_file


class ASRAudioPreprocessingTests(unittest.TestCase):
    def _write(self, audio: np.ndarray, sample_rate: int = 16000) -> Path:
        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        handle.close()
        path = Path(handle.name)
        sf.write(path, audio, sample_rate, subtype="FLOAT")
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_none_profile_preserves_pcm_conversion(self):
        audio = np.array([0.0, 0.25, -0.25], dtype=np.float32)
        prepared = prepare_audio_file(self._write(audio), 16000, "none")
        self.assertFalse(prepared.applied)
        self.assertEqual(np.frombuffer(prepared.pcm16, dtype="<i2").tolist(), [0, 8191, -8191])

    def test_repairs_only_abnormally_low_level_audio(self):
        sample_rate = 16000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        quiet = 0.001 * np.sin(2 * np.pi * 220 * time)
        prepared = prepare_audio_file(self._write(quiet), sample_rate, "low_level_gain")
        self.assertTrue(prepared.applied)
        self.assertGreater(prepared.gain_db, 20.0)
        self.assertAlmostEqual(prepared.output_rms_dbfs, -24.0, delta=0.2)

        normal = 0.08 * np.sin(2 * np.pi * 220 * time)
        untouched = prepare_audio_file(self._write(normal), sample_rate, "low_level_gain")
        self.assertFalse(untouched.applied)
        self.assertEqual(untouched.gain_db, 0.0)

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare_audio_file(self._write(np.zeros(16, dtype=np.float32)), 16000, "magic")


if __name__ == "__main__":
    unittest.main()
