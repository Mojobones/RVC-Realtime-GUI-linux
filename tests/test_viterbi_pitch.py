import types
import unittest

import numpy as np

from engine.settings import EngineSettings, PERFORMANCE_DEFAULT_KEYS
from infer import rmvpe
from infer.rmvpe import VITERBI_MAX_JUMP_BINS, viterbi_bins, voiced_viterbi_bins

BINS = 360


def salience_for(path, width=1.5, floor=0.01, seed=0):
    """RMVPE-like salience: a bump at each frame's bin over a little noise."""
    rng = np.random.default_rng(seed)
    grid = np.arange(BINS)
    rows = [np.exp(-0.5 * ((grid - center) / width) ** 2) for center in path]
    return np.clip(np.array(rows) + floor * rng.random((len(path), BINS)), 0, 1)


class ViterbiBinsTest(unittest.TestCase):
    def test_follows_a_moving_pitch(self):
        path = np.round(150 + 20 * np.sin(np.arange(80) / 6)).astype(int)
        np.testing.assert_array_equal(viterbi_bins(salience_for(path)), path)

    def test_ignores_a_single_frame_octave_flip(self):
        path = np.full(40, 100)
        salience = salience_for(path)
        salience[20] = salience_for([160])[0] * 1.0 + 0.6 * salience[20]  # octave up wins this frame
        self.assertEqual(int(np.argmax(salience[20])), 160)
        self.assertTrue((viterbi_bins(salience) == 100).all())

    def test_an_instant_octave_jump_becomes_a_quick_glide(self):
        # The best path spreads an impossible instant octave over a few
        # frames on both sides of it: a glide of about 70 ms, starting at
        # most 50 ms early and settled within 20 ms.
        path = np.r_[np.full(30, 100), np.full(30, 160)]
        found = viterbi_bins(salience_for(path))
        self.assertTrue((found[:25] == 100).all())
        self.assertTrue((found[32:] == 160).all())

    def test_ordinary_jumps_are_nearly_instant(self):
        path = np.r_[np.full(30, 100), np.full(30, 112)]  # 2.4 semitones
        found = viterbi_bins(salience_for(path))
        self.assertTrue((found[:29] == 100).all())
        self.assertTrue((found[31:] == 112).all())

    def test_moves_are_limited_per_frame(self):
        path = np.r_[np.full(10, 50), np.full(10, 250)]
        found = viterbi_bins(salience_for(path))
        self.assertLessEqual(int(np.abs(np.diff(found)).max()), VITERBI_MAX_JUMP_BINS - 1)


class VoicedRunsTest(unittest.TestCase):
    def test_restarts_after_an_unvoiced_gap(self):
        salience = salience_for(np.r_[np.full(20, 80), np.full(10, 0), np.full(20, 240)])
        salience[20:30] = 0.001  # unvoiced: nothing above the threshold
        found = voiced_viterbi_bins(salience, thred=0.03)
        self.assertTrue((found[:20] == 80).all())
        # No climb from 80: the new note is right from its first frame.
        self.assertTrue((found[30:] == 240).all())

    def test_decode_without_viterbi_is_unchanged(self):
        detector = rmvpe.RMVPE.__new__(rmvpe.RMVPE)
        detector.cents_mapping = np.pad(20 * np.arange(360) + 1997.3794084376191, (4, 4))
        salience = salience_for(np.full(30, 120)).astype(np.float32)
        salience[10] = salience_for([180])[0] * 1.0 + 0.6 * salience[10]
        plain = detector.decode(salience, thred=0.03)
        smoothed = detector.decode(salience, thred=0.03, viterbi=True)
        self.assertGreater(plain[10], plain[9] * 1.5)  # the flip
        self.assertAlmostEqual(smoothed[10], smoothed[9], delta=1.0)
        np.testing.assert_allclose(np.delete(plain, 10), np.delete(smoothed, 10))


class PitchSmoothingSettingTest(unittest.TestCase):
    def test_default_on_and_resettable(self):
        self.assertTrue(EngineSettings().pitch_smoothing)
        self.assertIn("pitch_smoothing", PERFORMANCE_DEFAULT_KEYS)

    def test_applies_live(self):
        from engine.core import RealtimeEngine

        engine = RealtimeEngine.__new__(RealtimeEngine)
        engine.settings = EngineSettings(pitch_smoothing=False)
        engine.rvc = types.SimpleNamespace(pitch_smoothing=True, change_key=lambda key: None)
        engine.running = False
        engine.apply_hot_settings({"pitch_smoothing": False})
        self.assertFalse(engine.rvc.pitch_smoothing)


if __name__ == "__main__":
    unittest.main()
