import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from infer import rtrvc
from infer.rtrvc import fill_short_gaps

V = 200.0  # a voiced frame


def track(*runs):
    """Build an f0 track from (value, frames) runs."""
    return np.concatenate([np.full(frames, value) for value, frames in runs])


class FillShortGapsTest(unittest.TestCase):
    def test_short_internal_gap_is_bridged_on_a_log_scale(self):
        f0 = track((100.0, 3), (0.0, 3), (400.0, 3))

        filled = fill_short_gaps(f0, max_gap=3)

        np.testing.assert_allclose(filled[3:6], [100 * 4 ** 0.25, 200.0, 100 * 4 ** 0.75])
        self.assertTrue(np.all(filled > 0))

    def test_long_gap_stays_unvoiced(self):
        f0 = track((V, 5), (0.0, 4), (V, 5))
        np.testing.assert_array_equal(fill_short_gaps(f0, max_gap=3)[5:9], 0.0)

    def test_gaps_touching_the_window_edges_stay_unvoiced(self):
        f0 = track((0.0, 2), (V, 5), (0.0, 1))
        filled = fill_short_gaps(f0, max_gap=3)
        np.testing.assert_array_equal(filled[:2], 0.0)
        self.assertEqual(filled[-1], 0.0)

    def test_all_unvoiced_and_all_voiced_are_unchanged(self):
        np.testing.assert_array_equal(fill_short_gaps(np.zeros(8)), 0.0)
        voiced = np.linspace(100, 300, 8)
        np.testing.assert_allclose(fill_short_gaps(voiced), voiced)

    def test_input_is_not_modified(self):
        f0 = track((V, 3), (0.0, 2), (V, 3))
        before = f0.copy()
        fill_short_gaps(f0)
        np.testing.assert_array_equal(f0, before)

    def test_full_interpolation_switch_restores_the_old_behaviour(self):
        f0 = track((0.0, 2), (V, 3), (0.0, 6), (V, 3))
        with mock.patch.object(rtrvc, "FULL_F0_INTERPOLATION", True):
            filled = fill_short_gaps(f0)
        self.assertTrue(np.all(filled > 0))


class DetectorsUseGapFillingTest(unittest.TestCase):
    """Every detector must keep long unvoiced runs (consonants, breaths) at 0."""

    def detector_output(self):
        # One 2-frame dropout (bridged) and one 12-frame consonant (kept).
        return track((V, 10), (0.0, 2), (V, 10), (0.0, 12), (V, 10)).astype(np.float32)

    def rvc(self):
        engine = rtrvc.RVC.__new__(rtrvc.RVC)
        engine.device = torch.device("cpu")
        engine.f0_mel_min = 1127 * np.log(1 + 50 / 700)
        engine.f0_mel_max = 1127 * np.log(1 + 1100 / 700)
        engine.is_half = False
        return engine

    def check(self, pitchf):
        pitchf = pitchf.cpu().numpy()
        self.assertTrue(np.all(pitchf[10:12] > 0), "short dropout should be bridged")
        np.testing.assert_array_equal(pitchf[22:34], 0.0)

    def test_rmvpe(self):
        engine = self.rvc()
        engine.model_rmvpe = SimpleNamespace(infer_from_audio=lambda x, thred: self.detector_output())
        _, pitchf = engine.get_f0_rmvpe(np.zeros(16000, dtype=np.float32), 0)
        self.check(pitchf)

    def test_fcpe(self):
        engine = self.rvc()
        output = torch.from_numpy(self.detector_output())[None, :, None]
        engine.model_fcpe = SimpleNamespace(infer=lambda *args, **kwargs: output)
        _, pitchf = engine.get_f0_fcpe(torch.zeros(16000), 0)
        self.check(pitchf)

    def test_pm(self):
        engine = self.rvc()
        detector = self.detector_output()
        pitch = SimpleNamespace(t1=1.5 / 65, selected_array={"frequency": detector})
        sound = SimpleNamespace(to_pitch_ac=lambda **kwargs: pitch)
        with mock.patch.object(rtrvc.parselmouth, "Sound", return_value=sound):
            _, pitchf = engine.get_f0(torch.zeros(detector.shape[0] * 160 - 160), 0, "pm")
        self.check(pitchf)


if __name__ == "__main__":
    unittest.main()
