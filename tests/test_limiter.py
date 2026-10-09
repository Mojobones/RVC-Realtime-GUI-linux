import unittest

import numpy as np
import torch

from tools.limiter import KNEE, soft_clip


class SoftClipTest(unittest.TestCase):
    def test_below_the_knee_is_untouched(self):
        audio = np.linspace(-KNEE, KNEE, 1001).astype(np.float32)
        np.testing.assert_array_equal(soft_clip(audio), audio)

    def test_never_exceeds_full_scale_and_keeps_sign(self):
        audio = np.linspace(-20, 20, 4001).astype(np.float32)
        clipped = soft_clip(audio)
        self.assertLessEqual(float(np.abs(clipped).max()), 1.0)
        np.testing.assert_array_equal(np.sign(clipped), np.sign(audio))

    def test_smooth_and_monotonic_through_the_knee(self):
        audio = np.linspace(0, 3, 30001)
        clipped = soft_clip(audio)
        slope = np.diff(clipped) / np.diff(audio)
        self.assertTrue(np.all(slope > 0))
        # No corner: the slope changes gradually where the curve starts to bend.
        self.assertLess(float(np.abs(np.diff(slope)).max()), 1e-3)

    def test_rounds_peaks_less_harshly_than_a_hard_clip(self):
        t = np.arange(4800) / 48000
        loud = (1.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

        def distortion(signal):
            spectrum = np.abs(np.fft.rfft(signal))
            bins = np.fft.rfftfreq(signal.shape[0], 1 / 48000)
            fundamental = spectrum[np.argmin(np.abs(bins - 220))]
            upper = spectrum[bins > 2000]  # energy far above the fundamental
            return float(np.sqrt(np.sum(upper**2)) / fundamental)

        self.assertLess(distortion(soft_clip(loud)), distortion(np.clip(loud, -1, 1)))

    def test_torch_and_numpy_agree(self):
        audio = np.linspace(-3, 3, 601).astype(np.float32)
        np.testing.assert_allclose(
            soft_clip(torch.from_numpy(audio)).numpy(), soft_clip(audio), atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
