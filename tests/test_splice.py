import math
import unittest

import torch

from tools.splice import CONFIDENCE_THRESHOLD, WsolaSplicer

RATE = 48000
BLOCK = 12000  # 0.25 s
PERIOD = 192  # 250 Hz at 48 kHz


def sine(start, length, period=PERIOD, phase=0.0):
    t = torch.arange(start, start + length, dtype=torch.float64)
    return torch.sin(2 * math.pi * t / period + phase).float()


class WsolaSplicerTest(unittest.TestCase):
    def setUp(self):
        self.splicer = WsolaSplicer(RATE, BLOCK, torch.device("cpu"))
        self.length = self.splicer.input_length

    def test_geometry_is_40ms_fade_and_10ms_search(self):
        self.assertEqual(self.splicer.fade, 1920)
        self.assertEqual(self.splicer.search, 480)
        self.assertEqual(self.length, BLOCK + 1920 + 480)

    def test_output_is_one_block_and_template_carries_over(self):
        out = self.splicer.splice(sine(0, self.length))
        self.assertEqual(out.shape[0], BLOCK)
        self.assertGreater(float(self.splicer.template.abs().max()), 0.5)

    def test_random_phase_chunks_splice_without_beating(self):
        """Chunks with arbitrary phase errors must still join at full amplitude.

        Plain overlap-add of two out-of-phase sines dips towards silence in
        the crossfade (to about 0.18 here); an aligned splice keeps it.
        """
        generator = torch.Generator().manual_seed(1)
        outputs, confidences = [], []
        for n in range(12):
            phase = float(torch.rand(1, generator=generator)) * 2 * math.pi
            chunk = sine(n * BLOCK, self.length, phase=phase)
            outputs.append(self.splicer.splice(chunk).clone())
            if n:
                confidences.append(self.splicer.last_confidence)
        signal = torch.cat(outputs)
        window = PERIOD * 2
        worst = min(
            float(signal[start : start + window].pow(2).mean().sqrt()) * math.sqrt(2)
            for start in range(BLOCK, signal.shape[0] - window, PERIOD)
        )
        self.assertGreater(worst, 0.97)
        self.assertGreater(min(confidences), 0.9)

    def test_first_seam_after_silence_starts_at_the_window_start(self):
        # Splicing early keeps the continuation away from the chunk's causal
        # edge, where edge artifacts live.
        self.splicer.splice(torch.zeros(self.length))
        self.assertEqual(self.splicer.last_offset, 0)

    def test_unmatched_audio_keeps_the_previous_offset(self):
        self.splicer.template[:] = sine(0, self.splicer.fade)
        self.splicer.last_offset = 137
        generator = torch.Generator().manual_seed(2)
        for _ in range(3):
            # Freshly synthesised noise has no true match with the template.
            self.splicer.splice(torch.randn(self.length, generator=generator) * 0.1)
            self.assertEqual(self.splicer.last_offset, 137)
            self.assertLess(self.splicer.last_confidence, CONFIDENCE_THRESHOLD)

    def test_prefers_the_peak_nearest_the_previous_seam(self):
        # The template's phase recurs every period: at offsets 380 and 188.
        # The previous seam was at 200, so 188 (12 samples away) wins over 380.
        self.splicer.template[:] = sine(0, self.splicer.fade)
        self.splicer.last_offset = 200

        self.splicer.splice(sine(-380, self.length))

        self.assertEqual(self.splicer.last_offset, 188)
        self.assertGreater(self.splicer.last_confidence, 0.99)

    def test_a_clearly_better_match_still_wins_over_continuity(self):
        # A damped template matches only one place well; the far peak must
        # still beat a poor match next to the previous offset.
        envelope = torch.exp(-torch.arange(self.splicer.fade, dtype=torch.float32) / 300.0)
        self.splicer.template[:] = sine(0, self.splicer.fade) * envelope
        self.splicer.last_offset = 0
        chunk = torch.zeros(self.length)
        chunk[400 : 400 + self.splicer.fade] = self.splicer.template

        self.splicer.splice(chunk)

        self.assertEqual(self.splicer.last_offset, 400)

    def test_short_blocks_keep_an_unfaded_continuation(self):
        splicer = WsolaSplicer(RATE, 960, torch.device("cpu"))  # 20 ms < 40 ms fade
        chunk = sine(0, splicer.input_length)
        expected = chunk[960 : 960 + splicer.fade].clone()

        splicer.splice(chunk)

        self.assertTrue(torch.allclose(splicer.template, expected))

    def test_reset_clears_the_template(self):
        self.splicer.splice(sine(0, self.length))
        self.splicer.last_offset = 99
        self.splicer.reset()
        self.assertEqual(float(self.splicer.template.abs().max()), 0.0)
        self.assertEqual(self.splicer.last_offset, 0)


if __name__ == "__main__":
    unittest.main()
