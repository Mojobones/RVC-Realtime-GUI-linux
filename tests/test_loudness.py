import math
import unittest

import numpy as np

from tools import loudness
from tools.loudness import LoudnessMeter, integrated_loudness, judge

RATE = 48000


def sine(amplitude, seconds, rate=RATE, hz=997.0):
    t = np.arange(int(rate * seconds)) / rate
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def speech_like(seconds, rate=RATE, seed=0):
    """Noise shaped by a syllable-rate envelope: peaky like speech."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(rate * seconds)) / rate
    envelope = np.maximum(np.sin(2 * np.pi * 3.0 * t), 0.0) ** 2
    return (0.05 * envelope * rng.standard_normal(t.shape[0])).astype(np.float32)


class KWeightingTest(unittest.TestCase):
    def test_matches_the_coefficients_published_in_bs1770(self):
        (shelf_b, shelf_a), (high_b, high_a) = loudness.k_weighting(48000)
        np.testing.assert_allclose(
            shelf_b, [1.53512485958697, -2.69169618940638, 1.19839281085285], atol=1e-12
        )
        np.testing.assert_allclose(shelf_a, [1.0, -1.69065929318241, 0.73248077421585], atol=1e-12)
        np.testing.assert_allclose(high_b, [1.0, -2.0, 1.0])
        np.testing.assert_allclose(high_a, [1.0, -1.99004745483398, 0.99007225036621], atol=1e-12)

    def test_reference_sine_reads_minus_3_01_lufs_at_common_rates(self):
        for rate in (48000, 44100):
            with self.subTest(rate=rate):
                self.assertAlmostEqual(integrated_loudness(sine(1.0, 10, rate), rate), -3.01, delta=0.02)
        self.assertAlmostEqual(integrated_loudness(sine(0.1, 10), RATE), -23.01, delta=0.02)


class GatingTest(unittest.TestCase):
    def test_pauses_do_not_lower_the_reading(self):
        tone = sine(0.1, 2)
        with_pauses = np.concatenate([tone, np.zeros(RATE * 3, np.float32)] * 4)
        # Averaging the pauses in would read 4 LU low (2 s of tone per 5 s);
        # gating leaves only the partly filled blocks at the pause edges.
        self.assertAlmostEqual(
            integrated_loudness(with_pauses, RATE), integrated_loudness(np.tile(tone, 4), RATE), delta=0.8
        )

    def test_silence_has_no_reading(self):
        self.assertIsNone(integrated_loudness(np.zeros(RATE * 2, np.float32), RATE))

    def test_block_size_does_not_change_the_result(self):
        signal = speech_like(8)
        whole = LoudnessMeter(RATE, window_seconds=math.inf)
        whole.process(signal)
        streamed = LoudnessMeter(RATE, window_seconds=math.inf)
        for start in range(0, signal.shape[0], 1234):
            streamed.process(signal[start : start + 1234])
        np.testing.assert_allclose(streamed.block_loudness(), whole.block_loudness(), atol=1e-9)


class SummaryTest(unittest.TestCase):
    def meter(self, seconds=10):
        meter = LoudnessMeter(RATE)
        meter.process(speech_like(seconds))
        return meter

    def test_gain_shifts_the_reading_without_measuring_again(self):
        meter = self.meter()
        quiet, louder = meter.summary(0.0), meter.summary(12.0)
        self.assertAlmostEqual(louder["lufs"] - quiet["lufs"], 12.0, delta=0.11)

    def test_waits_for_enough_speech(self):
        meter = self.meter(seconds=2)
        reading = meter.summary(0.0)
        self.assertIsNotNone(reading["lufs"])
        self.assertIsNone(reading["verdict"])
        self.assertEqual(LoudnessMeter(RATE).summary(0.0)["lufs"], None)

    def test_one_suggestion_reaches_good_from_any_gain(self):
        meter = self.meter()
        for target in loudness.TARGETS_LUFS.values():
            for gain in (-10.0, 0.0, 10.0, 20.0, 30.0, 40.0):
                with self.subTest(target=target, gain=gain):
                    reading = meter.summary(gain, target)
                    after = meter.summary(gain + reading["adjust_db"], target)
                    self.assertEqual(after["verdict"], "good")
                    self.assertAlmostEqual(after["lufs"], target, delta=1.0)
                    self.assertLessEqual(after["over_percent"], loudness.MAX_OVER_PERCENT)

    def test_over_full_scale_share_follows_the_gain(self):
        meter = self.meter()
        self.assertEqual(meter.summary(-20.0)["over_percent"], 0.0)
        self.assertGreater(meter.summary(40.0)["over_percent"], loudness.MAX_OVER_PERCENT)

    def test_window_keeps_only_recent_speech(self):
        meter = LoudnessMeter(RATE, window_seconds=5)
        meter.process(speech_like(10) * np.float32(0.01))
        meter.process(speech_like(10, seed=1))
        expected = LoudnessMeter(RATE, window_seconds=math.inf)
        expected.process(speech_like(10, seed=1)[-RATE * 5 :])
        self.assertAlmostEqual(meter.summary(0.0)["lufs"], expected.summary(0.0)["lufs"], delta=1.0)

    def test_reset_forgets_everything(self):
        meter = self.meter()
        meter.reset()
        self.assertIsNone(meter.summary(0.0)["lufs"])


class JudgeTest(unittest.TestCase):
    def test_verdicts_around_the_target(self):
        for target in loudness.TARGETS_LUFS.values():
            with self.subTest(target=target):
                self.assertEqual(judge(target, 0.0, math.inf, target), ("good", 0.0))
                self.assertEqual(judge(target - 7.3, 0.0, math.inf, target), ("quiet", 7.0))
                self.assertEqual(judge(target + 6.2, 0.0, math.inf, target), ("loud", -6.5))

    def test_recording_15_is_good_for_streaming_but_loud_for_voice_chat(self):
        # The user's recording measured -16.8 LUFS and sounded too loud.
        self.assertEqual(judge(-16.8, 0.0, 0.5, -16.0)[0], "good")
        self.assertEqual(judge(-16.8, 0.0, 0.5, -21.0), ("loud", -4.5))

    def test_a_raise_never_exceeds_the_clipping_headroom(self):
        self.assertEqual(judge(-29.0, 0.0, 3.2, -21.0), ("quiet", 3.0))

    def test_squashed_peaks_are_too_loud_even_at_the_target(self):
        verdict, change = judge(-16.0, 3.0, -2.2, -16.0)
        self.assertEqual(verdict, "loud")
        self.assertEqual(change, -2.5)


class EngineLoudnessTest(unittest.TestCase):
    def engine(self):
        from tests.test_engine_pipeline import passthrough_engine

        engine = passthrough_engine(block_time=0.1)
        engine.file_audio_source = None
        engine.settings.input_source = "microphone"
        engine.settings.output_gain_db = 6.0
        return engine

    def test_output_is_measured_before_the_gain(self):
        engine = self.engine()
        signal = speech_like(8)
        block = engine.block_frame
        for start in range(0, signal.shape[0] - block + 1, block):
            engine.audio_callback(signal[start : start + block, None].copy(), block, None, None)
        reading = engine.output_loudness()
        before_gain = engine.loudness.summary(0.0)["lufs"]
        self.assertAlmostEqual(reading["lufs"] - before_gain, 6.0, delta=0.11)
        engine.settings.input_source = "file"
        engine.settings.file_input_volume = 0.5
        self.assertAlmostEqual(
            engine.output_loudness()["lufs"] - before_gain, 6.0 - 6.02, delta=0.11
        )

    def test_the_target_setting_picks_the_target(self):
        engine = self.engine()
        engine.loudness.process(speech_like(8))
        self.assertEqual(engine.settings.loudness_target, "voice_chat")
        self.assertEqual(engine.output_loudness()["target_lufs"], -21.0)
        engine.settings.loudness_target = "streaming"
        streaming = engine.output_loudness()
        self.assertEqual(streaming["target_lufs"], -16.0)

    def test_changing_the_input_gain_starts_over(self):
        engine = self.engine()
        engine.rvc.change_key = lambda key: None
        engine.loudness.process(speech_like(6))
        engine.apply_hot_settings({"input_gain_db": 3.0})
        self.assertIsNone(engine.loudness.summary(0.0)["lufs"])


if __name__ == "__main__":
    unittest.main()
