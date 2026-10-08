import unittest
from unittest import mock

import numpy as np

from engine import core
from tools import rnnoise

RATE = rnnoise.SAMPLE_RATE


def voice_like(seconds, f0=140.0):
    """A harmonic signal with a speech-like spectral tilt and syllable envelope."""
    t = np.arange(int(seconds * RATE)) / RATE
    tone = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 20))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t) ** 2
    return (0.1 * tone * envelope).astype(np.float32)


def level_db(audio):
    return 10 * np.log10(np.mean(np.square(audio, dtype=np.float64)) + 1e-20)


@unittest.skipUnless(rnnoise.available(), "librnnoise not installed (sudo pacman -S rnnoise)")
class RNNoiseTest(unittest.TestCase):
    def test_rejects_partial_frames_and_keeps_length(self):
        denoiser = rnnoise.RNNoise()
        with self.assertRaises(ValueError):
            denoiser.process(np.zeros(100, dtype=np.float32))
        self.assertEqual(denoiser.process(np.zeros(4800, dtype=np.float32)).shape, (4800,))

    def test_output_delay_matches_the_engine_accounting(self):
        signal = voice_like(2.0)
        denoiser = rnnoise.RNNoise()
        output = np.concatenate(
            [denoiser.process(signal[i : i + 4800]) for i in range(0, signal.shape[0], 4800)]
        )
        lags = range(0, 2000)
        scores = [float(np.dot(signal[RATE // 2 : -2000], output[RATE // 2 + lag : -2000 + lag])) for lag in lags]
        # A periodic test tone cannot pin the peak to one sample; allow ±2.
        self.assertLessEqual(abs(int(np.argmax(scores)) - int(core.RNNOISE_DELAY_SECONDS * RATE)), 2)

    def test_suppresses_noise_and_keeps_voice(self):
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(RATE * 2) * 0.02).astype(np.float32)
        voice = voice_like(2.0)
        noise_out = rnnoise.RNNoise().process(noise)
        voice_out = rnnoise.RNNoise().process(voice)
        # Skip the first half second while the network settles.
        settle = RATE // 2
        self.assertLess(level_db(noise_out[settle:]) - level_db(noise[settle:]), -10.0)
        self.assertGreater(level_db(voice_out[settle:]) - level_db(voice[settle:]), -6.0)


class DenoiserSelectionTest(unittest.TestCase):
    def test_falls_back_to_spectral_gate_without_rnnoise(self):
        from tests.test_engine_pipeline import passthrough_engine

        with mock.patch.object(rnnoise, "available", return_value=False):
            engine = passthrough_engine(block_time=0.1)
        self.assertIsNone(engine.rnnoise)
        self.assertEqual(engine.input_denoise_delay(), core.FADE_SECONDS)

    @unittest.skipUnless(rnnoise.available(), "librnnoise not installed")
    def test_passthrough_with_rnnoise_delays_by_the_extra_20_ms(self):
        from tests.test_engine_pipeline import passthrough_engine

        engine = passthrough_engine(block_time=0.1)
        engine.settings.input_denoise = True
        self.assertIsNotNone(engine.rnnoise)
        signal = voice_like(3.0)
        block = engine.block_frame
        output = np.concatenate(
            [
                engine.audio_callback(signal[i : i + block, None].copy(), block, None, None).copy()
                for i in range(0, signal.shape[0] - block + 1, block)
            ]
        )
        lag = engine.splicer.fade + engine.splicer.search + int(core.RNNOISE_DELAY_SECONDS * RATE)
        start = RATE
        correlation = np.corrcoef(output[start:], signal[start - lag : output.shape[0] - lag])[0, 1]
        self.assertGreater(correlation, 0.95)


if __name__ == "__main__":
    unittest.main()
