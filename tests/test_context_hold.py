"""Context hold: freeze the model's past context while the input is silent.

An identity "model" (its output is its own input window) makes the expected
output exact: the input delayed by the 50 ms fade plus search span.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from engine.core import RealtimeEngine
from engine.settings import EngineSettings, SettingError, coerce_setting
from tools import rnnoise
from tools.wav_recorder import WavRecorder

RATE = 48000
BLOCK_TIME = 0.1
LAG = int(0.050 * RATE)


def engine_with_identity_model(hold_context):
    engine = RealtimeEngine.__new__(RealtimeEngine)
    engine.settings = EngineSettings(
        block_time=BLOCK_TIME, extra_time=0.5, hold_context=hold_context, rms_mix_rate=1.0
    )
    engine.config = SimpleNamespace(device=torch.device("cpu"))
    calls = []

    def infer(input_wav_res, block_frame_16k, skip_head, return_length, f0method):
        calls.append(1)
        return engine.input_wav[engine.extra_frame :].clone()

    engine.rvc = SimpleNamespace(
        tgt_sr=RATE,
        cache_pitch=torch.zeros(1024, dtype=torch.long),
        cache_pitchf=torch.zeros(1024),
        infer=infer,
    )
    engine.infer_calls = calls
    engine.function = "vc"
    engine.running = False
    engine.recorder = WavRecorder()
    engine.chunk_tap = None
    engine.monitor_device_index = None
    engine.output_queue = None
    engine.monitor_queue = None
    engine.last_input_meter_update = 0.0
    engine.last_output_meter_update = 0.0
    engine.emit = lambda event, data: None
    engine.prepare_inference(RATE)
    return engine


def speech_pause_speech():
    """1 s of 'speech', a 3 s pause (longer than the 0.5 s context), 1 s more."""
    rng = np.random.default_rng(4)
    signal = np.zeros(5 * RATE, dtype=np.float32)
    signal[: RATE] = rng.standard_normal(RATE) * 0.1
    signal[4 * RATE :] = rng.standard_normal(RATE) * 0.1
    return signal


def run(engine, signal, snapshot_at=None):
    block = engine.block_frame
    outputs, snapshots = [], {}
    for start in range(0, signal.shape[0], block):
        if snapshot_at and start in snapshot_at:
            snapshots[start] = engine.input_wav.clone()
        chunk = signal[start : start + block, None].copy()
        outputs.append(engine.audio_callback(chunk, block, None, None).copy())
    return np.concatenate(outputs), snapshots


class ContextHoldTest(unittest.TestCase):
    def test_pause_freezes_context_and_skips_inference(self):
        engine = engine_with_identity_model(hold_context=True)
        signal = speech_pause_speech()
        pause_start, resume = int(1.2 * RATE), 4 * RATE

        output, snapshots = run(engine, signal, snapshot_at={pause_start, resume})

        # The context is untouched for the whole pause.
        self.assertTrue(torch.equal(snapshots[pause_start], snapshots[resume]))
        speech_blocks = 2 * RATE // engine.block_frame
        # Plus the first silent block, which is converted normally (hangover).
        self.assertEqual(len(engine.infer_calls), speech_blocks + 1)
        np.testing.assert_array_equal(output[RATE + LAG : resume], 0.0)

    def test_word_ending_is_not_faded_out_by_the_hold(self):
        # Output lags input by 50 ms, so the first silent input block's output
        # still holds the end of the word; it must come out intact.
        engine = engine_with_identity_model(hold_context=True)
        signal = speech_pause_speech()

        output, _ = run(engine, signal)

        np.testing.assert_allclose(
            output[RATE : RATE + LAG], signal[RATE - LAG : RATE], atol=1e-5
        )

    def test_a_single_silent_block_between_words_is_not_held(self):
        engine = engine_with_identity_model(hold_context=True)
        rng = np.random.default_rng(9)
        signal = (rng.standard_normal(2 * RATE) * 0.1).astype(np.float32)
        gap = slice(RATE, RATE + engine.block_frame)  # exactly one silent block
        signal[gap] = 0.0

        output, _ = run(engine, signal)

        self.assertEqual(len(engine.infer_calls), signal.shape[0] // engine.block_frame)
        start = RATE // 2
        np.testing.assert_allclose(
            output[start:], signal[start - LAG : output.shape[0] - LAG], atol=1e-5
        )

    def test_resume_does_not_replay_the_previous_word(self):
        engine = engine_with_identity_model(hold_context=True)
        signal = speech_pause_speech()
        resume = 4 * RATE

        output, _ = run(engine, signal)

        # Before the new speech reaches the output (50 ms after it starts),
        # nothing from the frozen context may be heard.
        np.testing.assert_array_equal(output[resume : resume + LAG], 0.0)
        # After the 5 ms fade-in, the output is the input delayed as usual.
        settled = resume + LAG + int(0.005 * RATE)
        np.testing.assert_allclose(
            output[settled:], signal[settled - LAG : output.shape[0] - LAG], atol=1e-5
        )

    def test_without_hold_the_pause_overwrites_the_context(self):
        engine = engine_with_identity_model(hold_context=False)
        signal = speech_pause_speech()

        _, snapshots = run(engine, signal, snapshot_at={4 * RATE})

        self.assertEqual(len(engine.infer_calls), signal.shape[0] // engine.block_frame)
        self.assertEqual(float(snapshots[4 * RATE].abs().max()), 0.0)

    def test_passthrough_is_not_held(self):
        engine = engine_with_identity_model(hold_context=True)
        engine.function = "im"
        signal = speech_pause_speech()

        _, snapshots = run(engine, signal, snapshot_at={4 * RATE})

        self.assertFalse(engine.context_held)
        self.assertEqual(float(snapshots[4 * RATE].abs().max()), 0.0)

    def test_noise_gate_above_the_default_raises_the_hold_level(self):
        engine = engine_with_identity_model(hold_context=True)
        quiet = np.full(engine.block_frame, 0.005, dtype=np.float32)  # about -46 dBFS
        self.assertFalse(engine.block_is_silent(quiet))
        engine.settings.noise_gate_db = -40.0
        self.assertTrue(engine.block_is_silent(quiet))


if __name__ == "__main__":
    unittest.main()


def fan_noise(seconds, level_db, rng):
    """Steady low-passed noise, like a fan or air conditioner."""
    white = rng.standard_normal(int(seconds * RATE))
    noise = np.zeros_like(white)
    alpha = np.exp(-2 * np.pi * 300 / RATE)
    for i in range(1, white.shape[0]):
        noise[i] = alpha * noise[i - 1] + (1 - alpha) * white[i]
    return (noise / np.std(noise) * 10 ** (level_db / 20)).astype(np.float32)


def voice_like(seconds, f0=140.0):
    t = np.arange(int(seconds * RATE)) / RATE
    tone = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 20))
    return (0.1 * tone * (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t) ** 2)).astype(np.float32)


class HoldDetectorTest(unittest.TestCase):
    def noisy_pause(self):
        rng = np.random.default_rng(6)
        room = fan_noise(6.0, -35.0, rng)  # far above the -50 dB loudness threshold
        room[: RATE] += voice_like(1.0)
        room[5 * RATE :] += voice_like(1.0)
        return room

    def held_blocks_during_pause(self, detector):
        engine = engine_with_identity_model(hold_context=True)
        engine.settings.hold_detector = detector
        signal = self.noisy_pause()
        block = engine.block_frame
        held = 0
        for start in range(0, signal.shape[0], block):
            engine.audio_callback(signal[start : start + block, None].copy(), block, None, None)
            if 1.5 * RATE <= start < 4.5 * RATE:
                held += engine.context_held
        return held, int(3.0 * RATE / block)

    @unittest.skipUnless(rnnoise.available(), "librnnoise not installed")
    def test_loudness_never_holds_in_a_noisy_room(self):
        held, total = self.held_blocks_during_pause("level")
        self.assertEqual(held, 0)

    @unittest.skipUnless(rnnoise.available(), "librnnoise not installed")
    def test_voice_detection_holds_through_room_noise(self):
        held, total = self.held_blocks_during_pause("voice")
        self.assertGreaterEqual(held, 0.8 * total)

    @unittest.skipUnless(rnnoise.available(), "librnnoise not installed")
    def test_voice_detection_does_not_hold_speech(self):
        engine = engine_with_identity_model(hold_context=True)
        engine.settings.hold_detector = "voice"
        speech = voice_like(2.0)
        block = engine.block_frame
        held = []
        for start in range(0, speech.shape[0], block):
            engine.audio_callback(speech[start : start + block, None].copy(), block, None, None)
            held.append(engine.context_held)
        self.assertFalse(any(held[5:]))  # after RNNoise settles

    def test_voice_detection_falls_back_to_loudness_without_rnnoise(self):
        with mock.patch.object(rnnoise, "available", return_value=False):
            engine = engine_with_identity_model(hold_context=True)
        engine.settings.hold_detector = "voice"
        self.assertIsNone(engine.vad)
        quiet = np.zeros(engine.block_frame, dtype=np.float32)
        loud = np.full(engine.block_frame, 0.1, dtype=np.float32)
        self.assertTrue(engine.block_is_silent(quiet))
        self.assertFalse(engine.block_is_silent(loud))

    def test_rejects_unknown_detectors(self):
        with self.assertRaises(SettingError):
            coerce_setting("hold_detector", "magic")
        self.assertEqual(coerce_setting("hold_detector", "voice"), "voice")


class LateDetectionTest(unittest.TestCase):
    """A detector that reports speech 20 ms late (like RNNoise) must not clip words."""

    def late_detector_run(self, pre_roll=True):
        engine = engine_with_identity_model(hold_context=True)
        on_time = engine.block_is_quiet
        late = int(0.020 * RATE)
        # Judge each block without its last 20 ms, as a lagging detector would.
        engine.block_is_silent = lambda mono: on_time(mono[:-late])
        if not pre_roll:
            original = engine.hold_silence

            def forget_held_block(start_time):
                result = original(start_time)
                engine.held_block = None
                return result

            engine.hold_silence = forget_held_block
        rng = np.random.default_rng(8)
        signal = np.zeros(5 * RATE, dtype=np.float32)
        signal[:RATE] = rng.standard_normal(RATE) * 0.1
        onset = 4 * RATE - int(0.015 * RATE)  # 15 ms before a block boundary
        signal[onset:] = rng.standard_normal(signal.shape[0] - onset) * 0.1
        output, _ = run(engine, signal)
        return output, signal, onset

    def test_word_start_is_kept_with_pre_roll(self):
        output, signal, onset = self.late_detector_run()
        attack = slice(onset + LAG, onset + LAG + int(0.030 * RATE))
        np.testing.assert_allclose(output[attack], signal[onset : onset + int(0.030 * RATE)], atol=1e-5)

    def test_without_pre_roll_the_word_start_is_lost(self):
        output, signal, onset = self.late_detector_run(pre_roll=False)
        attack = output[onset + LAG : onset + LAG + int(0.010 * RATE)]
        self.assertLess(float(np.abs(attack).max()), 1e-6)
