"""Context hold: freeze the model's past context while the input is silent.

An identity "model" (its output is its own input window) makes the expected
output exact: the input delayed by the 50 ms fade plus search span.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from engine.core import RealtimeEngine
from engine.settings import EngineSettings
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
        self.assertEqual(len(engine.infer_calls), speech_blocks)
        # Silence after the 40 ms fade-out of the last word.
        np.testing.assert_array_equal(output[int(1.04 * RATE) : resume], 0.0)

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
