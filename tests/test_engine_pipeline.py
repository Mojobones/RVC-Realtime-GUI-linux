"""Run the real engine buffers and WSOLA splice in passthrough mode.

Passthrough skips the model, so every chunk continues the previous one
exactly: WSOLA stays at offset 0 and the output must be the input delayed by
exactly the fade plus the search span, 50 ms.  This checks the buffer
geometry end to end.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from engine.core import RealtimeEngine
from engine.settings import EngineSettings
from tools.wav_recorder import WavRecorder

RATE = 48000


def passthrough_engine(block_time):
    engine = RealtimeEngine.__new__(RealtimeEngine)
    engine.settings = EngineSettings(block_time=block_time, extra_time=0.5)
    engine.config = SimpleNamespace(device=torch.device("cpu"))
    engine.rvc = SimpleNamespace(
        tgt_sr=RATE,
        cache_pitch=torch.zeros(1024, dtype=torch.long),
        cache_pitchf=torch.zeros(1024),
    )
    engine.function = "im"
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


class PassthroughPipelineTest(unittest.TestCase):
    def run_signal(self, engine, signal):
        outputs = []
        block = engine.block_frame
        for start in range(0, signal.shape[0] - block + 1, block):
            chunk = signal[start : start + block, None].copy()
            outputs.append(engine.audio_callback(chunk, block, None, None).copy())
        return np.concatenate(outputs)

    def test_output_is_input_delayed_by_50_ms(self):
        engine = passthrough_engine(block_time=0.1)
        rng = np.random.default_rng(3)
        signal = (rng.standard_normal(RATE * 2) * 0.1).astype(np.float32)

        output = self.run_signal(engine, signal)

        lag = engine.splicer.fade + engine.splicer.search
        self.assertEqual(lag, int(0.050 * RATE))
        # Skip the first blocks while the buffers fill.
        start = 3 * engine.block_frame
        np.testing.assert_allclose(
            output[start:], signal[start - lag : output.shape[0] - lag], atol=1e-5
        )
        self.assertEqual(engine.splicer.last_offset, 0)
        self.assertGreater(engine.splicer.last_confidence, 0.99)

    def test_blocks_shorter_than_the_fade_still_line_up(self):
        engine = passthrough_engine(block_time=0.02)
        t = np.arange(RATE, dtype=np.float32)
        signal = (0.3 * np.sin(2 * np.pi * 180.0 * t / RATE)).astype(np.float32)

        output = self.run_signal(engine, signal)

        lag = engine.splicer.fade + engine.splicer.search
        start = 10 * engine.block_frame
        np.testing.assert_allclose(
            output[start:], signal[start - lag : output.shape[0] - lag], atol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
