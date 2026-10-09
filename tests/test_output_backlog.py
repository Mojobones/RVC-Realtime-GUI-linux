"""A standing output backlog drains by dropping blocks of silence only.

If inference falls behind for a moment, the audio that piled up stays
queued for the rest of the session (input and output run at the same rate),
adding up to several chunks of delay.  Dropping silent blocks while too much
is queued shortens a pause instead, which is inaudible.
"""

import queue
import unittest

import numpy as np

from engine import core
from tools.audio_fifo import AudioFrameFifo

RATE = 48000


def engine_with_fifo(block_time=0.14):
    from tests.test_engine_pipeline import passthrough_engine

    engine = passthrough_engine(block_time=block_time)
    engine.output_queue = AudioFrameFifo(1, max_frames=engine.block_frame * 4)
    engine.native_input_fifo = None
    engine.file_audio_source = None
    engine.settings.input_source = "microphone"
    engine.latest_infer_time = 25
    engine.running = True
    return engine


def drain(fifo, frames):
    fifo.read(frames)


class BacklogTrimTest(unittest.TestCase):
    def setUp(self):
        self.engine = engine_with_fifo()
        self.block = self.engine.block_frame
        self.silence = np.zeros(self.block, dtype=np.float32)
        self.speech = (0.1 * np.sin(np.arange(self.block) / 5)).astype(np.float32)

    def limit(self):
        return self.block + int(core.BACKLOG_MARGIN_SECONDS * RATE)

    def test_silence_drains_a_backlog(self):
        fifo = self.engine.output_queue
        fifo.write(np.zeros((self.block * 4, 1), np.float32))  # 560 ms stuck
        for _ in range(6):  # real time: one block in, one block out
            self.engine.queue_output(fifo, self.silence)
            drain(fifo, self.block)
        self.assertLess(fifo.available_frames, self.limit())
        self.assertGreater(self.engine.trimmed_frames, 0)

    def test_speech_is_never_dropped(self):
        fifo = self.engine.output_queue
        fifo.write(np.zeros((self.block * 3, 1), np.float32))
        before = fifo.available_frames
        self.engine.queue_output(fifo, self.speech)
        self.assertEqual(fifo.available_frames, before + self.block)
        self.assertEqual(self.engine.trimmed_frames, 0)

    def test_a_normal_queue_keeps_its_silence(self):
        fifo = self.engine.output_queue
        fifo.write(np.zeros((self.block // 2, 1), np.float32))  # usual leftover
        self.engine.queue_output(fifo, self.silence)
        self.assertEqual(fifo.available_frames, self.block // 2 + self.block)

    def test_margin_grows_with_slow_inference(self):
        fifo = self.engine.output_queue
        self.engine.latest_infer_time = 80  # 2 x 80 ms > 60 ms
        fifo.write(np.zeros((self.block + int(0.1 * RATE), 1), np.float32))
        self.engine.queue_output(fifo, self.silence)
        self.assertEqual(self.engine.trimmed_frames, 0)

    def test_file_playback_is_left_alone(self):
        fifo = self.engine.output_queue
        self.engine.settings.input_source = "file"
        fifo.write(np.zeros((self.block * 3, 1), np.float32))
        self.engine.queue_output(fifo, self.silence)
        self.assertEqual(self.engine.trimmed_frames, 0)

    def test_alsa_block_queue_is_trimmed_too(self):
        blocks = queue.Queue(maxsize=3)
        for _ in range(3):
            blocks.put(self.silence)
        self.engine.queue_output(blocks, self.silence)
        self.assertEqual(blocks.qsize(), 3)
        self.assertEqual(self.engine.trimmed_frames, self.block)

    def test_queued_ms_counts_input_and_output(self):
        self.engine.native_input_fifo = AudioFrameFifo(1)
        self.engine.native_input_fifo.write(np.zeros((4800, 1), np.float32))
        self.engine.output_queue.write(np.zeros((9600, 1), np.float32))
        self.assertEqual(self.engine.queued_ms(), 300)
        self.engine.running = False
        self.assertIsNone(self.engine.queued_ms())



class UnderrunCounterTest(unittest.TestCase):
    """The output callback counts the times it found too little audio queued."""

    def setUp(self):
        self.engine = engine_with_fifo()
        self.engine.output_flowing = False
        self.engine.output_underruns = 0
        self.engine.output_selectors = []
        self.period = 1024

    def callback(self):
        out = np.zeros((self.period, 2), np.float32)
        self.engine.output_audio_callback(out, self.period, None, None)

    def test_waiting_for_the_first_chunk_is_not_an_underrun(self):
        for _ in range(5):
            self.callback()
        self.assertEqual(self.engine.output_underruns, 0)

    def test_running_dry_after_audio_started_counts(self):
        self.engine.output_queue.write(np.full((1500, 1), 0.1, np.float32))
        self.callback()  # full period
        self.callback()  # 476 left: short
        self.callback()  # empty
        self.assertEqual(self.engine.output_underruns, 2)

    def test_not_counted_once_stopped(self):
        self.engine.output_queue.write(np.full((1024, 1), 0.1, np.float32))
        self.callback()
        self.engine.running = False
        self.callback()
        self.assertEqual(self.engine.output_underruns, 0)


if __name__ == "__main__":
    unittest.main()
