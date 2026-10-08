import io
import queue
import unittest

from engine.server import LogWriter
from engine.settings import EngineSettings, db_to_linear
from tools.audio_fifo import enqueue_latest


class GainConversionTest(unittest.TestCase):
    def test_unity_gain(self):
        self.assertEqual(db_to_linear(0), 1.0)

    def test_positive_gain(self):
        self.assertAlmostEqual(db_to_linear(6), 1.9952623149688795)

    def test_negative_gain(self):
        self.assertAlmostEqual(db_to_linear(-6), 0.5011872336272722)

    def test_settings_expose_linear_gains(self):
        settings = EngineSettings(output_gain_db=-6.0)
        self.assertAlmostEqual(settings.output_gain, 0.5011872336272722)


class MonitorQueueTest(unittest.TestCase):
    def test_full_queue_discards_oldest_block(self):
        blocks = queue.Queue(maxsize=2)
        enqueue_latest(blocks, "oldest")
        enqueue_latest(blocks, "middle")
        enqueue_latest(blocks, "latest")

        self.assertEqual(blocks.get_nowait(), "middle")
        self.assertEqual(blocks.get_nowait(), "latest")


class SettingsDefaultsTest(unittest.TestCase):
    def test_index_rate_default_matches_original_realtime_rvc(self):
        self.assertEqual(EngineSettings().index_rate, 0.0)

    def test_rms_mix_default_matches_original_realtime_rvc(self):
        self.assertEqual(EngineSettings().rms_mix_rate, 0.0)

    def test_performance_defaults_are_valid(self):
        settings = EngineSettings()
        self.assertGreater(settings.block_time, 0)
        self.assertGreater(settings.crossfade_time, 0)
        self.assertGreater(settings.extra_time, 0)


class LogWriterTest(unittest.TestCase):
    def test_writes_to_log_buffer_and_console_mirror(self):
        received = []
        server = type("Server", (), {"append_log": lambda self, text: received.append(text)})()
        mirror = io.StringIO()
        writer = LogWriter(server, mirror)

        writer.write("startup complete\n")
        writer.flush()

        self.assertEqual(received, ["startup complete\n"])
        self.assertEqual(mirror.getvalue(), "startup complete\n")


if __name__ == "__main__":
    unittest.main()
