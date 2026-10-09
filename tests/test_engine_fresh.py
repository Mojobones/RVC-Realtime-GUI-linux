"""A freshly constructed engine must survive its periodic tick before any stream.

``tick`` runs about 20 times per second from the moment the engine starts,
long before the first Start, so everything it touches has to exist after
``__init__`` alone.  (An underrun counter set only per stream once broke
every tick until conversion started.)
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from engine import core
from engine.settings import EngineSettings


class FakeDevices:
    inputs = {}
    outputs = {}

    def __init__(self, sd):
        pass

    def refresh(self):
        pass

    def resolve(self, value, kind):
        return value

    def resolve_monitor(self, value):
        return value

    def to_dict(self):
        return {"inputs": [], "outputs": []}


class FreshEngineTest(unittest.TestCase):
    def test_tick_meters_and_state_work_before_any_stream(self):
        events = []
        with tempfile.TemporaryDirectory() as temp, \
             mock.patch.object(core, "Config", lambda: SimpleNamespace(device=torch.device("cpu"))), \
             mock.patch.object(core, "DeviceCatalog", FakeDevices), \
             mock.patch.object(core, "MODELS_ROOT", temp), \
             mock.patch.object(core, "VOICE_PITCH_PATH", str(Path(temp) / "voice_pitch.json")), \
             mock.patch.object(core, "load_settings", lambda: EngineSettings()):
            engine = core.RealtimeEngine(lambda event, data: events.append((event, data)))
            for _ in range(3):
                engine.tick()
            engine.meters()
            engine.state()
        self.assertFalse(engine.running)
        self.assertEqual(engine.output_underruns, 0)


if __name__ == "__main__":
    unittest.main()
