import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from engine.core import RealtimeEngine
from engine.settings import EngineSettings, coerce_setting, load_model_settings, save_model_settings
from infer.rtrvc import DEFAULT_PROTECT, protect_consonants


class ProtectConsonantsTest(unittest.TestCase):
    def setUp(self):
        self.blended = torch.full((1, 6, 4), 10.0)
        self.original = torch.zeros((1, 6, 4))
        # Frames 2-3 are unvoiced (a consonant).
        self.pitchf = torch.tensor([[200.0, 210.0, 0.0, 0.0, 190.0, 180.0]])

    def test_unvoiced_frames_keep_most_of_the_original_features(self):
        out = protect_consonants(self.blended, self.original, self.pitchf, 0.33)

        torch.testing.assert_close(out[0, [0, 1, 4, 5]], self.blended[0, [0, 1, 4, 5]])
        torch.testing.assert_close(out[0, 2:4], torch.full((2, 4), 3.3))

    def test_zero_protect_restores_the_original_on_consonants(self):
        out = protect_consonants(self.blended, self.original, self.pitchf, 0.0)
        torch.testing.assert_close(out[0, 2:4], self.original[0, 2:4])

    def test_works_in_half_precision(self):
        out = protect_consonants(self.blended.half(), self.original.half(), self.pitchf, 0.33)
        self.assertEqual(out.dtype, torch.float16)


class ProtectSettingTest(unittest.TestCase):
    def test_default_and_range(self):
        self.assertEqual(EngineSettings().protect, DEFAULT_PROTECT)
        self.assertEqual(coerce_setting("protect", 0.9), 0.5)
        self.assertEqual(coerce_setting("protect", -1), 0.0)

    def test_saved_with_the_model_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            save_model_settings(directory, EngineSettings(protect=0.2))
            with open(os.path.join(directory, "realtime_settings.json"), encoding="utf-8") as saved:
                self.assertEqual(json.load(saved)["protect"], 0.2)
            self.assertEqual(load_model_settings(directory)["protect"], 0.2)

    def test_changes_apply_live(self):
        engine = RealtimeEngine.__new__(RealtimeEngine)
        engine.settings = EngineSettings(protect=0.1)
        engine.running = False
        received = []
        engine.rvc = SimpleNamespace(change_protect=received.append)

        engine.apply_hot_settings({"protect": 0.1})

        self.assertEqual(received, [0.1])


if __name__ == "__main__":
    unittest.main()
