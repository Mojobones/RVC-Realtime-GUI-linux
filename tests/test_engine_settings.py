import json
import os
import tempfile
import unittest

from engine.settings import (
    EngineSettings,
    SettingError,
    coerce_setting,
    load_model_settings,
    load_settings,
    save_model_settings,
    save_settings,
)


class CoerceSettingTest(unittest.TestCase):
    def test_numbers_are_converted_and_clamped(self):
        self.assertEqual(coerce_setting("pitch", 12), 12.0)
        self.assertEqual(coerce_setting("block_time", 0.001), 0.02)
        self.assertEqual(coerce_setting("index_rate", 1.5), 1.0)
        self.assertEqual(coerce_setting("file_input_volume", -1), 0.0)

    def test_rejects_wrong_types_and_unknown_keys(self):
        for key, value in (
            ("pitch", "high"),
            ("pitch", True),
            ("pitch", float("nan")),
            ("input_denoise", 1),
            ("f0method", "crepe"),
            ("recording_mode", "surround"),
            ("nonsense", 1),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaises(SettingError):
                    coerce_setting(key, value)

    def test_empty_monitor_device_means_disabled(self):
        self.assertIsNone(coerce_setting("monitor_device", ""))
        self.assertIsNone(coerce_setting("monitor_device", None))
        self.assertEqual(coerce_setting("monitor_device", "[JACK] Out"), "[JACK] Out")


class PersistenceTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.directory.name, "engine.json")

    def tearDown(self):
        self.directory.cleanup()

    def test_global_settings_round_trip_without_model_settings(self):
        settings = EngineSettings(model_name="Voice", block_time=0.4, pitch=5.0)
        save_settings(settings, self.path)

        with open(self.path, encoding="utf-8") as saved_file:
            saved = json.load(saved_file)
        self.assertNotIn("pitch", saved)
        loaded = load_settings(self.path)
        self.assertEqual(loaded.model_name, "Voice")
        self.assertEqual(loaded.block_time, 0.4)
        self.assertEqual(loaded.pitch, 0.0)

    def test_invalid_saved_values_fall_back_to_defaults(self):
        with open(self.path, "w", encoding="utf-8") as config_file:
            json.dump({"block_time": "slow", "f0method": "crepe", "extra_time": 1.0}, config_file)

        loaded = load_settings(self.path)

        self.assertEqual(loaded.block_time, EngineSettings().block_time)
        self.assertEqual(loaded.f0method, "rmvpe")
        self.assertEqual(loaded.extra_time, 1.0)

    def test_missing_or_corrupt_file_gives_defaults(self):
        self.assertEqual(load_settings(self.path), EngineSettings())
        with open(self.path, "w", encoding="utf-8") as config_file:
            config_file.write("{not json")
        self.assertEqual(load_settings(self.path), EngineSettings())

    def test_model_settings_keep_the_tk_file_format(self):
        settings = EngineSettings(noise_gate_db=-40.0, pitch=3.0)
        save_model_settings(self.directory.name, settings)

        with open(
            os.path.join(self.directory.name, "realtime_settings.json"), encoding="utf-8"
        ) as saved_file:
            saved = json.load(saved_file)
        self.assertEqual(saved["threhold"], -40.0)
        self.assertNotIn("noise_gate_db", saved)
        loaded = load_model_settings(self.directory.name)
        self.assertEqual(loaded["noise_gate_db"], -40.0)
        self.assertEqual(loaded["pitch"], 3.0)


if __name__ == "__main__":
    unittest.main()
