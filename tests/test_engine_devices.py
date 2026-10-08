import unittest
from types import SimpleNamespace

from engine.devices import DeviceCatalog, build_endpoints

HOSTAPIS = [
    {"name": "ALSA", "devices": [0, 1, 2]},
    {"name": "JACK Audio Connection Kit", "devices": [3, 4, 5]},
]
DEVICES = [
    {"name": "HD-Audio Generic: ALC897 Analog (hw:3,0)", "max_input_channels": 2, "max_output_channels": 6},
    {"name": "pipewire", "max_input_channels": 128, "max_output_channels": 128},
    {"name": "default", "max_input_channels": 128, "max_output_channels": 128},
    {"name": "Elgato Wave XLR Mono", "max_input_channels": 1, "max_output_channels": 0},
    {"name": "Creative BT-W6 Analog Stereo", "max_input_channels": 2, "max_output_channels": 2},
    {"name": "GB207 Pro", "max_input_channels": 0, "max_output_channels": 8},
]


class BuildEndpointsTest(unittest.TestCase):
    def test_hides_raw_alsa_devices_and_lists_jack_first(self):
        inputs, outputs = build_endpoints(DEVICES, HOSTAPIS)

        self.assertNotIn("[ALSA] HD-Audio Generic: ALC897 Analog (hw:3,0)", inputs)
        self.assertEqual(
            list(inputs),
            [
                "[JACK] Creative BT-W6 Analog Stereo",
                "[JACK] Elgato Wave XLR Mono",
                "[ALSA] default",
                "[ALSA] pipewire",
            ],
        )
        self.assertTrue(list(outputs)[0].startswith("[JACK]"))

    def test_show_all_includes_raw_alsa_devices(self):
        inputs, _ = build_endpoints(DEVICES, HOSTAPIS, show_all=True)

        self.assertIn("[ALSA] HD-Audio Generic: ALC897 Analog (hw:3,0)", inputs)

    def test_jack_channel_pairs_and_selectors(self):
        inputs, outputs = build_endpoints(DEVICES, HOSTAPIS)

        self.assertEqual(inputs["[JACK] Elgato Wave XLR Mono"].selectors, [0, 0])
        self.assertEqual(inputs["[JACK] Creative BT-W6 Analog Stereo"].selectors, [0, 1])
        self.assertEqual(outputs["[JACK] GB207 Pro — 3 / 4"].selectors, [2, 3])
        self.assertEqual(outputs["[JACK] GB207 Pro — 7 / 8"].index, 5)
        self.assertNotIn("[JACK] GB207 Pro", outputs)
        self.assertEqual(outputs["[ALSA] pipewire"].selectors, [])


class FakeSoundDevice:
    def __init__(self):
        self.default = SimpleNamespace(device=(2, 2))

    def _terminate(self):
        pass

    def _initialize(self):
        pass

    def query_devices(self):
        return DEVICES

    def query_hostapis(self):
        return HOSTAPIS


class DeviceCatalogTest(unittest.TestCase):
    def setUp(self):
        self.catalog = DeviceCatalog(FakeSoundDevice())
        self.catalog.refresh()

    def test_resolve_keeps_existing_label(self):
        label = "[JACK] Elgato Wave XLR Mono"
        self.assertEqual(self.catalog.resolve(label, "input"), label)

    def test_resolve_matches_same_device_when_pair_label_changes(self):
        self.assertEqual(
            self.catalog.resolve("[JACK] GB207 Pro", "output"),
            "[JACK] GB207 Pro — 1 / 2",
        )

    def test_resolve_falls_back_to_portaudio_default(self):
        self.assertEqual(self.catalog.resolve("[JACK] Unplugged", "input"), "[ALSA] default")
        self.assertEqual(self.catalog.resolve("", "output"), "[ALSA] default")

    def test_missing_monitor_is_disabled(self):
        self.assertIsNone(self.catalog.resolve_monitor("[JACK] Unplugged"))
        self.assertIsNone(self.catalog.resolve_monitor(None))

    def test_to_dict_reports_one_based_channels(self):
        outputs = {item["label"]: item for item in self.catalog.to_dict()["outputs"]}

        self.assertEqual(outputs["[JACK] GB207 Pro — 3 / 4"]["channels"], [3, 4])
        self.assertEqual(outputs["[JACK] GB207 Pro — 3 / 4"]["api"], "JACK")


if __name__ == "__main__":
    unittest.main()
