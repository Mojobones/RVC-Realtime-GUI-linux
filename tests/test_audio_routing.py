import unittest

import numpy as np

from tools.audio_routing import (
    ALSA_API,
    JACK_API,
    is_native_api,
    scatter_mono,
    select_channels,
)


class AudioRoutingTest(unittest.TestCase):
    def test_only_jack_uses_native_period(self):
        self.assertTrue(is_native_api(JACK_API))
        self.assertFalse(is_native_api(ALSA_API))
        self.assertFalse(is_native_api("File"))

    def test_select_channels_picks_selected_jack_ports(self):
        indata = np.arange(12, dtype=np.float32).reshape(3, 4)

        selected = select_channels(indata, [2, 3])

        np.testing.assert_array_equal(selected, indata[:, 2:4])

    def test_select_channels_duplicates_mono_port(self):
        indata = np.arange(3, dtype=np.float32).reshape(3, 1)

        selected = select_channels(indata, [0, 0])

        self.assertEqual(selected.shape, (3, 2))
        np.testing.assert_array_equal(selected[:, 0], selected[:, 1])

    def test_select_channels_without_selectors_keeps_all_columns(self):
        indata = np.zeros((4, 2), dtype=np.float32)

        self.assertIs(select_channels(indata, []), indata)

    def test_scatter_mono_writes_only_selected_ports(self):
        outdata = np.zeros((4, 4), dtype=np.float32)
        mono = np.ones(4, dtype=np.float32)

        scatter_mono(outdata, mono, [2, 3])

        np.testing.assert_array_equal(outdata[:, :2], 0.0)
        np.testing.assert_array_equal(outdata[:, 2:], 1.0)

    def test_scatter_mono_short_block_leaves_tail_silent(self):
        outdata = np.zeros((4, 2), dtype=np.float32)
        mono = np.ones(2, dtype=np.float32)

        scatter_mono(outdata, mono, [])

        np.testing.assert_array_equal(outdata[:2], 1.0)
        np.testing.assert_array_equal(outdata[2:], 0.0)


if __name__ == "__main__":
    unittest.main()
