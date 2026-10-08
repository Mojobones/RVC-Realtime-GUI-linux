import unittest

import numpy as np

from tools.context_bench import RATE, find_pauses, insert_pauses


class ContextBenchTest(unittest.TestCase):
    def test_finds_pauses_and_maps_onsets_after_inserted_silence(self):
        rng = np.random.default_rng(0)
        speech = lambda seconds: (rng.standard_normal(int(seconds * RATE)) * 0.1).astype(np.float32)
        silence = lambda seconds: np.zeros(int(seconds * RATE), dtype=np.float32)
        audio = np.concatenate([speech(1.0), silence(0.4), speech(1.0), silence(0.5), speech(1.0)])

        pauses = find_pauses(audio)
        longer, onsets_a, onsets_b = insert_pauses(audio, pauses, 3.0)

        self.assertEqual(len(pauses), 2)
        self.assertEqual(longer.shape[0], audio.shape[0] + 2 * 3 * RATE)
        for onset_a, onset_b in zip(onsets_a, onsets_b):
            # The same speech follows each onset in both versions.
            np.testing.assert_array_equal(
                longer[onset_b : onset_b + 1000], audio[onset_a : onset_a + 1000]
            )
            self.assertLess(np.abs(longer[onset_b - 3 * RATE : onset_b - RATE]).max(), 1e-9)


if __name__ == "__main__":
    unittest.main()
