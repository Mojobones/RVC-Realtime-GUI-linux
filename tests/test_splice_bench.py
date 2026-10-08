import json
import math
import os
import tempfile
import unittest

import numpy as np

from tools.splice import WsolaSplicer
from tools.splice_bench import SolaSplicer, compare, replay

RATE = 48000
BLOCK = 4800


def sine_chunks(count, length, rng):
    t = np.arange((count + 2) * BLOCK + length)
    signal = (0.3 * np.sin(2 * math.pi * 150.0 * t / RATE)).astype(np.float32)
    return [
        signal[n * BLOCK + int(rng.integers(0, 96)) :][:length].copy() for n in range(count)
    ]


class SpliceBenchTest(unittest.TestCase):
    def test_replay_runs_both_algorithms_on_the_same_chunks(self):
        import torch

        length = WsolaSplicer(RATE, BLOCK, torch.device("cpu")).input_length
        chunks = sine_chunks(6, length, np.random.default_rng(0))
        for splicer_class in (SolaSplicer, WsolaSplicer):
            audio, confidences, offsets = replay(chunks, RATE, BLOCK, splicer_class)
            self.assertEqual(audio.shape[0], 6 * BLOCK)
            self.assertEqual(confidences.shape, offsets.shape)
            self.assertGreater(confidences.min(), 0.99)

    def test_compare_writes_wavs_and_reports_both(self):
        import torch

        length = WsolaSplicer(RATE, BLOCK, torch.device("cpu")).input_length
        chunks = sine_chunks(8, length, np.random.default_rng(1))
        with tempfile.TemporaryDirectory() as directory:
            np.savez_compressed(
                os.path.join(directory, "chunks.npz"),
                **{str(i): chunk for i, chunk in enumerate(chunks)},
            )
            with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as meta:
                json.dump({"samplerate": RATE, "block": BLOCK}, meta)

            results = compare(directory)

            self.assertEqual(set(results), {"sola", "wsola"})
            for name in results:
                self.assertTrue(os.path.isfile(os.path.join(directory, f"{name}.wav")))


if __name__ == "__main__":
    unittest.main()
