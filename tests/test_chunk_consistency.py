"""Consecutive real-time chunks must render the same moment identically.

Each inference re-renders the end of the previous chunk.  With time-aligned
noise and a continuous harmonic phase (infer/rtrvc.py, CHUNK_CONSISTENCY),
the overlapping part of two chunks should agree before WSOLA splices them.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from infer import rtrvc
from infer.module.models import SineGen

SR = 40000
UPP = SR // 100  # samples per 10 ms frame


def f0_track(frames):
    """A gliding pitch with an unvoiced gap, in Hz per 10 ms frame."""
    f0 = 140 + 60 * np.sin(np.linspace(0, 3, frames))
    f0[40:46] = 0.0
    return torch.tensor(f0, dtype=torch.float32)


def render(gen, f0, phase0=None):
    silent = torch.zeros(f0.shape[0] * UPP)
    phase = None if phase0 is None else torch.tensor([phase0])
    out, _, _ = gen(f0[None, :], UPP, noise=silent, phase0=phase)
    return out[0, :, 0]


class SineGenPhaseTest(unittest.TestCase):
    def setUp(self):
        self.gen = SineGen(SR, harmonic_num=0, sine_amp=0.1, noise_std=0.003)
        self.f0 = f0_track(120)
        self.frames, self.shift = 60, 25  # window and per-chunk advance

    def overlap(self, carry_phase):
        first = render(self.gen, self.f0[: self.frames], 0.0)
        phase = float((self.f0[: self.shift].sum() * UPP / SR) % 1.0) if carry_phase else 0.0
        second = render(self.gen, self.f0[self.shift : self.shift + self.frames], phase)
        overlap = (self.frames - self.shift) * UPP
        return first[self.shift * UPP :], second[:overlap]

    def test_carried_phase_makes_overlapping_chunks_agree(self):
        earlier, later = self.overlap(carry_phase=True)
        np.testing.assert_allclose(later.numpy(), earlier.numpy(), atol=2e-3)

    def test_without_carried_phase_the_overlap_disagrees(self):
        earlier, later = self.overlap(carry_phase=False)
        self.assertGreater(float((later - earlier).abs().max()), 0.05)

    def test_defaults_match_the_original_behaviour(self):
        f0 = self.f0[:30][None, :]
        torch.manual_seed(5)
        default, _, _ = self.gen(f0, UPP)
        torch.manual_seed(5)
        torch.rand(1, 1)  # SineGen draws rand_ini before its noise
        explicit_noise = torch.randn(1, 30 * UPP, 1)
        explicit, _, _ = self.gen(f0, UPP, noise=explicit_noise, phase0=torch.tensor([0.0]))
        torch.testing.assert_close(default, explicit)


class StreamNoiseTest(unittest.TestCase):
    def rvc(self):
        engine = rtrvc.RVC.__new__(rtrvc.RVC)
        engine.cache_pitch = torch.arange(1024, dtype=torch.long)
        engine.cache_pitchf = torch.arange(1024, dtype=torch.float32)
        engine.net_g = SimpleNamespace(dec=SimpleNamespace(upp=UPP))
        engine.reset_stream_state()
        engine.cache_noise = torch.randn(192, 1024)
        engine.source_noise = torch.randn(30 * UPP)
        return engine

    def test_advance_keeps_shared_frames_and_draws_new_ones(self):
        engine, shift = self.rvc(), 10
        noise_before = engine.cache_noise.clone()
        source_before = engine.source_noise.clone()
        pitch_before = engine.cache_pitch.clone()

        engine.advance(shift)

        torch.testing.assert_close(engine.cache_noise[:, :-shift], noise_before[:, shift:])
        torch.testing.assert_close(engine.source_noise[: -shift * UPP], source_before[shift * UPP :])
        torch.testing.assert_close(engine.cache_pitch[:-shift], pitch_before[shift:])
        self.assertFalse(torch.equal(engine.cache_noise[:, -shift:], noise_before[:, -shift:]))
        self.assertFalse(torch.equal(engine.source_noise[-shift * UPP :], source_before[-shift * UPP :]))

    def test_reset_forgets_the_stream(self):
        engine = self.rvc()
        engine.phase_start = 0.4
        engine.reset_stream_state()
        self.assertIsNone(engine.cache_noise)
        self.assertIsNone(engine.source_noise)
        self.assertEqual(engine.phase_start, 0.0)


if __name__ == "__main__":
    unittest.main()
