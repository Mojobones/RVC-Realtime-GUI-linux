"""The decoder renders context frames before each chunk and drops them.

A stand-in synthesizer stamps every output sample with its absolute frame
number, so the test can see exactly which frames ``RVC.infer`` returns.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from infer import rtrvc

TGT_SR = 32000
UPP = TGT_SR // 100  # samples per 10 ms frame


class FrameStampingSynth:
    inter_channels = 4
    dec = SimpleNamespace(upp=UPP)

    def __init__(self):
        self.calls = []

    def infer(self, phone, lengths, pitch, nsff0, sid, skip_head, return_length, return_length2,
              noise=None, source_noise=None, phase0=None):
        self.calls.append((skip_head, return_length, return_length2, source_noise.shape[0]))
        frames = torch.arange(skip_head, skip_head + return_length2, dtype=torch.float32)
        return (frames.repeat_interleave(UPP)[None, None, :],)


def stub_rvc():
    rvc = rtrvc.RVC.__new__(rtrvc.RVC)
    rvc.config = SimpleNamespace(is_half=False)
    rvc.device = torch.device("cpu")
    rvc.model, rvc.version, rvc.if_f0 = None, "v2", 1
    rvc.f0_up_key, rvc.formant_shift, rvc.index_rate = 0.0, 0.0, 0.0
    rvc.protect, rvc.infer_count = 0.5, 0
    rvc.tgt_sr, rvc.resample_kernel = TGT_SR, {}
    rvc.cache_pitch = torch.zeros(1024, dtype=torch.long)
    rvc.cache_pitchf = torch.zeros(1024)
    rvc.net_g = FrameStampingSynth()
    rvc.reset_stream_state()

    def get_f0(x, key, method):
        frames = x.shape[0] // 160 + 1
        rvc.last_f0 = None
        return torch.full((frames,), 60, dtype=torch.long), torch.full((frames,), 150.0)

    rvc.get_f0 = get_f0
    return rvc


def run(rvc, skip_head=100, return_length=20, block_frames=14):
    input_wav = torch.zeros(160 * (skip_head + return_length))

    def features(model, feats, version, padding_mask=None):
        # One 20 ms feature per 320 samples, as HuBERT produces.
        return torch.zeros(1, feats.shape[1] // 320, 768)

    with mock.patch.object(rtrvc, "extract_hubert_features", features):
        return rvc.infer(input_wav, 160 * block_frames, skip_head, return_length, "rmvpe")


class DecoderContextTest(unittest.TestCase):
    def test_context_frames_are_rendered_and_dropped(self):
        rvc = stub_rvc()
        with mock.patch.object(rtrvc, "DECODER_CONTEXT_FRAMES", 8):
            audio = run(rvc, skip_head=100, return_length=20)
        # The decoder was asked for 8 frames before the window...
        skip, length, length2, source = rvc.net_g.calls[-1]
        self.assertEqual((skip, length, length2), (92, 28, 28))
        self.assertEqual(source, 28 * UPP)  # time-aligned source noise covers them
        # ...and only the window's own frames come back.
        self.assertEqual(audio.shape[0], 20 * UPP)
        self.assertEqual(float(audio[0]), 100.0)
        self.assertEqual(float(audio[-1]), 119.0)

    def test_without_context_the_window_is_unchanged(self):
        rvc = stub_rvc()
        with mock.patch.object(rtrvc, "DECODER_CONTEXT_FRAMES", 0):
            audio = run(rvc)
        self.assertEqual(rvc.net_g.calls[-1][:3], (100, 20, 20))
        self.assertEqual((float(audio[0]), audio.shape[0]), (100.0, 20 * UPP))

    def test_context_is_limited_to_the_frames_that_exist(self):
        rvc = stub_rvc()
        with mock.patch.object(rtrvc, "DECODER_CONTEXT_FRAMES", 8):
            audio = run(rvc, skip_head=5, return_length=20)
        self.assertEqual(rvc.net_g.calls[-1][:3], (0, 25, 25))
        self.assertEqual(float(audio[0]), 5.0)

    def test_carried_phase_advances_from_the_decoded_start(self):
        # 150 Hz for 14 frames from the decoded start: the phase carried to
        # the next chunk is the decoded start's phase plus that advance.
        rvc = stub_rvc()
        with mock.patch.object(rtrvc, "DECODER_CONTEXT_FRAMES", 8):
            run(rvc, block_frames=14)
        expected = (150.0 * 14 * UPP / TGT_SR) % 1.0
        self.assertAlmostEqual(rvc.phase_start, expected, places=5)

    def test_default_is_80_ms(self):
        self.assertEqual(rtrvc.DECODER_CONTEXT_FRAMES, 8)


if __name__ == "__main__":
    unittest.main()
