"""WSOLA splicing of consecutive real-time inference chunks.

Each inference returns ``block + FADE + search`` samples (search = 10 ms).
The head of each chunk overlaps the previous chunk's *natural continuation*:
the samples it produced right after its emitted block.  WSOLA picks the
offset in ``[0, search]`` whose waveform best matches that continuation,
crossfades over a fixed 40 ms, and emits ``block`` samples.

The last ``search - offset`` samples of every chunk are never used.  They are
the model's causal edge, where it has no future context and edge artifacts
live, so splicing early in the window (small offsets) keeps the continuation
template clear of them.  Compared with the previous SOLA splice this

- treats the previous seam's offset as the expected position: a nearby,
  almost-as-good match is preferred over a distant one, and
- keeps the previous offset when no candidate is a confident match (silence,
  noise, unvoiced consonants), instead of jumping to a random argmax that can
  drag the template into the chunk's edge;
- reports a real confidence (correlation normalised by both energies).
"""

import numpy as np
import torch
import torch.nn.functional as F

FADE_SECONDS = 0.040
SEARCH_SECONDS = 0.010
#: Below this normalised correlation the seam is not a meaningful match.
#: Tuned on real RVC output (tools/splice_bench.py): 0.3 discarded usable
#: weak matches; 0.15 keeps them while still ignoring unmatched noise.
CONFIDENCE_THRESHOLD = 0.15
#: Score penalty for moving the splice by the whole search span.  0.05 cut
#: the time skipped or repeated at seams by about 20% with no loss of seam
#: confidence or spectral continuity on real RVC output.
CONTINUITY_PENALTY = 0.05
#: Mean template power below this (about -80 dBFS) counts as silence.
SILENCE_POWER = 1e-8


class WsolaSplicer:
    def __init__(self, samplerate, block, device, dtype=torch.float32):
        zc = samplerate // 100
        # Read at construction so benchmarks can try other lengths.
        self.fade = int(round(FADE_SECONDS * 100)) * zc
        # One 10 ms frame, so the inference length stays a whole number of
        # frames at any rate (zc is odd at 44.1 kHz).
        self.search = zc
        self.block = block
        self.device = device
        ramp = torch.linspace(0.0, 1.0, steps=self.fade, device=device, dtype=dtype)
        self.fade_in = torch.sin(0.5 * np.pi * ramp) ** 2
        self.fade_out = 1 - self.fade_in
        self.ones = torch.ones(1, 1, self.fade, device=device, dtype=dtype)
        self.candidates = torch.arange(self.search + 1, device=device, dtype=dtype)
        self.template = torch.zeros(self.fade, device=device, dtype=dtype)
        self.last_offset = 0
        self.last_confidence = 0.0

    @property
    def input_length(self):
        """Samples each inference must return."""
        return self.block + self.fade + self.search

    def reset(self):
        self.template.zero_()
        self.last_offset = 0
        self.last_confidence = 0.0

    def find_offset(self, infer_wav):
        """Return ``(offset, confidence)`` for splicing ``infer_wav``."""
        window = infer_wav[None, None, : self.fade + self.search]
        numerator = F.conv1d(window, self.template[None, None, :])[0, 0]
        candidate_energy = F.conv1d(window**2, self.ones)[0, 0]
        template_energy = torch.sum(self.template**2)
        ncc = numerator / torch.sqrt(candidate_energy * template_energy + 1e-12)
        penalty = CONTINUITY_PENALTY * (self.candidates - self.last_offset).abs() / self.search
        best = torch.argmax(ncc - penalty)
        best_offset, confidence, template_power = torch.stack(
            (best.to(ncc.dtype), ncc[best], template_energy / self.fade)
        ).tolist()
        if template_power < SILENCE_POWER or confidence < CONFIDENCE_THRESHOLD:
            return self.last_offset, confidence
        return int(best_offset), confidence

    def splice(self, infer_wav, crossfade=True):
        """Crossfade ``infer_wav`` onto the previous chunk; return ``block`` samples.

        ``crossfade=False`` starts fresh after silence (no previous chunk to
        join), so the head of the chunk is not faded in over 40 ms.
        """
        if not crossfade:
            self.template.zero_()
            self.last_offset, self.last_confidence = 0, 0.0
            output = infer_wav
            self.template[:] = output[self.block : self.block + self.fade]
            return output[: self.block]
        offset, confidence = self.find_offset(infer_wav)
        self.last_offset, self.last_confidence = offset, confidence
        output = infer_wav[offset:]
        # The samples right after the emitted block are the natural
        # continuation the next chunk is matched against.  Copy them before
        # fading: with a block shorter than the fade the regions overlap.
        continuation = output[self.block : self.block + self.fade].clone()
        output[: self.fade] *= self.fade_in
        output[: self.fade] += self.template * self.fade_out
        self.template[:] = continuation
        return output[: self.block]
