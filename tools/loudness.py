"""Speech loudness of the converted output, for the output-level indicator.

Loudness follows ITU-R BS.1770 (LUFS): K-weighting, 400 ms blocks with 75 %
overlap, an absolute gate at -70 LUFS and a relative gate 10 LU below the
ungated level, so pauses between words do not pull the reading down.

The meter measures the model output *before* the output gain and adds the
current gain when asked.  Moving the Output gain slider therefore updates
the reading at once, and the recommendation stays valid without measuring
again.  Sample magnitudes are kept as a histogram for the same reason: how
often the soft clipper (tools/limiter.py) has to squash the voice can be
read off for any gain.

Two targets: voice chat at -21 LUFS (where the user's by-ear gains landed,
-19 to -24, through five models) and streaming at -16 LUFS, the usual level
of streamed voice.  Speech peaks sit about 20 dB above its loudness, so at
-16 the soft clipper rounds the loudest peaks a little (about 0.05 % of
samples reach full scale); at -21 peaks stay near -5 dBFS.  The voice only
counts as too loud by that measure once more than 1 % of samples would
exceed full scale, around -10 LUFS.
"""

import collections
import math
import threading

import numpy as np
from scipy.signal import lfilter, lfilter_zi

ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = -10.0
BLOCK_SECONDS = 0.4
HOP_SECONDS = 0.1
#: Blocks quieter than this before the gain are never speech at any gain.
FLOOR_LUFS = -100.0
#: Measured speech kept for the reading (the most recent blocks win).
WINDOW_SECONDS = 20.0
#: Speech needed before a verdict is given.
MIN_SPEECH_SECONDS = 5.0
#: Target speech loudness per setting, and the band around it that counts as good.
TARGETS_LUFS = {"voice_chat": -21.0, "streaming": -16.0}
DEFAULT_TARGET = "voice_chat"
GOOD_BAND_LU = 3.0
#: More than this share of speech samples over full scale (before the soft
#: clipper) is squashed audibly: too loud whatever the loudness says.
MAX_OVER_PERCENT = 1.0
#: Histogram of sample magnitudes, in dBFS before the gain.
HISTOGRAM_LOW_DB = -60.0
HISTOGRAM_HIGH_DB = 30.0
HISTOGRAM_STEP_DB = 0.5
HISTOGRAM_BINS = int((HISTOGRAM_HIGH_DB - HISTOGRAM_LOW_DB) / HISTOGRAM_STEP_DB)


def k_weighting(samplerate):
    """The two BS.1770 K-weighting biquads ``((b, a), (b, a))`` for any rate.

    libebur128's bilinear-transform design, which reproduces the 48 kHz
    coefficients published in BS.1770 exactly.
    """
    # Stage 1: high shelf (head acoustics).
    fc, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * fc / samplerate)
    vh = 10 ** (gain_db / 20)
    vb = vh**0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf_b = np.array([(vh + vb * k / q + k * k), 2 * (k * k - vh), (vh - vb * k / q + k * k)]) / a0
    shelf_a = np.array([1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0])
    # Stage 2: high pass (RLB weighting).
    fc, q = 38.13547087613982, 0.5003270373253953
    k = math.tan(math.pi * fc / samplerate)
    a0 = 1 + k / q + k * k
    high_b = np.array([1.0, -2.0, 1.0])
    high_a = np.array([1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0])
    return (shelf_b, shelf_a), (high_b, high_a)


def gated_loudness(block_loudness):
    """BS.1770 gated loudness of 400 ms block loudness values (LUFS)."""
    values = np.asarray(block_loudness, dtype=np.float64)
    values = values[values > ABSOLUTE_GATE_LUFS]
    if values.size == 0:
        return None
    ungated = 10 * np.log10(np.mean(10 ** (values / 10)))
    values = values[values > ungated + RELATIVE_GATE_LU]
    return float(10 * np.log10(np.mean(10 ** (values / 10))))


def integrated_loudness(audio, samplerate):
    """Loudness of a whole mono recording (LUFS), or ``None`` if silent."""
    meter = LoudnessMeter(samplerate, window_seconds=math.inf)
    meter.process(np.asarray(audio, dtype=np.float32))
    return gated_loudness(meter.block_loudness())


class LoudnessMeter:
    """Streaming speech loudness of a mono signal before the output gain.

    ``process`` runs on the audio thread and ``summary`` on the command
    thread; a lock keeps the block history consistent between them.
    """

    def __init__(self, samplerate, window_seconds=WINDOW_SECONDS):
        self.lock = threading.Lock()
        self.samplerate = samplerate
        self.hop = int(round(HOP_SECONDS * samplerate))
        self.hops_per_block = int(round(BLOCK_SECONDS / HOP_SECONDS))
        self.filters = k_weighting(samplerate)
        self.max_blocks = (
            None if math.isinf(window_seconds) else int(window_seconds / HOP_SECONDS)
        )
        self.reset()

    def reset(self):
        with self.lock:
            self._reset()

    def _reset(self):
        self.states = [lfilter_zi(b, a) * 0.0 for b, a in self.filters]
        self.pending = np.zeros(0, dtype=np.float64)
        self.pending_histogram = np.zeros(HISTOGRAM_BINS + 1, dtype=np.int64)
        # Mean square and sample histogram of each completed 100 ms hop.
        self.recent_hops = collections.deque(maxlen=self.hops_per_block)
        #: (block loudness LUFS, sample histogram of its newest hop)
        self.blocks = collections.deque(maxlen=self.max_blocks)

    def process(self, audio):
        """Add mono samples (float, before the output gain)."""
        with self.lock:
            self._process(np.asarray(audio, dtype=np.float64))

    def _process(self, audio):
        weighted = audio
        for index, (b, a) in enumerate(self.filters):
            weighted, self.states[index] = lfilter(b, a, weighted, zi=self.states[index])
        position = 0
        while position < audio.shape[0]:
            need = self.hop - self.pending.shape[0]
            take_weighted = weighted[position : position + need]
            take_raw = audio[position : position + need]
            self.pending = np.concatenate([self.pending, take_weighted**2])
            self.pending_histogram += magnitude_histogram(take_raw)
            position += take_weighted.shape[0]
            if self.pending.shape[0] == self.hop:
                self.finish_hop()

    def finish_hop(self):
        self.recent_hops.append((float(np.mean(self.pending)), self.pending_histogram))
        self.pending = np.zeros(0, dtype=np.float64)
        self.pending_histogram = np.zeros(HISTOGRAM_BINS + 1, dtype=np.int64)
        if len(self.recent_hops) < self.hops_per_block:
            return
        mean_square = np.mean([hop[0] for hop in self.recent_hops])
        loudness = -0.691 + 10 * math.log10(mean_square) if mean_square > 0 else -math.inf
        if loudness > FLOOR_LUFS:
            self.blocks.append((loudness, self.recent_hops[-1][1]))

    def block_loudness(self, gain_db=0.0):
        return [loudness + gain_db for loudness, _ in self.blocks]

    def summary(self, gain_db, target_lufs=TARGETS_LUFS[DEFAULT_TARGET]):
        """The reading at ``gain_db`` of output gain, for the protocol.

        ``{"lufs", "target_lufs", "speech_seconds", "over_percent",
        "verdict", "adjust_db"}``; ``verdict`` is ``None`` (not enough speech yet),
        ``"quiet"``, ``"good"`` or ``"loud"``, and ``adjust_db`` the change to
        the output gain that reaches the target (0 when good).
        """
        with self.lock:
            blocks = list(self.blocks)
        values = np.array([loudness + gain_db for loudness, _ in blocks])
        lufs = gated_loudness(values)
        if lufs is None:
            return {"lufs": None, "target_lufs": target_lufs, "speech_seconds": 0.0,
                    "over_percent": 0.0, "verdict": None, "adjust_db": 0.0}
        ungated = 10 * np.log10(np.mean(10 ** (values[values > ABSOLUTE_GATE_LUFS] / 10)))
        speech = (values > ABSOLUTE_GATE_LUFS) & (values > ungated + RELATIVE_GATE_LU)
        speech_seconds = float(speech.sum() * HOP_SECONDS)
        histogram = sum(
            (block[1] for block, keep in zip(blocks, speech) if keep),
            np.zeros(HISTOGRAM_BINS + 1, dtype=np.int64),
        )
        over = over_percent(histogram, gain_db)
        verdict, adjust = None, 0.0
        if speech_seconds >= MIN_SPEECH_SECONDS:
            verdict, adjust = judge(lufs, over, headroom_db(histogram, gain_db), target_lufs)
        return {
            "lufs": round(lufs, 1),
            "target_lufs": target_lufs,
            "speech_seconds": round(speech_seconds, 1),
            "over_percent": round(over, 2),
            "verdict": verdict,
            "adjust_db": adjust,
        }


def judge(lufs, over, headroom, target_lufs):
    """``(verdict, output-gain change in 0.5 dB steps)``.

    The change aims at ``target_lufs`` but never past ``headroom`` (how far the
    gain can rise before more than MAX_OVER_PERCENT of samples exceed full
    scale), so following a suggestion cannot make the voice "too loud".
    """
    change = floor_half(min(target_lufs - lufs, headroom))
    if over > MAX_OVER_PERCENT:
        return "loud", min(-0.5, change)
    if lufs < target_lufs - GOOD_BAND_LU:
        return "quiet", max(0.0, change)
    if lufs > target_lufs + GOOD_BAND_LU:
        return "loud", change
    return "good", 0.0


def round_half(value):
    return round(value * 2) / 2


def floor_half(value):
    return math.floor(value * 2) / 2


def magnitude_histogram(samples):
    """Counts of |sample| in HISTOGRAM_STEP_DB bins; bin 0 holds the quietest."""
    magnitude = np.abs(samples)
    db = 20 * np.log10(np.maximum(magnitude, 1e-12))
    bins = np.floor((db - HISTOGRAM_LOW_DB) / HISTOGRAM_STEP_DB).astype(np.int64) + 1
    bins = np.clip(bins, 0, HISTOGRAM_BINS)
    return np.bincount(bins, minlength=HISTOGRAM_BINS + 1)


def over_percent(histogram, gain_db):
    """Share of samples above full scale at ``gain_db`` of gain (%)."""
    total = histogram.sum()
    if total == 0:
        return 0.0
    # Bin i (>= 1) starts at HISTOGRAM_LOW_DB + (i - 1) * step before gain.
    first = int(math.ceil((-gain_db - HISTOGRAM_LOW_DB) / HISTOGRAM_STEP_DB)) + 1
    first = min(max(first, 1), HISTOGRAM_BINS + 1)
    return float(histogram[first:].sum() * 100.0 / total)


def headroom_db(histogram, gain_db):
    """How far the gain can change (dB, may be negative) while at most
    MAX_OVER_PERCENT of samples exceed full scale."""
    total = histogram.sum()
    if total == 0:
        return math.inf
    # Samples at or above each bin's lower edge, loudest first.
    at_or_above = np.cumsum(histogram[::-1])[::-1]
    allowed = total * MAX_OVER_PERCENT / 100.0
    # The quietest bin edge with no more than `allowed` samples at or above it.
    index = int(np.argmax(at_or_above <= allowed))
    if at_or_above[index] > allowed:
        return math.inf
    edge_db = HISTOGRAM_LOW_DB + (index - 1) * HISTOGRAM_STEP_DB
    return -edge_db - gain_db
