"""Recommend a pitch shift that moves the user's voice onto a model's voice.

Both sides are summarised by their median pitch: the source speaker's from a
clip analysed once per model, the user's from live input while converting.
The recommended shift puts the user's median on the source median:

    shift = 12 * log2(source_hz / voice_hz)     (semitones, 0.1 steps)

Medians move with what a clip contains (calm vs. animated speech), so the
result is a starting point to fine-tune by ear, good to about +-0.3 semitones.
"""

import json
import math
import os
import subprocess

import numpy as np

#: Pitch histogram range and resolution, in semitones (MIDI note numbers).
LOWEST_NOTE = 24.0  # ~33 Hz
HIGHEST_NOTE = 96.0  # ~1047 Hz
BIN_SEMITONES = 0.1
BIN_COUNT = int(round((HIGHEST_NOTE - LOWEST_NOTE) / BIN_SEMITONES))
FRAMES_PER_SECOND = 100  # RVC pitch frames are 10 ms
#: Older speech fades out once this much voiced speech has been heard, so the
#: median follows the current mic and voice (five minutes of voicing).
VOICE_MEMORY_SECONDS = 300.0
#: Voiced speech needed before a median is trusted for a recommendation.
MIN_VOICED_SECONDS = 20.0
#: RMVPE runs on pieces this long, so each GPU burst stays short.
ANALYSIS_PIECE_SECONDS = 10.0
ANALYSIS_RATE = 16000


def hz_to_note(hz):
    return 69.0 + 12.0 * np.log2(np.asarray(hz, dtype=np.float64) / 440.0)


def note_to_hz(note):
    return float(440.0 * 2.0 ** ((note - 69.0) / 12.0))


def recommended_shift(voice_hz, source_hz):
    """Semitones moving ``voice_hz`` onto ``source_hz``, rounded to 0.1."""
    if not voice_hz or not source_hz or voice_hz <= 0 or source_hz <= 0:
        return None
    shift = 12.0 * math.log2(source_hz / voice_hz)
    return round(min(max(shift, -24.0), 24.0), 1)


class PitchHistogram:
    """Recency-weighted histogram of voiced pitch, for a running median."""

    def __init__(self, memory_seconds=VOICE_MEMORY_SECONDS):
        self.counts = np.zeros(BIN_COUNT, dtype=np.float64)
        self.capacity = memory_seconds * FRAMES_PER_SECOND
        self.changed = False

    def add(self, f0_hz):
        """Add pitch frames in Hz; unvoiced (0) and out-of-range frames are ignored."""
        f0_hz = np.asarray(f0_hz, dtype=np.float64).ravel()
        f0_hz = f0_hz[np.isfinite(f0_hz) & (f0_hz > 0)]
        if f0_hz.size == 0:
            return
        bins = np.floor((hz_to_note(f0_hz) - LOWEST_NOTE) / BIN_SEMITONES).astype(np.int64)
        bins = bins[(bins >= 0) & (bins < BIN_COUNT)]
        if bins.size == 0:
            return
        self.counts += np.bincount(bins, minlength=BIN_COUNT)
        total = self.counts.sum()
        if total > self.capacity:
            self.counts *= self.capacity / total
        self.changed = True

    @property
    def voiced_seconds(self):
        return float(self.counts.sum()) / FRAMES_PER_SECOND

    def median_hz(self):
        """Median pitch, interpolated within its bin; ``None`` when empty."""
        total = self.counts.sum()
        if total <= 0:
            return None
        cumulative = np.cumsum(self.counts)
        index = int(np.searchsorted(cumulative, total / 2.0))
        before = cumulative[index - 1] if index > 0 else 0.0
        fraction = (total / 2.0 - before) / max(self.counts[index], 1e-12)
        return note_to_hz(LOWEST_NOTE + (index + fraction) * BIN_SEMITONES)

    def summary(self):
        """``{"median_hz", "seconds", "needed_seconds"}`` for the protocol;
        the median is ``None`` until ``needed_seconds`` of voicing is heard."""
        seconds = self.voiced_seconds
        median = self.median_hz() if seconds >= MIN_VOICED_SECONDS else None
        return {
            "median_hz": None if median is None else round(median, 2),
            "seconds": round(seconds, 1),
            "needed_seconds": MIN_VOICED_SECONDS,
        }

    def reset(self):
        self.counts[:] = 0.0
        self.changed = True

    def save(self, path):
        from engine.settings import write_json_atomic

        nonzero = np.flatnonzero(self.counts)
        write_json_atomic(
            path,
            {
                "bin_semitones": BIN_SEMITONES,
                "lowest_note": LOWEST_NOTE,
                "counts": {str(int(i)): round(float(self.counts[i]), 3) for i in nonzero},
            },
        )
        self.changed = False

    @classmethod
    def load(cls, path):
        histogram = cls()
        try:
            with open(path, "r", encoding="utf-8") as saved_file:
                saved = json.load(saved_file)
            if (
                saved.get("bin_semitones") != BIN_SEMITONES
                or saved.get("lowest_note") != LOWEST_NOTE
            ):
                return histogram
            for key, value in saved.get("counts", {}).items():
                index = int(key)
                if 0 <= index < BIN_COUNT and isinstance(value, (int, float)) and value > 0:
                    histogram.counts[index] = float(value)
        except (OSError, ValueError, AttributeError, TypeError):
            return cls()
        return histogram


AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".webm")


def source_files(paths):
    """Audio files to analyse: each file given, plus the audio files directly
    inside each folder given (not subfolders, which datasets use for other
    things, such as validation clips of other voices)."""
    files = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(
                os.path.join(path, name)
                for name in sorted(os.listdir(path))
                if name.lower().endswith(AUDIO_EXTENSIONS)
                and os.path.isfile(os.path.join(path, name))
            )
        elif os.path.isfile(path):
            files.append(path)
        else:
            raise FileNotFoundError(path)
    return files


def decoded_pieces(ffmpeg_path, path, piece):
    """Decode a file as 16 kHz mono float32, yielding ``piece``-sample arrays."""
    command = [
        ffmpeg_path, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", path,
        "-vn", "-ac", "1", "-ar", str(ANALYSIS_RATE), "-f", "f32le", "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            data = process.stdout.read(piece * 4)
            if not data:
                break
            yield np.frombuffer(data[: len(data) // 4 * 4], dtype=np.float32)
        error = process.stderr.read().decode("utf-8", "replace").strip()
        if process.wait() != 0:
            lines = error.splitlines()
            raise ValueError(lines[-1] if lines else f"ffmpeg could not decode {path}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()


def measure_source_pitch(ffmpeg_path, paths, detect_f0, pause=lambda: None):
    """Median voiced pitch of every frame of the given files (or folders).

    ``detect_f0(audio_16k) -> f0 Hz per 10 ms frame (0 = unvoiced)``.  Each
    file is analysed in full: a speaker's pitch drifts over a long recording,
    so samples of it are not accurate enough for 0.1-semitone steps.
    ``pause()`` runs between pieces (to share the GPU with live conversion).
    Returns ``(median_hz, voiced_seconds)``; median is ``None`` without voice.
    """
    files = source_files(paths)
    if not files:
        raise FileNotFoundError("no audio files")
    piece = int(ANALYSIS_PIECE_SECONDS * ANALYSIS_RATE)
    voiced = []
    for path in files:
        for chunk in decoded_pieces(ffmpeg_path, path, piece):
            if chunk.shape[0] < ANALYSIS_RATE // 2:
                continue
            if chunk.shape[0] < piece:
                # Same shape every call (CUDA Graphs); padding is silent.
                chunk = np.pad(chunk, (0, piece - chunk.shape[0]))
            f0 = np.asarray(detect_f0(chunk), dtype=np.float64)
            voiced.append(f0[np.isfinite(f0) & (f0 > 0)])
            pause()
    voiced = np.concatenate(voiced) if voiced else np.zeros(0)
    if voiced.size == 0:
        return None, 0.0
    # The median of log-pitch equals the log of the median pitch.
    return float(np.median(voiced)), voiced.size / FRAMES_PER_SECOND
