"""Streaming RNNoise (Xiph) through ctypes.

Uses the system library (``sudo pacman -S rnnoise`` on Arch/CachyOS).  RNNoise
processes 480-sample (10 ms) frames at 48 kHz, with samples on a 16-bit scale,
and keeps a recurrent state, so one ``RNNoise`` instance must see the input
stream in order.
"""

import ctypes
import ctypes.util

import numpy as np

SAMPLE_RATE = 48000
FRAME_SIZE = 480
_SCALE = 32768.0


def _load_library():
    for name in (ctypes.util.find_library("rnnoise"), "librnnoise.so.0", "librnnoise.so"):
        if not name:
            continue
        try:
            library = ctypes.CDLL(name)
        except OSError:
            continue
        library.rnnoise_create.restype = ctypes.c_void_p
        library.rnnoise_create.argtypes = [ctypes.c_void_p]
        library.rnnoise_destroy.restype = None
        library.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        library.rnnoise_process_frame.restype = ctypes.c_float
        library.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        return library
    return None


_LIBRARY = _load_library()


def available():
    return _LIBRARY is not None


class RNNoise:
    """One denoiser state; call ``process`` with consecutive 48 kHz audio."""

    def __init__(self):
        if _LIBRARY is None:
            raise RuntimeError("librnnoise was not found; install the rnnoise package")
        self._state = _LIBRARY.rnnoise_create(None)
        if not self._state:
            raise RuntimeError("rnnoise_create failed")
        self._frame = np.zeros(FRAME_SIZE, dtype=np.float32)
        self._pointer = self._frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        #: Voice probability of each 10 ms frame of the last ``process`` call.
        self.voice_probabilities = np.zeros(0, dtype=np.float32)

    def process(self, audio):
        """Denoise float32 audio in [-1, 1] whose length is a multiple of 480."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.shape[0] % FRAME_SIZE:
            raise ValueError(f"RNNoise needs a multiple of {FRAME_SIZE} samples")
        output = np.empty_like(audio)
        probabilities = np.empty(audio.shape[0] // FRAME_SIZE, dtype=np.float32)
        for index, start in enumerate(range(0, audio.shape[0], FRAME_SIZE)):
            self._frame[:] = audio[start : start + FRAME_SIZE] * _SCALE
            probabilities[index] = _LIBRARY.rnnoise_process_frame(
                self._state, self._pointer, self._pointer
            )
            output[start : start + FRAME_SIZE] = self._frame / _SCALE
        self.voice_probabilities = probabilities
        return output

    def close(self):
        if self._state:
            _LIBRARY.rnnoise_destroy(self._state)
            self._state = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
