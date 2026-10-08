"""Compare input noise reduction methods on a clean recording plus known noise.

    .venv/bin/python -m tools.denoise_bench --input speech.wav [--snr 15]

Fan-like coloured noise, mains hum and keyboard-like clicks are mixed into the
recording at ``--snr`` dB.  Each method runs through the engine's real input
path (passthrough mode, no model), and is scored against the clean recording:
SI-SDR of the speech (higher = cleaner, undistorted voice) and the level of
noise left in the pauses (lower = quieter background).  WAVs are written for
listening.
"""

import argparse
import os
import sys

import numpy as np

RATE = 48000
DEFAULT_DIR = os.path.join("logs", "denoise_bench")


def make_noise(length, rng):
    white = rng.standard_normal(length)
    # Fan-like: white noise through a gentle low-pass (about -6 dB/octave above 300 Hz).
    fan = np.zeros(length)
    alpha = np.exp(-2 * np.pi * 300 / RATE)
    for i in range(1, length):
        fan[i] = alpha * fan[i - 1] + (1 - alpha) * white[i]
    fan /= np.std(fan)
    t = np.arange(length) / RATE
    hum = sum(np.sin(2 * np.pi * 60 * k * t) / k for k in (1, 2, 3, 5))
    hum /= np.std(hum)
    clicks = np.zeros(length)
    for start in rng.integers(0, length - 400, size=int(length / RATE * 6)):
        clicks[start : start + 400] += rng.standard_normal(400) * np.exp(-np.arange(400) / 60)
    clicks /= np.std(clicks) + 1e-12
    return (fan + 0.4 * hum + 0.8 * clicks).astype(np.float32)


def si_sdr(estimate, reference):
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    scale = np.dot(estimate, reference) / (np.dot(reference, reference) + 1e-12)
    target = scale * reference
    return 10 * np.log10(np.sum(target**2) / (np.sum((estimate - target) ** 2) + 1e-12))


def run_engine(audio, denoiser):
    """Engine input path in passthrough mode; returns output aligned to the input."""
    from unittest import mock

    from tests.test_engine_pipeline import passthrough_engine
    from tools import rnnoise

    patch = mock.patch.object(rnnoise, "available", return_value=denoiser == "rnnoise")
    with patch:
        engine = passthrough_engine(block_time=0.1)
    engine.settings.input_denoise = denoiser != "none"
    engine.settings.hold_context = False
    block = engine.block_frame
    output = np.concatenate(
        [
            engine.audio_callback(audio[i : i + block, None].copy(), block, None, None).copy()
            for i in range(0, audio.shape[0] - block + 1, block)
        ]
    )
    lag = engine.splicer.fade + engine.splicer.search
    if denoiser != "none":
        lag += int(engine.input_denoise_delay() * RATE)
    aligned = np.zeros_like(audio)
    usable = min(output.shape[0] - lag, audio.shape[0])
    aligned[:usable] = output[lag : lag + usable]
    return aligned


def main(argv=None):
    import librosa
    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True, help="clean speech WAV")
    parser.add_argument("--snr", type=float, default=15.0, help="speech-to-noise ratio (dB)")
    parser.add_argument("--dir", default=DEFAULT_DIR)
    options = parser.parse_args(argv)

    clean, source_rate = sf.read(options.input, dtype="float32", always_2d=True)
    clean = librosa.to_mono(clean.T)
    if source_rate != RATE:
        clean = librosa.resample(clean, orig_sr=source_rate, target_sr=RATE)
    clean = clean[: clean.shape[0] // 4800 * 4800]
    hop = RATE // 100
    frame_db = 20 * np.log10(
        np.maximum(librosa.feature.rms(y=clean, frame_length=4 * hop, hop_length=hop)[0], 1e-6)
    )
    speech_frames = frame_db > -45
    speech = np.repeat(speech_frames, hop)[: clean.shape[0]]
    speech = np.pad(speech, (0, clean.shape[0] - speech.shape[0]))
    speech_power = np.mean(clean[speech] ** 2)
    noise = make_noise(clean.shape[0], np.random.default_rng(0))
    noise *= np.sqrt(speech_power / 10 ** (options.snr / 10)) / np.std(noise)
    noisy = (clean + noise).astype(np.float32)

    os.makedirs(options.dir, exist_ok=True)
    sf.write(os.path.join(options.dir, "noisy_input.wav"), noisy, RATE)
    settle = RATE  # let RNNoise and the gate settle
    region = np.zeros(clean.shape[0], dtype=bool)
    region[settle:-RATE // 10] = True
    print(f"Noise mixed in at {options.snr:g} dB SNR (fan + hum + clicks)\n")
    print(f"{'method':<16}{'speech SI-SDR (dB)':>20}{'noise left in pauses (dBFS)':>30}")
    for method in ("none", "spectral_gate", "rnnoise"):
        output = run_engine(noisy, method)
        sf.write(os.path.join(options.dir, f"{method}.wav"), output, RATE)
        voiced = region & speech
        pauses = region & ~speech
        quality = si_sdr(output[voiced], clean[voiced])
        residual = 10 * np.log10(np.mean(output[pauses] ** 2) + 1e-20)
        print(f"{method:<16}{quality:>20.2f}{residual:>30.1f}")
    print(f"\nListen: {options.dir}/noisy_input.wav, spectral_gate.wav, rnnoise.wav")
    return 0


if __name__ == "__main__":
    sys.exit(main())
