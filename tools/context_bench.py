"""Measure how context hold affects the first word after a long pause.

    .venv/bin/python -m tools.context_bench --model "My Voice" --input speech.wav

The recording (A) is copied with ``--pause`` seconds of silence inserted at
each of its natural pauses (B).  A pause longer than the context (extra time)
wipes it unless context hold keeps it.  Each run is converted through the real
engine offline, and the first ``--window`` seconds of every word that follows
an inserted pause are compared with the same word in A, whose context was
intact, using log-mel spectrogram distance (dB; lower = closer).  A second
conversion of A gives the noise floor: RVC's synthesiser is stochastic, so
even identical runs differ.  WAVs of every run are written for listening.
"""

import argparse
import os
import sys

import numpy as np

DEFAULT_DIR = os.path.join("logs", "context_bench")
RATE = 48000
SPEECH_DB = -45.0


def find_pauses(audio, min_pause=0.25):
    """Return (start, end) sample ranges of pauses between speech."""
    import librosa

    hop = RATE // 100
    rms = librosa.feature.rms(y=audio, frame_length=4 * hop, hop_length=hop)[0]
    speech = 20 * np.log10(np.maximum(rms, 1e-6)) > SPEECH_DB
    pauses, start = [], None
    for frame, is_speech in enumerate(speech):
        if not is_speech and start is None:
            start = frame
        elif is_speech and start is not None:
            if frame - start >= min_pause * 100 and start > 0:
                pauses.append((start * hop, frame * hop))
            start = None
    return pauses


def insert_pauses(audio, pauses, seconds, block=None, onset_offset=None):
    """Insert silence in the middle of each pause; return (audio, onsets A, onsets B).

    With ``block`` and ``onset_offset`` (samples), each inserted silence is
    padded so the following onset lands ``onset_offset`` before a block
    boundary: the worst case for a silence detector that reports late.
    """
    pieces, previous, onsets_a, onsets_b = [], 0, [], []
    shift = 0
    for start, end in pauses:
        middle = (start + end) // 2
        inserted = int(seconds * RATE)
        if block and onset_offset is not None:
            onset = end + shift + inserted
            inserted += (block - onset_offset - onset) % block
        pieces += [audio[previous:middle], np.zeros(inserted, dtype=audio.dtype)]
        previous = middle
        shift += inserted
        onsets_a.append(end)
        onsets_b.append(end + shift)
    pieces.append(audio[previous:])
    return np.concatenate(pieces), onsets_a, onsets_b


def log_mel(audio):
    import librosa

    mel = librosa.feature.melspectrogram(
        y=audio, sr=RATE, n_fft=2048, hop_length=RATE // 100, n_mels=80
    )
    return 10 * np.log10(np.maximum(mel, 1e-10))


def onset_distance(output, onset, reference, reference_onset, lag, window):
    """Mean |log-mel difference| (dB) of an onset, allowing ±3 frames of jitter."""
    length = int(window * RATE)
    a = log_mel(reference[reference_onset + lag : reference_onset + lag + length])
    best = np.inf
    for shift in range(-3, 4):
        start = onset + lag + shift * (RATE // 100)
        b = log_mel(output[start : start + length])
        frames = min(a.shape[1], b.shape[1])
        best = min(best, float(np.mean(np.abs(a[:, :frames] - b[:, :frames]))))
    return best


def convert(engine, audio, hold_context, pre_roll=True):
    """Run ``audio`` through the engine offline with fresh state."""
    engine.pre_roll = pre_roll
    engine.settings.hold_context = hold_context
    engine.settings.hold_detector = engine.bench_detector
    engine.prepare_inference(RATE)
    engine.rvc.cache_pitch.zero_()
    engine.rvc.cache_pitchf.zero_()
    block = engine.block_frame
    outputs = []
    for start in range(0, audio.shape[0] - block + 1, block):
        chunk = audio[start : start + block, None].copy()
        outputs.append(engine.audio_callback(chunk, block, None, None).copy())
    return np.concatenate(outputs)


def main(argv=None):
    import librosa
    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True, help="speech WAV with pauses")
    parser.add_argument("--pause", type=float, default=3.0, help="inserted silence (s)")
    parser.add_argument("--window", type=float, default=0.5, help="onset length scored (s)")
    parser.add_argument("--block", type=float, default=0.25, help="chunk seconds")
    parser.add_argument("--detector", default="level", choices=("level", "voice"))
    parser.add_argument(
        "--onset-offset-ms",
        type=float,
        help="place every onset this many ms before a block boundary (late-detection worst case)",
    )
    parser.add_argument(
        "--room-noise-db",
        type=float,
        help="add steady fan-like room noise at this level (dBFS) to every run",
    )
    parser.add_argument("--dir", default=DEFAULT_DIR)
    options = parser.parse_args(argv)

    from engine.core import RealtimeEngine

    audio, source_rate = sf.read(options.input, dtype="float32", always_2d=True)
    audio = librosa.to_mono(audio.T)
    if source_rate != RATE:
        audio = librosa.resample(audio, orig_sr=source_rate, target_sr=RATE)
    pauses = find_pauses(audio)
    if not pauses:
        raise SystemExit("No pauses found in the recording.")
    offset = None if options.onset_offset_ms is None else int(options.onset_offset_ms * RATE / 1000)
    with_pauses, onsets_a, onsets_b = insert_pauses(
        audio, pauses, options.pause, int(options.block * RATE), offset
    )
    if options.room_noise_db is not None:
        from tools.denoise_bench import make_noise

        rng = np.random.default_rng(0)
        for name in ("audio", "with_pauses"):
            signal = locals()[name]
            noise = make_noise(signal.shape[0], rng)
            noise *= 10 ** (options.room_noise_db / 20) / np.std(noise)
            if name == "audio":
                audio = (signal + noise).astype(np.float32)
            else:
                with_pauses = (signal + noise).astype(np.float32)

    engine = RealtimeEngine(lambda event, data: None)
    model = engine.models_by_name.get(options.model)
    if model is None:
        raise SystemExit(f"Unknown model {options.model!r}; have: {sorted(engine.models_by_name)}")
    engine.settings.block_time = options.block
    engine.settings.model_name = options.model
    engine.settings.__dict__.update(engine.model_settings_for(options.model))
    engine.require_assets(engine.settings.f0method)
    engine.function = "vc"
    engine.bench_detector = options.detector
    engine.load_model(model)

    runs = {
        "A": convert(engine, audio, hold_context=False),
        "A_rerun": convert(engine, audio, hold_context=False),
        "B_hold_off": convert(engine, with_pauses, hold_context=False),
        "B_hold_on": convert(engine, with_pauses, hold_context=True),
        "B_hold_on_no_preroll": convert(engine, with_pauses, hold_context=True, pre_roll=False),
    }
    lag = engine.splicer.fade + engine.splicer.search
    os.makedirs(options.dir, exist_ok=True)
    for name, output in runs.items():
        sf.write(os.path.join(options.dir, f"{name}.wav"), output, RATE)

    print(f"{len(pauses)} onsets after a {options.pause:g} s pause; "
          f"log-mel distance to A over the first {options.window:g} s (dB, lower = closer)\n")
    print(f"{'onset (s in A)':<16}{'A rerun (floor)':>17}{'hold off':>11}{'hold on':>10}")
    totals = {"A_rerun": [], "B_hold_off": [], "B_hold_on": []}
    for onset_a, onset_b in zip(onsets_a, onsets_b):
        row = {
            "A_rerun": onset_distance(runs["A_rerun"], onset_a, runs["A"], onset_a, lag, options.window),
            "B_hold_off": onset_distance(runs["B_hold_off"], onset_b, runs["A"], onset_a, lag, options.window),
            "B_hold_on": onset_distance(runs["B_hold_on"], onset_b, runs["A"], onset_a, lag, options.window),
        }
        for key, value in row.items():
            totals[key].append(value)
        print(f"{onset_a / RATE:<16.2f}{row['A_rerun']:>17.2f}{row['B_hold_off']:>11.2f}{row['B_hold_on']:>10.2f}")
    print(f"{'mean':<16}{np.mean(totals['A_rerun']):>17.2f}"
          f"{np.mean(totals['B_hold_off']):>11.2f}{np.mean(totals['B_hold_on']):>10.2f}")
    attack = int(0.020 * RATE)

    def attack_db(output, onset, reference_onset):
        level = np.sqrt(np.mean(output[onset + lag : onset + lag + attack] ** 2))
        reference = runs["A"][reference_onset + lag : reference_onset + lag + attack]
        return 20 * np.log10(level / (np.sqrt(np.mean(reference**2)) + 1e-12) + 1e-12)

    print("\nFirst 20 ms of each onset vs A (dB; 0 = attack fully kept):")
    print(f"{'onset (s in A)':<16}{'pre-roll':>10}{'no pre-roll':>13}")
    attacks = []
    for onset_a, onset_b in zip(onsets_a, onsets_b):
        row = (
            attack_db(runs["B_hold_on"], onset_b, onset_a),
            attack_db(runs["B_hold_on_no_preroll"], onset_b, onset_a),
        )
        attacks.append(row)
        print(f"{onset_a / RATE:<16.2f}{row[0]:>10.1f}{row[1]:>13.1f}")
    attacks = np.array(attacks)
    print(f"{'mean':<16}{attacks[:, 0].mean():>10.1f}{attacks[:, 1].mean():>13.1f}")
    print(f"\nListen: {options.dir}/B_hold_off.wav vs {options.dir}/B_hold_on.wav")
    return 0


if __name__ == "__main__":
    sys.exit(main())
