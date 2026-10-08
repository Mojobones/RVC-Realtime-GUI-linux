"""Compare the previous SOLA splice with WSOLA on identical RVC inference chunks.

Splicing does not change what the model is given, so one offline pass records
every raw inference chunk and both algorithms are replayed on the same data:

    .venv/bin/python -m tools.splice_bench record --model "My Voice" --input speech.wav
    .venv/bin/python -m tools.splice_bench compare

``record`` needs the inference assets and a model; no audio devices are opened.
``compare`` writes ``sola.wav`` and ``wsola.wav`` for listening and prints seam
statistics: confidence (normalised correlation at the chosen offset), the time
cut or repeated per seam, and spectral flux at seams relative to the rest.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from tools.splice import CONFIDENCE_THRESHOLD, WsolaSplicer

DEFAULT_DIR = os.path.join("logs", "splice_bench")
#: Seams whose template is quieter than this (-60 dBFS power) are inaudible
#: and excluded from the confidence statistics.
AUDIBLE_POWER = 1e-6


class SolaSplicer:
    """The splice the engine used before WSOLA (DDSP-SVC SOLA), unchanged."""

    def __init__(self, samplerate, block, device):
        reference = WsolaSplicer(samplerate, block, device)
        self.fade, self.search, self.block = reference.fade, reference.search, block
        self.fade_in, self.fade_out = reference.fade_in, reference.fade_out
        self.ones = reference.ones
        self.template = torch.zeros(self.fade, device=device)
        self.last_offset = 0

    def splice(self, infer_wav):
        window = infer_wav[None, None, : self.fade + self.search]
        numerator = F.conv1d(window, self.template[None, None, :])
        denominator = torch.sqrt(F.conv1d(window**2, self.ones) + 1e-8)
        offset = int(torch.argmax(numerator[0, 0] / denominator[0, 0]))
        self.last_offset = offset
        output = infer_wav[offset:]
        output[: self.fade] *= self.fade_in
        output[: self.fade] += self.template * self.fade_out
        self.template[:] = output[self.block : self.block + self.fade]
        return output[: self.block]


def true_confidence(template, window, offset):
    """Normalised correlation of the template with the candidate at ``offset``."""
    candidate = window[offset : offset + template.shape[0]]
    denominator = torch.sqrt(torch.sum(template**2) * torch.sum(candidate**2)) + 1e-12
    return float(torch.sum(template * candidate) / denominator)


def replay(chunks, samplerate, block, splicer_class):
    """Splice recorded chunks; return (audio, confidences, offsets).

    Confidences and offsets cover audible seams only (template above -60 dBFS).
    """
    splicer = splicer_class(samplerate, block, torch.device("cpu"))
    outputs, confidences, offsets = [], [], []
    for chunk in chunks:
        chunk = torch.as_tensor(chunk, dtype=torch.float32).clone()
        template = splicer.template.clone()
        window = chunk.clone()
        outputs.append(splicer.splice(chunk).clone().numpy())
        if float(torch.mean(template**2)) > AUDIBLE_POWER:
            confidences.append(true_confidence(template, window, splicer.last_offset))
            offsets.append(splicer.last_offset)
    return np.concatenate(outputs), np.array(confidences), np.array(offsets)


def seam_flux_ratio(audio, block, fade, n_fft=1024, hop=256):
    """Mean spectral flux in splice regions divided by the median elsewhere."""
    if audio.shape[0] < n_fft * 2:
        return float("nan")
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop]
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1))
    flux = np.maximum(np.diff(spectrum, axis=0), 0).sum(axis=1)
    centres = (np.arange(flux.shape[0]) + 1) * hop + n_fft // 2
    in_seam = ((centres % block) < fade + n_fft // 2) & (centres > block)
    if not in_seam.any() or in_seam.all():
        return float("nan")
    return float(flux[in_seam].mean() / (np.median(flux[~in_seam]) + 1e-12))


def compare(directory):
    import soundfile as sf

    data = np.load(os.path.join(directory, "chunks.npz"))
    with open(os.path.join(directory, "meta.json"), encoding="utf-8") as meta_file:
        meta = json.load(meta_file)
    rate, block = meta["samplerate"], meta["block"]
    chunks = [data[key] for key in sorted(data.files, key=int)]
    results = {}
    for name, splicer_class in (("sola", SolaSplicer), ("wsola", WsolaSplicer)):
        audio, confidences, offsets = replay(chunks, rate, block, splicer_class)
        sf.write(os.path.join(directory, f"{name}.wav"), audio, rate)
        shifts = np.abs(np.diff(offsets)) * 1000.0 / rate if offsets.size > 1 else np.zeros(1)
        fade = WsolaSplicer(rate, block, torch.device("cpu")).fade
        results[name] = {
            "audible_seams": int(confidences.size),
            "confidence_mean": float(confidences.mean()) if confidences.size else float("nan"),
            "confidence_p5": float(np.percentile(confidences, 5)) if confidences.size else float("nan"),
            "low_confidence_seams": int((confidences < CONFIDENCE_THRESHOLD).sum()),
            "time_shift_ms_mean": float(shifts.mean()),
            "time_shift_ms_max": float(shifts.max()),
            "seam_flux_ratio": seam_flux_ratio(audio, block, fade),
        }
    print(f"{'metric':<24}{'sola':>12}{'wsola':>12}")
    for key in results["sola"]:
        print(f"{key:<24}{results['sola'][key]:>12.3f}{results['wsola'][key]:>12.3f}")
    print(f"\nListen: {directory}/sola.wav and {directory}/wsola.wav")
    return results


def record(model_name, input_path, directory, block_time, f0method):
    import librosa
    import soundfile as sf

    from engine.core import RealtimeEngine

    engine = RealtimeEngine(lambda event, data: None)
    model = engine.models_by_name.get(model_name)
    if model is None:
        raise SystemExit(f"Unknown model {model_name!r}; have: {sorted(engine.models_by_name)}")
    engine.settings.block_time = block_time
    engine.settings.f0method = f0method
    engine.settings.model_name = model_name
    engine.settings.__dict__.update(engine.model_settings_for(model_name))
    engine.require_assets(f0method)
    engine.function = "vc"
    engine.load_model(model)
    rate = 48000
    engine.prepare_inference(rate)

    audio, source_rate = sf.read(input_path, dtype="float32", always_2d=True)
    audio = librosa.to_mono(audio.T)
    if source_rate != rate:
        audio = librosa.resample(audio, orig_sr=source_rate, target_sr=rate)
    chunks = []
    engine.chunk_tap = lambda infer_wav: chunks.append(infer_wav.detach().cpu().clone().numpy())
    block = engine.block_frame
    for start in range(0, audio.shape[0] - block + 1, block):
        engine.audio_callback(audio[start : start + block, None].copy(), block, None, None)
    os.makedirs(directory, exist_ok=True)
    np.savez_compressed(
        os.path.join(directory, "chunks.npz"), **{str(i): c for i, c in enumerate(chunks)}
    )
    with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as meta_file:
        json.dump(
            {"samplerate": rate, "block": block, "model": model_name, "f0method": f0method},
            meta_file,
        )
    print(f"Recorded {len(chunks)} chunks of {model_name} to {directory}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    record_parser = sub.add_parser("record", help="record raw inference chunks")
    record_parser.add_argument("--model", required=True)
    record_parser.add_argument("--input", required=True, help="speech or singing WAV")
    record_parser.add_argument("--block", type=float, default=0.25, help="chunk seconds")
    record_parser.add_argument("--f0", default="rmvpe", choices=("rmvpe", "fcpe", "pm"))
    record_parser.add_argument("--dir", default=DEFAULT_DIR)
    compare_parser = sub.add_parser("compare", help="replay chunks through SOLA and WSOLA")
    compare_parser.add_argument("--dir", default=DEFAULT_DIR)
    options = parser.parse_args(argv)
    if options.command == "record":
        record(options.model, options.input, options.dir, options.block, options.f0)
    else:
        compare(options.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
