"""Do breaths and consonants come out as noise rather than forced pitch?

    .venv/bin/python -m tools.consonant_bench --model "My Voice" --input speech.wav

The recording is converted through the real engine offline (context hold off)
with the old F0 handling (every frame voiced), the old handling again (the
noise floor: RVC's synthesiser is stochastic), and the new short-gap filling.
Frames of the input are classified by RMVPE over the whole recording:
voiced, unvoiced-audible (consonants and breaths above -45 dBFS) and silence.

Unvoiced-audible frames should become more noise-like, like the input:
higher spectral flatness and fewer frames where RMVPE finds a pitch in the
output.  Voiced frames must not regress: they should stay voiced, and the
new output should be as close to the old as the old is to itself.
"""

import argparse
import os
import sys
from unittest import mock

import numpy as np

RATE = 48000
HOP = RATE // 100
AUDIBLE_DB = -45.0
DEFAULT_DIR = os.path.join("logs", "consonant_bench")


def rmvpe_f0(rmvpe, audio_48k):
    import librosa

    return rmvpe.infer_from_audio(librosa.resample(audio_48k, orig_sr=RATE, target_sr=16000), thred=0.03)


def frame_flatness(audio):
    import librosa

    return librosa.feature.spectral_flatness(y=audio, n_fft=1024, hop_length=HOP)[0]


def log_mel(audio):
    import librosa

    mel = librosa.feature.melspectrogram(y=audio, sr=RATE, n_fft=2048, hop_length=HOP, n_mels=80)
    return 10 * np.log10(np.maximum(mel, 1e-10))


def main(argv=None):
    import librosa
    import soundfile as sf
    import torch

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--block", type=float, default=0.25)
    parser.add_argument("--max-gaps", default="8", help="comma-separated gap limits to compare")
    parser.add_argument("--dir", default=DEFAULT_DIR)
    options = parser.parse_args(argv)

    from engine.assets import RMVPE_PATH
    from engine.core import RealtimeEngine
    from infer import rtrvc
    from infer.rmvpe import RMVPE

    audio, source_rate = sf.read(options.input, dtype="float32", always_2d=True)
    audio = librosa.to_mono(audio.T)
    if source_rate != RATE:
        audio = librosa.resample(audio, orig_sr=source_rate, target_sr=RATE)

    engine = RealtimeEngine(lambda event, data: None)
    model = engine.models_by_name.get(options.model)
    if model is None:
        raise SystemExit(f"Unknown model {options.model!r}; have: {sorted(engine.models_by_name)}")
    engine.settings.model_name = options.model
    engine.settings.__dict__.update(engine.model_settings_for(options.model))
    engine.settings.block_time = options.block
    engine.settings.hold_context = False
    engine.settings.f0method = "rmvpe"
    engine.require_assets("rmvpe")
    engine.function = "vc"
    engine.load_model(model)

    def convert(full_interpolation, max_gap=None):
        gap = rtrvc.MAX_F0_GAP_FRAMES if max_gap is None else max_gap
        with mock.patch.object(rtrvc, "FULL_F0_INTERPOLATION", full_interpolation), mock.patch.object(
            rtrvc, "MAX_F0_GAP_FRAMES", gap
        ):
            engine.prepare_inference(RATE)
            engine.rvc.cache_pitch.zero_()
            engine.rvc.cache_pitchf.zero_()
            block = engine.block_frame
            output = np.concatenate(
                [
                    engine.audio_callback(audio[i : i + block, None].copy(), block, None, None).copy()
                    for i in range(0, audio.shape[0] - block + 1, block)
                ]
            )
        lag = engine.splicer.fade + engine.splicer.search
        aligned = np.zeros(audio.shape[0], dtype=np.float32)
        usable = min(output.shape[0] - lag, audio.shape[0])
        aligned[:usable] = output[lag : lag + usable]
        return aligned

    gaps = [int(value) for value in options.max_gaps.split(",")]
    runs = {"before": convert(True), "before_rerun": convert(True)}
    for gap in gaps:
        runs[f"after_gap{gap}"] = convert(False, gap)
    os.makedirs(options.dir, exist_ok=True)
    sf.write(os.path.join(options.dir, "input.wav"), audio, RATE)
    for name, output in runs.items():
        sf.write(os.path.join(options.dir, f"{name}.wav"), output, RATE)

    device = engine.config.device
    rmvpe = RMVPE(RMVPE_PATH, is_half=engine.config.is_half, device=device)
    f0_input = rmvpe_f0(rmvpe, audio)
    level = 20 * np.log10(
        np.maximum(librosa.feature.rms(y=audio, frame_length=4 * HOP, hop_length=HOP)[0], 1e-6)
    )
    frames = min(f0_input.shape[0], level.shape[0]) - 2
    usable = np.zeros(frames, dtype=bool)
    usable[RATE // HOP : frames - 10] = True  # skip warm-up and the tail
    voiced = (f0_input[:frames] > 0) & usable
    unvoiced_audible = (f0_input[:frames] == 0) & (level[:frames] > AUDIBLE_DB) & usable

    flat_input = frame_flatness(audio)[:frames]
    results = {}
    for name, output in runs.items():
        f0_out = rmvpe_f0(rmvpe, output)[:frames]
        flat = frame_flatness(output)[:frames]
        results[name] = {
            "flatness": float(np.median(np.log10(flat[unvoiced_audible] + 1e-12))),
            "pitched_%": 100.0 * float(np.mean(f0_out[unvoiced_audible] > 0)),
            "voiced_kept_%": 100.0 * float(np.mean(f0_out[voiced] > 0)),
        }
    mel = {name: log_mel(output)[:, :frames] for name, output in runs.items()}

    def mel_distance(a, b, mask):
        return float(np.mean(np.abs(mel[a][:, mask] - mel[b][:, mask])))

    print(
        f"{int(unvoiced_audible.sum())} consonant/breath frames and {int(voiced.sum())} voiced "
        f"frames (10 ms) in the input\n"
    )
    print("Consonants and breaths (unvoiced-audible frames):")
    input_pitched = 100.0 * float(np.mean(f0_input[:frames][unvoiced_audible] > 0))
    input_flatness = float(np.median(np.log10(flat_input[unvoiced_audible] + 1e-12)))
    print(f"{'run':<16}{'consonants pitched %':>22}{'flatness':>10}{'voiced kept %':>15}{'voiced mel dist':>17}")
    print(f"{'input':<16}{input_pitched:>22.1f}{input_flatness:>10.2f}")
    for name in runs:
        distance = "" if name == "before" else f"{mel_distance(name, 'before', voiced):>17.2f}"
        r = results[name]
        print(f"{name:<16}{r['pitched_%']:>22.1f}{r['flatness']:>10.2f}{r['voiced_kept_%']:>15.1f}{distance}")
    print("(voiced mel dist: dB to 'before'; before_rerun is the noise floor)")
    print(f"\nListen: {options.dir}/input.wav, before.wav, after_gap*.wav")
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
