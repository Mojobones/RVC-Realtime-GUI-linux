"""How much past context does RMVPE need for accurate real-time pitch?

    .venv/bin/python -m tools.pitch_bench --input speech.wav [--device cuda:1]

The reference is RMVPE run over the whole recording at once (full past and
future context).  The streaming runs mimic the engine: for every block, RMVPE
sees a window of ``context`` seconds ending at the newest input sample, and the
frames that feed the emitted audio (the block, 50 ms before the window's end)
are compared with the reference.  Reported per context length: gross pitch
errors (> 50 cents off), octave errors (> 600 cents), median error in cents,
voicing mistakes, and the time per RMVPE call.
"""

import argparse
import sys
import time

import numpy as np

RATE = 16000
HOP = 160
#: RMVPE windows are whole multiples of 32 mel frames, minus one hop.
STEP = 5120
OUTPUT_LAG = int(0.050 * RATE)


def window_length(seconds):
    """The RMVPE window the engine would use for at least ``seconds``."""
    samples = max(int(seconds * RATE), 1)
    return STEP * ((samples - 1) // STEP + 1) - HOP


def engine_default_window(block_time):
    """The window today's rtrvc.py formula picks (block + 50 ms, rounded up)."""
    return window_length((int(block_time * RATE) + 800) / RATE)


def cents(estimate, reference):
    return 1200 * np.log2(np.maximum(estimate, 1e-3) / np.maximum(reference, 1e-3))


def evaluate(rmvpe, audio, reference, block_time, window, device):
    import torch

    block = int(block_time * RATE)
    errors, voicing_mistakes, voiced_total, durations = [], 0, 0, []
    for end in range(window, audio.shape[0] + 1, block):
        segment = audio[end - window : end]
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        f0 = rmvpe.infer_from_audio(segment, thred=0.03)
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - started)
        # Frames of the newest block as it reaches the output.
        first = (end - block - OUTPUT_LAG) // HOP
        last = (end - OUTPUT_LAG) // HOP
        local = f0[(first - (end - window) // HOP) : (last - (end - window) // HOP)]
        truth = reference[first:last][: local.shape[0]]
        local = local[: truth.shape[0]]
        voiced = truth > 0
        voiced_total += int(voiced.sum())
        voicing_mistakes += int(np.sum(voiced != (local > 0)))
        both = voiced & (local > 0)
        errors.extend(np.abs(cents(local[both], truth[both])).tolist())
    errors = np.array(errors)
    timed = np.array(durations[3:] or durations)  # skip warm-up calls
    return {
        "gross_%": 100.0 * float(np.mean(errors > 50)) if errors.size else float("nan"),
        "octave_%": 100.0 * float(np.mean(errors > 600)) if errors.size else float("nan"),
        "median_cents": float(np.median(errors)) if errors.size else float("nan"),
        "voicing_%": 100.0 * voicing_mistakes / max(voiced_total, 1),
        "ms_per_call": 1000.0 * float(np.median(timed)),
    }


def main(argv=None):
    import librosa
    import soundfile as sf
    import torch

    from engine.assets import RMVPE_PATH
    from infer.rmvpe import RMVPE

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--half", action="store_true", help="fp16, as the engine uses on RTX GPUs")
    parser.add_argument("--blocks", default="0.1,0.25")
    parser.add_argument("--contexts", default="0.32,0.64,0.96,1.28,1.92,2.56")
    options = parser.parse_args(argv)

    audio, source_rate = sf.read(options.input, dtype="float32", always_2d=True)
    audio = librosa.resample(librosa.to_mono(audio.T), orig_sr=source_rate, target_sr=RATE)
    if options.device.startswith("cuda"):
        # CUDA Graph capture runs on the current device's stream.
        torch.cuda.set_device(options.device)
    rmvpe = RMVPE(RMVPE_PATH, is_half=options.half, device=options.device)
    reference = rmvpe.infer_from_audio(audio, thred=0.03)
    print(
        f"{torch.cuda.get_device_name(options.device) if options.device.startswith('cuda') else 'CPU'}"
        f", {'fp16' if options.half else 'fp32'}; reference voiced frames: {int((reference > 0).sum())}\n"
    )
    print(f"{'block':<7}{'context':>9}{'gross %':>9}{'octave %':>10}{'median ¢':>10}"
          f"{'voicing %':>11}{'ms/call':>9}")
    for block_time in (float(value) for value in options.blocks.split(",")):
        default = engine_default_window(block_time)
        windows = sorted({default, *(window_length(float(c)) for c in options.contexts.split(","))})
        for window in windows:
            if window < int(block_time * RATE) + OUTPUT_LAG:
                continue
            result = evaluate(rmvpe, audio, reference, block_time, window, options.device)
            label = f"{(window + HOP) / RATE:.2f}s" + (" *" if window == default else "")
            print(f"{block_time:<7}{label:>9}{result['gross_%']:>9.2f}{result['octave_%']:>10.2f}"
                  f"{result['median_cents']:>10.1f}{result['voicing_%']:>11.2f}{result['ms_per_call']:>9.1f}")
        print()
    print("* = the window today's engine uses for that block size")
    return 0


if __name__ == "__main__":
    sys.exit(main())
