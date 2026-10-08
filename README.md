# RVC-Realtime-GUI

[English](README.md) | [日本語](README_ja.md)

> ## Latest high-performance real-time build
>
> **This project includes the real-time inference performance updates introduced
> after 2026-07-18.** It combines the updated inference path, CUDA Graph
> warm-up, and input/output noise-reduction fixes in a dedicated desktop client
> for low-latency RVC voice conversion.

<img width="1760" height="752" alt="RVC-Realtime-GUI screenshot" src="https://github.com/user-attachments/assets/d001d48b-9f00-4eeb-a90c-1474099e8454" />

RVC-Realtime-GUI is a desktop client for low-latency, real-time RVC
(Retrieval-based Voice Conversion).

> **This fork is a Linux port.** Audio runs through PortAudio on PipeWire:
> **JACK (via pipewire-jack)** is the low-latency route that replaces ASIO, and
> PipeWire's ALSA PCMs (`pipewire`, `default`) are the zero-setup fallback.
> WASAPI, ASIO, and the Windows launchers have been removed.

This repository contains the source for the **CUDA 12.8 standard build**.
It is maintained as a focused derivative of
[RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).

## Performance-focused updates

- **Updated real-time inference implementation** based on upstream changes
  released after 2026-07-18
- **CUDA Graph warm-up** to reduce steady-state GPU inference overhead
- **Input/output noise-reduction fixes** from the updated real-time path
- **CUDA 12.8 standard runtime**, including Blackwell-compatible environments
- **Native JACK period handling**: JACK streams run at the PipeWire quantum
  and are re-chunked to the RVC block size through a frame FIFO

## Linux setup

### 1. System packages (Arch / CachyOS)

```sh
sudo pacman -S --needed portaudio pipewire-jack ffmpeg tk noto-fonts noto-fonts-cjk
```

On other distributions, install the equivalents: PortAudio built with JACK
support, PipeWire's JACK replacement, FFmpeg, Tk, and the Noto fonts.

### 2. Python environment

Python **3.12** is required (`numpy==1.26.4` and `torch==2.7.1` do not support
newer versions). With [uv](https://docs.astral.sh/uv/):

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements_realtime_cuda128.txt \
    --index-strategy unsafe-best-match
```

### 3. Inference assets

The HuBERT and RMVPE weights are not stored in Git. Download the CUDA 12.8
package from the
[Hugging Face download page](https://huggingface.co/niel-blue/RVC-Realtime-GUI)
and copy its `assets/` folder (and any `models/` you want) into this checkout:

```text
assets/
  hubert_base/      # Transformers-format HuBERT/ContentVec (config.json, pytorch_model.bin)
  rmvpe/rmvpe.pt
```

### 4. Run

```sh
./run.sh
```

FFmpeg is found on `PATH` (a binary at `tools/ffmpeg/ffmpeg` is used as a
fallback).

> **First launch may take longer than usual** while the inference assets and
> GPU runtime are initialized.

## Low-latency audio with JACK (PipeWire)

Devices are listed as `[JACK] …` and `[ALSA] …`. JACK devices map one-to-one
to PipeWire nodes; devices with more than two channels are offered per channel
pair (for example `[JACK] My Interface — 3 / 4`).

- The JACK period is the PipeWire quantum. Set it for the app only with
  `PIPEWIRE_QUANTUM=128/48000 ./run.sh`, or for the whole graph with
  `pw-metadata -n settings 0 clock.force-quantum 128`.
- JACK streams follow the PipeWire graph rate (usually 48 kHz); RVC resamples
  internally.
- To send converted audio to Discord, OBS, and so on, route RVC's JACK ports
  to a virtual sink with qpwgraph or Helvum, or select that sink as the output.
- Watch for xruns with `pw-top`.

On machines with several NVIDIA GPUs, **Automatic** selects the GPU with the
most streaming multiprocessors; you can also pick one in the GPU menu or set
`CUDA_VISIBLE_DEVICES`.

## Adding models

Create one folder per model inside `models`:

```text
models/
  MyVoice/
    MyVoice.pth
    added_MyVoice.index
    preview.png
```

- The **folder name** is displayed as the model name.
- Place the model `.pth` and its `added_*.index` file in the same folder.
- An optional `.png`, `.jpg`, or `.jpeg` image is shown automatically as the
  model preview.
- Restart the app or use **Reload** after adding or replacing a model.

## Highlights

- Dedicated CustomTkinter desktop interface with Japanese and English UI
- CUDA 12.8 standard build, including current NVIDIA GPU support
- Real-time RVC inference with CUDA Graph warm-up
- JACK (PipeWire) and ALSA audio-device routing with per-channel-pair selection
- Independent input, output, and monitor device selection
- Model gallery and model-specific general settings
- WAV recording: separate input/output, mix, or split L/R recording
- Audio-file input through FFmpeg
- Runtime-log display and log-file export

## Source layout

| Path | Purpose |
| --- | --- |
| `app/` | Application entry point and GUI |
| `infer/` | RVC real-time inference, HuBERT, RMVPE, and FCPE code |
| `tools/` | GUI adapter, audio routing, recording, file input, and helpers |
| `run.sh` | Linux launcher (uses `.venv/`) |
| `configs/config.py` | CUDA device and precision selection |
| `models/README.md` | Model folder layout used by the packaged application |
| `tests/` | Source-level regression tests |

## Not stored in Git

The Git repository intentionally excludes all user-specific and large binary
files:

- The Python virtual environment (`.venv/`), PyTorch, and CUDA libraries
- RVC `.pth` models and FAISS `.index` files
- HuBERT and RMVPE weight files
- Audio recordings, logs, window position, and local device settings

The weights come from the Hugging Face release package, not the source history.

## Development

```sh
.venv/bin/python -m pytest tests
./run.sh
```

`README_ja.md` still describes the Windows build and has not been updated for
the Linux port yet.

## Upstream and license

This project is based on RVC-WebUI by RVC-Project. See [NOTICE.md](NOTICE.md)
and [LICENSE](LICENSE) for attribution and license information.
