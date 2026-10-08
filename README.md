# RVC-Realtime-GUI

[English](README.md) | [日本語](README_ja.md)

> ## Latest high-performance real-time build
>
> **This project includes the real-time inference performance updates introduced
> after 2026-07-18.** It combines the updated inference path, CUDA Graph
> warm-up, and input/output noise-reduction fixes in a dedicated desktop client
> for low-latency RVC voice conversion.

RVC-Realtime-GUI is a desktop client for low-latency, real-time RVC
(Retrieval-based Voice Conversion).

> **This fork is a Linux port.** Audio runs through PortAudio on PipeWire:
> **JACK (via pipewire-jack)** is the low-latency route that replaces ASIO, and
> PipeWire's ALSA PCMs (`pipewire`, `default`) are the zero-setup fallback.
> The interface is a native Wayland app built with
> [libcosmic](https://github.com/pop-os/libcosmic), and it drives a separate
> Python inference engine. WASAPI, ASIO, the Windows launchers, and the Tk
> interface have been removed.

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
sudo pacman -S --needed portaudio pipewire-jack ffmpeg rnnoise rustup
rustup default stable
```

On other distributions, install the equivalents: PortAudio built with JACK
support, PipeWire's JACK replacement, FFmpeg, RNNoise (used for input noise
reduction; spectral gating is the fallback), and a Rust toolchain (1.93 or
newer). Building libcosmic also needs the usual Wayland development files
(`libxkbcommon`, `wayland`).

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
./run.sh                              # builds the interface on first run
scripts/install-desktop-entry.sh      # optional: add it to the app launcher
```

The interface starts the engine (`.venv/bin/python -m engine.server`) and
shows its progress while PyTorch and the audio devices load. The engine stops
on its own about 10 seconds after the window closes.

FFmpeg is found on `PATH` (a binary at `tools/ffmpeg/ffmpeg` is used as a
fallback). Set `RVC_UI_LANGUAGE=ja` or `=en` to override the desktop language.

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

Drag a model's `.pth` file onto the window, together with its `.index` file
if it has one. The app copies them into `models/<name>/` (named after the
`.pth`) and selects the new model. Dropping several `.pth` files adds one model
each; an `.index` goes with the model whose name it contains.

You can also create the folders yourself, one per model inside `models`:

```text
models/
  MyVoice/
    MyVoice.pth
    added_MyVoice.index
```

- The **folder name** is displayed as the model name.
- Place the model `.pth` and its `added_*.index` file in the same folder.
- Restart the app or use **Reload** after adding or replacing a model.

## Highlights

- Native COSMIC (libcosmic) interface with Japanese and English UI
- CUDA 12.8 standard build, including current NVIDIA GPU support
- Real-time RVC inference with CUDA Graph warm-up
- JACK (PipeWire) and ALSA audio-device routing with per-channel-pair selection
- Independent input, output, and monitor device selection
- Model-specific voice and level settings
- WAV recording: separate input/output, mix, or split L/R recording
- Audio-file input through FFmpeg
- Runtime-log display and log-file export
- Scriptable engine: control conversion from the command line with `engine.cli`

## Source layout

| Path | Purpose |
| --- | --- |
| `ui/` | COSMIC interface (Rust, libcosmic) |
| `engine/` | Headless real-time engine, socket server, and CLI client |
| `docs/engine-protocol.md` | Protocol between the interface and the engine |
| `infer/` | RVC real-time inference, HuBERT, RMVPE, and FCPE code |
| `tools/` | Audio routing, recording, file input, and helpers |
| `run.sh` | Launcher (builds `ui/` on first run) |
| `configs/config.py` | CUDA device and precision selection |
| `models/README.md` | Model folder layout used by the packaged application |
| `tests/` | Source-level regression tests |

## Not stored in Git

The Git repository intentionally excludes all user-specific and large binary
files:

- The Python virtual environment (`.venv/`), PyTorch, and CUDA libraries
- The Rust build output (`ui/target/`)
- RVC `.pth` models and FAISS `.index` files
- HuBERT and RMVPE weight files
- Audio recordings, logs, and local settings (`configs/engine.json`)

The weights come from the Hugging Face release package, not the source history.

## Development

```sh
.venv/bin/python -m pytest tests                 # engine and helpers
cargo test --manifest-path ui/Cargo.toml         # interface
.venv/bin/python -m engine.server &              # run the engine on its own…
.venv/bin/python -m engine.cli watch             # …and inspect its events
```

**Context hold** (Performance → *Hold context during silence*, on by
default) stops silence from pushing your last words out of the model's
context, so the first word after a pause isn't slurred. In a noisy room, set
*Silence detection* to *Voice detection (experimental)*: RNNoise then decides
when you are speaking, so fan or hum noise doesn't count as speech. To measure
it on your own model and voice:

```sh
.venv/bin/python -m tools.context_bench --model "My Voice" --input speech.wav
```

Chunks are spliced with WSOLA and a fixed 40 ms crossfade
(`tools/splice.py`). To compare it with the previous SOLA splice on your own
model, record real inference chunks once, then replay them through both:

```sh
.venv/bin/python -m tools.splice_bench record --model "My Voice" --input speech.wav
.venv/bin/python -m tools.splice_bench compare   # writes logs/splice_bench/{sola,wsola}.wav
```

The interface tracks libcosmic's `master` branch; `Cargo.lock` pins the exact
commit. Update it with `cargo update --manifest-path ui/Cargo.toml -p libcosmic`.

`README_ja.md` still describes the Windows build and has not been updated for
the Linux port yet.

## Upstream and license

This project is based on RVC-WebUI by RVC-Project. See [NOTICE.md](NOTICE.md)
and [LICENSE](LICENSE) for attribution and license information.
