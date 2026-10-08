# Engine protocol (version 1)

The real-time engine runs as its own process and is controlled over a Unix
stream socket. The COSMIC front end, the `engine.cli` client, and the tests
all use this protocol.

```sh
.venv/bin/python -m engine.server [--socket PATH] [--exit-when-idle SECONDS]
.venv/bin/python -m engine.cli watch          # print events
.venv/bin/python -m engine.cli state          # current state as JSON
```

- **Socket:** `$XDG_RUNTIME_DIR/rvc-realtime.sock` by default (mode `0600`).
  The server refuses to start if another engine already listens there.
- **Framing:** one UTF-8 JSON object per line (`\n`), in both directions.
- **Startup:** the socket is bound *before* PyTorch and the engine load
  (about 3 s). Clients can connect immediately; engine commands fail with
  `not_ready` until the `ready` event.
- **`--exit-when-idle`:** the server shuts down after no client has been
  connected for that many seconds. A front end that spawns the engine should
  pass this so the engine never outlives it.
- **Ordering:** engine commands run one at a time on a single thread, in the
  order received. A slow command (`start` loads the model and warms up CUDA
  Graphs, which can take seconds) delays later ones; events keep flowing.

## Messages

```jsonc
// client -> engine
{"id": 7, "cmd": "update_settings", "args": {"settings": {"pitch": 12}}}

// engine -> client: response (exactly one per request, matched by "id")
{"id": 7, "ok": true, "result": null}
{"id": 7, "ok": false, "error": {"code": "bad_setting", "message": "pitch must be a number", "details": {"key": "pitch"}}}

// engine -> client: event (any time, interleaved with responses)
{"event": "meters", "data": {...}}
```

`message` is English and meant as a fallback or for logs; front ends should
translate by `code`.

## Commands

| Command | Args | Result |
| --- | --- | --- |
| `ping` | | `{"protocol": 1, "ready": bool}` (works before ready) |
| `get_state` | | [State](#state) |
| `update_settings` | `{"settings": {key: value, ...}}` | `null` |
| `reset_settings` | `{"group": "general" \| "performance"}` | `null` |
| `reload_models` | | `null` |
| `import_model` | `{"paths": ["/abs/Voice.pth", "/abs/added_Voice.index", …]}` | `{"models": ["Voice"]}`; copies each `.pth` (with matching `.index` files) into `models/<name>/` and selects the first |
| `reload_devices` | | `null` |
| `start` | `{"function": "vc" \| "passthrough"}` (default `vc`) | `null` |
| `stop` | | `null` |
| `file_select` | `{"path": "/abs/file.wav"}` | `null` |
| `file_play` / `file_pause` / `file_stop` | | `null` |
| `file_seek` | `{"seconds": number}` | `null` |
| `record_start` | | `null` |
| `record_stop` | | `{"paths": ["/…/file.wav", …]}` |
| `get_log` | | `{"text": "…"}` (last 40,000 characters) |
| `clear_log` | | `null` |
| `save_log` | | `{"path": "logs/RuntimeLog_….txt"}` |
| `shutdown` | | `null`, then the server stops the engine and exits |

`update_settings` validates every key before applying any. Changing a
*restart* setting while running stops the stream (with status
`settings_changed`); the others apply live.

## Settings

| Key | Type | Notes |
| --- | --- | --- |
| `model_name` | string | restart; must be a model name from state |
| `gpu` | string | restart; `"auto"` or a GPU `id` from state |
| `input_source` | `"microphone"` \| `"file"` | restart |
| `input_device` / `output_device` | string | restart; a device `label` from state |
| `monitor_device` | string \| null | restart; `null` disables the monitor |
| `block_time` | number (s) | restart; min 0.02 |
| `extra_time` | number (s) | restart |
| `input_denoise` / `output_denoise` | bool | live; input uses RNNoise at 48 kHz when `librnnoise` is installed, otherwise spectral gating (TorchGate) |
| `hold_context` | bool | live (default on): while every 10 ms frame of a block is below −50 dBFS (or the noise gate, if higher), conversion emits silence and skips inference, so the model's past context survives the pause |
| `rms_mix_rate` | number 0–1 | live (volume envelope) |
| `f0method` | `"pm"` \| `"rmvpe"` \| `"fcpe"` | live |
| `recording_folder` | string | live |
| `recording_mode` | `"separate"` \| `"mix"` \| `"stereo"` | live (next recording) |
| `file_input_volume` | number 0–1 | live |
| `input_gain_db` / `output_gain_db` / `monitor_gain_db` | number | live, per model |
| `noise_gate_db` | number | live, per model (−60 = off) |
| `pitch` | number (semitones) | live, per model |
| `formant` | number | live, per model |
| `index_rate` | number 0–1 | live, per model |

Chunks are spliced with WSOLA using a fixed 40 ms crossfade and a 10 ms
search span (see `tools/splice.py`); there is no crossfade setting.

Per-model settings are saved beside the model (`realtime_settings.json`)
and replaced with that model's values when `model_name` changes. The others
are saved in `configs/engine.json`.

## State

Sent as the `ready` and `state` events and returned by `get_state`.
Front ends can re-render entirely from the latest state.

```jsonc
{
  "running": false,
  "function": "vc",                    // or "passthrough"
  "settings": { /* every key above */ },
  "models": [{"name": "MyVoice", "model_file": "MyVoice.pth",
              "index_file": "added_MyVoice.index"}],
  "devices": {
    "inputs":  [{"label": "[JACK] Elgato Wave XLR Mono", "api": "JACK",
                 "name": "Elgato Wave XLR Mono", "channels": [1, 1]}],
    "outputs": [{"label": "[ALSA] default", "api": "ALSA", "name": "default", "channels": []}]
  },
  "gpus": [{"id": "auto", "label": null}, {"id": "0", "label": "GPU 0 — NVIDIA GeForce RTX 4080"}],
  "samplerate": null,                  // Hz while running
  "delay_ms": null,                    // estimated end-to-end delay while running
  "recording": false,
  "file": {"path": null, "duration": null, "playing": false},
  "ffmpeg": true,
  "missing_assets": [],               // e.g. ["assets/rmvpe/rmvpe.pt"]
  "rnnoise": true                     // input noise reduction uses RNNoise (else spectral gating)
}
```

`index_file` may be `null`. `channels` lists the selected
1-based JACK ports; it is empty for ALSA devices.

## Events

| Event | Data |
| --- | --- |
| `hello` | `{"protocol": 1, "ready": bool, "init_error": string\|null, "state": State\|null}`, sent once on connect |
| `ready` | State; the engine finished starting |
| `fatal` | `{"code": "init_failed", "message": traceback}`; the engine could not start |
| `state` | State, after anything changes |
| `meters` | about 20 Hz while running or playing a file: `{"input", "output", "monitor"}` (0–1, −60 dB to 0 dB), `"infer_ms"`, `"recording_seconds"`, `"file_position"` (each may be `null`) |
| `status` | `{"code": …, …}`: progress and results, listed below |
| `error` | `{"code", "message"}`: a problem not tied to a request (e.g. `monitor_failed`, `file_error`) |
| `log` | `{"text": "…"}`: engine stdout/stderr as it is written |

Status codes: `importing_model`, `model_imported` (`names`), `preparing`, `loading_model`, `preparing_inference`,
`starting_audio`, `conversion_started`, `passthrough_started`,
`conversion_stopped`, `passthrough_stopped`, `settings_changed`,
`stream_stopped` (the audio device went away), `file_selected` (`name`),
`playing`, `paused`, `playback_stopped`, `recording_started`,
`recording_saved` (`paths`).

## Error codes

`bad_request`, `unknown_command`, `not_ready`, `init_failed`,
`internal_error`, `bad_setting`, `bad_device`, `gpu_invalid`, `no_model`,
`model_file_missing`, `index_missing`, `no_audio_file`, `ffmpeg_missing`,
`no_common_samplerate`, `audio_start_failed`, `record_requires_running`,
`recording_failed`, `assets_missing` (details: `paths`), `model_load_failed`, `import_no_model_file`, `import_failed`, `file_error`, `monitor_failed`, `log_save_failed`.
