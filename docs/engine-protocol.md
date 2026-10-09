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
| `delete_model` | `{"name": "Voice"}` | `null`; moves `models/<name>/` to the desktop Trash (`gio trash`, recoverable) and selects another model if it was selected. Refused with `model_in_use` while converting with it |
| `rename_model` | `{"name": "Voice", "new_name": "Voice v2"}` | `null`; renames `models/<name>/` (settings go with it; the selection follows). Names must be unique (ignoring case), not start with `.`, and avoid `/ \ : * ? " < > \|`. Refused with `model_in_use` while converting with it |
| `reload_devices` | | `null` |
| `start` | `{"function": "vc" \| "passthrough"}` (default `vc`) | `null` |
| `stop` | | `null` |
| `file_select` | `{"path": "/abs/file.wav"}` | `null` |
| `file_play` / `file_pause` / `file_stop` | | `null` |
| `file_seek` | `{"seconds": number}` | `null` |
| `record_start` | | `null` |
| `record_stop` | | `{"paths": ["/…/file.wav", …]}` |
| `analyze_source_pitch` | `{"paths": ["/abs/clip.wav", "/abs/dataset-folder", …]}` | `null`; measures the median pitch of the selected model's speaker in the background (see [Pitch match](#pitch-match)) |
| `reset_voice_pitch` | | `null`; forgets the user's measured voice pitch |
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
| `hold_detector` | `"level"` \| `"voice"` | live (default `"level"`): how context hold detects silence. `"voice"` (experimental) uses RNNoise's voice-activity probability, which ignores steady room noise; it needs RNNoise at 48 kHz and falls back to `"level"` otherwise |
| `loudness_target` | `"voice_chat"` \| `"streaming"` | live (default `"voice_chat"`): the output-level target, -21 or -16 LUFS (see [Output level](#output-level)) |
| `rms_mix_rate` | number 0–1 | live (volume envelope) |
| `f0method` | `"pm"` \| `"rmvpe"` \| `"fcpe"` | live |
| `pitch_smoothing` | bool | live (default on): Viterbi pitch tracking for RMVPE, the most likely continuous pitch path through each window instead of each 10 ms frame's own peak (restarts after unvoiced gaps; at most ~2.2 semitones of movement per frame). No added latency, about 3 ms of CPU per block |
| `recording_folder` | string | live |
| `recording_mode` | `"separate"` \| `"mix"` \| `"stereo"` | live (next recording) |
| `file_input_volume` | number 0–1 | live |
| `input_gain_db` / `output_gain_db` / `monitor_gain_db` | number | live, per model |
| `noise_gate_db` | number | live, per model (−60 = off) |
| `pitch` | number (semitones) | live, per model |
| `formant` | number | live, per model |
| `index_rate` | number 0–1 | live, per model |
| `protect` | number 0–0.5 | live, per model (default 0.33; 0.5 = off): on unvoiced frames keep `1 − protect` of the original features when index blending is on (RVC's "protect") |
| `source_pitch_hz` | number (Hz) | per model (0 = not measured): median pitch of the model's speaker, set by `analyze_source_pitch`; kept by `reset_settings` |

Chunks are spliced with WSOLA using a fixed 40 ms crossfade and a 10 ms
search span (see `tools/splice.py`); there is no crossfade setting.

## Pitch match

The front end recommends the pitch that moves the user's median pitch onto
the model speaker's: `12 × log2(source_pitch_hz / voice_pitch.median_hz)`
semitones, rounded to 0.1. Medians depend on what the audio contains, so
the recommendation is a starting point to fine-tune by ear.

- **The model's speaker:** `analyze_source_pitch` runs RMVPE over every
  frame of the given files, and of the audio files directly inside the given
  folders (not subfolders). Use the audio the model was trained on: a
  speaker's pitch can drift by more than a semitone over a long recording.
  The work runs on its own thread and its own RMVPE; while conversion runs
  it pauses between 10 s pieces to leave the GPU to the stream. Progress is
  reported as status `analyzing_pitch`, then `pitch_analyzed` (the result is
  stored with the model that was selected when it started), or an `error`
  event (`no_voice_found`, `pitch_analysis_failed`).
- **The user:** while converting from the microphone, the input pitch of
  every inference is added to a recency-weighted histogram (the last ~5
  minutes of voiced speech count most), saved in `configs/voice_pitch.json`.
  `median_hz` stays `null` until `needed_seconds` of voiced speech is heard.

## Output level

While running, `meters.loudness` reports how loud the output sounds to
others: `{"lufs", "target_lufs", "speech_seconds", "over_percent", "verdict",
"adjust_db"}`.

- `lufs` is speech loudness (ITU-R BS.1770: K-weighted, gated so pauses do
  not count) over the last ~20 s of output, at the current output gain (and
  file volume). The engine measures before the gain, so the reading follows
  the Output gain slider at once. Changing `input_gain_db` starts over.
- `over_percent`: share of speech samples that would exceed full scale and
  be squashed by the soft clipper.
- `verdict`: `null` until 5 s of speech; `"good"` within 3 LU of the target
  (`loudness_target`: -21 LUFS for voice chat, -16 LUFS for streaming, the
  usual level of streamed voice), `"quiet"` below, `"loud"` above
  or whenever `over_percent` exceeds 1 %.
- `adjust_db`: output-gain change (0.5 dB steps) that reaches the target,
  never raising so far that more than 1 % of samples exceed full scale.

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
              "index_file": "added_MyVoice.index",
              "size_bytes": 56223809,       // of the .pth
              "modified": 1791067045}],     // .pth modification time (Unix seconds)
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
  "rnnoise": true,                    // input noise reduction uses RNNoise (else spectral gating)
  "voice_pitch": {"median_hz": 114.2, "seconds": 312.5, "needed_seconds": 20},
  "analyzing_pitch": false            // analyze_source_pitch is running
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
| `meters` | about 20 Hz while running or playing a file: `{"input", "output", "monitor"}` (0–1, −60 dB to 0 dB), `"infer_ms"`, `"recording_seconds"`, `"file_position"`, `"queued_ms"` (audio waiting in the input and output buffers right now; a standing backlog is drained by dropping silent blocks), `"underruns"` (output callbacks that found too little audio queued since the stream started: each is an audible gap; also logged), `"voice_pitch"` (as in State, live while running), `"loudness"` (see [Output level](#output-level)) (each may be `null`) |
| `status` | `{"code": …, …}`: progress and results, listed below |
| `error` | `{"code", "message"}`: a problem not tied to a request (e.g. `monitor_failed`, `file_error`) |
| `log` | `{"text": "…"}`: engine stdout/stderr as it is written |

Status codes: `importing_model`, `model_imported` (`names`), `model_deleted` (`name`), `model_renamed` (`name`, `new_name`), `preparing`, `loading_model`, `preparing_inference`,
`starting_audio`, `conversion_started`, `passthrough_started`,
`conversion_stopped`, `passthrough_stopped`, `settings_changed`,
`stream_stopped` (the audio device went away), `file_selected` (`name`),
`playing`, `paused`, `playback_stopped`, `recording_started`,
`recording_saved` (`paths`), `analyzing_pitch` (`name`, `files`),
`pitch_analyzed` (`name`, `median_hz`, `seconds`).

## Error codes

`bad_request`, `unknown_command`, `not_ready`, `init_failed`,
`internal_error`, `bad_setting`, `bad_device`, `gpu_invalid`, `no_model`,
`model_file_missing`, `index_missing`, `no_audio_file`, `ffmpeg_missing`,
`no_common_samplerate`, `audio_start_failed`, `record_requires_running`,
`recording_failed`, `assets_missing` (details: `paths`), `model_load_failed`, `import_no_model_file`, `import_failed`, `file_error`, `monitor_failed`, `log_save_failed`,
`pitch_analysis_busy`, `pitch_analysis_failed`, `no_voice_found`,
`model_in_use`, `delete_failed`, `bad_model_name` (details: `reason`),
`model_name_taken`, `rename_failed`.
