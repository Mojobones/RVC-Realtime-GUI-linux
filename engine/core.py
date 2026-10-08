"""Headless real-time RVC engine.

This module holds everything the Tk GUI used to do besides drawing widgets:
model and device discovery, settings, stream routing, inference, file input,
and recording.  Front ends talk to it through ``engine.server``.

Threading: every public method is called from one command thread (the
server serializes commands and ``tick``).  PortAudio callbacks, the native
JACK worker, and the file decoder only touch the audio buffers and the
``latest_*`` meter values.
"""

import os
import queue
import shutil
import sys
import threading
import time
import traceback

import librosa
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat

from configs.config import Config, get_device_dtype_sm, infer_device
from engine.devices import DeviceCatalog
from engine.settings import (
    DEFAULT_RECORDING_FOLDER,
    MODEL_SETTING_KEYS,
    PERFORMANCE_DEFAULT_KEYS,
    PROJECT_ROOT,
    RESTART_SETTING_KEYS,
    EngineSettings,
    coerce_setting,
    load_model_settings,
    load_settings,
    save_model_settings,
    save_settings,
)
from engine.version import APP_TITLE, BUILD_LABEL
from infer import rtrvc as rvc_for_realtime
from tools.audio_fifo import AudioFrameFifo, enqueue_latest
from tools.audio_routing import is_native_api, scatter_mono, select_channels
from tools.cuda_graph import cuda_graph_enabled, run_cuda_graph
from tools.file_audio_source import FileAudioSource
from tools.model_import import ModelImportError, import_models
from tools.model_registry import discover_models
from tools.torchgate import TorchGate
from tools.wav_recorder import WavRecorder

MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
MODEL_SETTINGS_SAVE_DELAY_SECONDS = 0.25
METER_INTERVAL_SECONDS = 0.1
FUNCTIONS = {"vc": "vc", "passthrough": "im"}


class EngineError(Exception):
    """An error a front end can show; ``code`` is stable and translatable."""

    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


def peak_meter(samples):
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    peak_db = 20.0 * np.log10(max(peak, 1e-6))
    return float(np.clip((peak_db + 60.0) / 60.0, 0.0, 1.0))


class RealtimeEngine:
    def __init__(self, emit):
        """``emit(event, payload)`` delivers events to connected clients."""
        self.emit = emit
        self.config = Config()
        self.auto_gpu_index = infer_device.index if infer_device.type == "cuda" else None
        printt(APP_TITLE)
        printt(BUILD_LABEL)
        printt("RVC_CUDA_GRAPH=%s", os.environ.get("RVC_CUDA_GRAPH", "0"))
        self.running = False
        self.function = "vc"
        self.delay_time = 0.0
        self.samplerate = None
        self.input_channels = 1
        self.output_channels = 2
        self.input_api = ""
        self.output_api = ""
        self.monitor_api = ""
        self.input_selectors = []
        self.output_selectors = []
        self.monitor_selectors = []
        self.monitor_device_index = None
        self.stream = None
        self.input_stream = None
        self.output_stream = None
        self.monitor_stream = None
        self.output_queue = None
        self.monitor_queue = None
        self.native_input_fifo = None
        self.native_worker_stop = None
        self.native_worker_wakeup = None
        self.native_worker = None
        self.last_input_meter_update = 0.0
        self.last_output_meter_update = 0.0
        self.latest_input_meter = 0.0
        self.latest_output_meter = 0.0
        self.latest_monitor_meter = 0.0
        self.latest_infer_time = 0
        self.recorder = WavRecorder()
        self.file_audio_source = None
        self.file_input_path = ""
        self.file_state = (False, False)
        self.pending_model_settings_name = None
        self.model_settings_save_due = 0.0
        self.ffmpeg_path = shutil.which("ffmpeg") or os.path.join(
            PROJECT_ROOT, "tools", "ffmpeg", "ffmpeg"
        )
        os.makedirs(MODELS_ROOT, exist_ok=True)
        self.devices = DeviceCatalog(sd)
        self.refresh_gpu_options()
        self.refresh_models()
        self.devices.refresh()
        self.settings = load_settings()
        self.normalize_settings()
        self.settings.__dict__.update(self.model_settings_for(self.settings.model_name))
        self.log_startup_diagnostics()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def state(self):
        source = self.file_audio_source
        return {
            "running": self.running,
            "function": "passthrough" if self.function == "im" else "vc",
            "settings": self.settings.to_dict(),
            "models": [
                {
                    "name": model.name,
                    "model_file": model.model_path.name,
                    "index_file": model.index_path.name if model.index_path else None,
                }
                for model in self.models
            ],
            "devices": self.devices.to_dict(),
            "gpus": self.gpu_options,
            "samplerate": self.samplerate if self.running else None,
            "delay_ms": int(np.round(self.delay_time * 1000)) if self.running else None,
            "recording": self.recorder.active,
            "file": {
                "path": self.file_input_path or None,
                "duration": source.duration if source else None,
                "playing": bool(source and source.playing),
            },
            "ffmpeg": os.path.isfile(self.ffmpeg_path),
        }

    def emit_state(self):
        self.emit("state", self.state())

    def status(self, code, **details):
        self.emit("status", {"code": code, **details})

    # ------------------------------------------------------------------
    # Models, GPUs, devices
    # ------------------------------------------------------------------
    def refresh_models(self):
        self.models = discover_models(MODELS_ROOT)
        self.models_by_name = {model.name: model for model in self.models}

    def model_settings_for(self, model_name):
        model = self.models_by_name.get(model_name)
        return load_model_settings(str(model.directory) if model else None)

    def refresh_gpu_options(self):
        self.gpu_options = [{"id": "auto", "label": None}]
        if not torch.cuda.is_available():
            return
        for index in range(torch.cuda.device_count()):
            device, _, _, _ = get_device_dtype_sm(index)
            if device.type != "cuda":
                continue
            self.gpu_options.append(
                {
                    "id": str(index),
                    "label": f"GPU {index} — {torch.cuda.get_device_name(index)}",
                }
            )

    def apply_selected_gpu(self):
        gpu = self.settings.gpu
        index = self.auto_gpu_index if gpu == "auto" else int(gpu)
        if index is None:
            return
        try:
            self.config.select_cuda_device(index)
        except ValueError as error:
            raise EngineError("gpu_invalid", str(error)) from error

    def normalize_settings(self):
        """Point saved model/device/GPU choices at ones that exist now."""
        settings = self.settings
        if settings.model_name not in self.models_by_name:
            settings.model_name = self.models[0].name if self.models else ""
        settings.input_device = self.devices.resolve(settings.input_device, "input")
        settings.output_device = self.devices.resolve(settings.output_device, "output")
        settings.monitor_device = self.devices.resolve_monitor(settings.monitor_device)
        if settings.gpu not in {option["id"] for option in self.gpu_options}:
            settings.gpu = "auto"

    def reload_models(self):
        self.flush_model_settings_save(force=True)
        previous = self.settings.model_name
        self.refresh_models()
        self.normalize_settings()
        if self.settings.model_name != previous and self.running:
            self.stop()
            self.status("settings_changed")
        self.settings.__dict__.update(self.model_settings_for(self.settings.model_name))
        self.emit_state()

    def import_model(self, paths):
        """Copy dropped .pth/.index files into models/ and select the first."""
        if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
            raise EngineError("bad_request", "paths must be a list of strings")
        self.status("importing_model")
        try:
            names = import_models(paths, MODELS_ROOT)
        except ModelImportError as error:
            raise EngineError(error.code, error.message) from error
        printt("Imported model(s): %s", ", ".join(names))
        self.refresh_models()
        self.update_settings({"model_name": names[0]})
        self.status("model_imported", names=names)
        self.emit_state()
        return names

    def reload_devices(self):
        if self.running:
            self.stop()
            self.status("settings_changed")
        self.devices.refresh()
        self.normalize_settings()
        save_settings(self.settings)
        self.emit_state()

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def update_settings(self, changes):
        if not isinstance(changes, dict) or not changes:
            raise EngineError("bad_request", "settings must be a non-empty object")
        coerced = {}
        for key, value in changes.items():
            try:
                coerced[key] = coerce_setting(key, value)
            except ValueError as error:
                raise EngineError("bad_setting", str(error), key=key) from error
        if "model_name" in coerced and coerced["model_name"] not in self.models_by_name:
            raise EngineError("no_model", f"Unknown model: {coerced['model_name']}")
        for key in ("input_device", "output_device"):
            endpoints = self.devices.inputs if key == "input_device" else self.devices.outputs
            if key in coerced and coerced[key] not in endpoints:
                raise EngineError("bad_device", f"Unknown device: {coerced[key]}", key=key)
        if coerced.get("monitor_device") and coerced["monitor_device"] not in self.devices.outputs:
            raise EngineError("bad_device", "Unknown monitor device", key="monitor_device")
        if "gpu" in coerced and coerced["gpu"] not in {o["id"] for o in self.gpu_options}:
            raise EngineError("gpu_invalid", f"Unknown GPU: {coerced['gpu']}")

        changed = {
            key: value
            for key, value in coerced.items()
            if getattr(self.settings, key) != value
        }
        if not changed:
            return
        if "model_name" in changed:
            self.flush_model_settings_save(force=True)
        for key, value in changed.items():
            setattr(self.settings, key, value)
        if "model_name" in changed:
            # Per-model settings follow the newly selected model.
            self.settings.__dict__.update(self.model_settings_for(changed["model_name"]))
        elif any(key in MODEL_SETTING_KEYS for key in changed):
            self.schedule_model_settings_save()
        self.apply_hot_settings(changed)
        if self.running and any(key in RESTART_SETTING_KEYS for key in changed):
            self.stop()
            self.status("settings_changed")
        if any(key not in MODEL_SETTING_KEYS for key in changed):
            save_settings(self.settings)
        self.emit_state()

    def apply_hot_settings(self, changed):
        if not hasattr(self, "rvc"):
            return
        if "pitch" in changed:
            self.rvc.change_key(self.settings.pitch)
        if "formant" in changed:
            self.rvc.change_formant(self.settings.formant)
        if "index_rate" in changed:
            self.rvc.change_index_rate(self.settings.index_rate)
        if "input_denoise" in changed and self.running:
            self.delay_time += (1 if self.settings.input_denoise else -1) * min(
                self.settings.crossfade_time, 0.04
            )

    def reset_settings(self, group):
        defaults = EngineSettings()
        if group == "general":
            keys = MODEL_SETTING_KEYS
        elif group == "performance":
            keys = PERFORMANCE_DEFAULT_KEYS
        else:
            raise EngineError("bad_request", "group must be general or performance")
        self.update_settings({key: getattr(defaults, key) for key in keys})
        if group == "general":
            self.flush_model_settings_save(force=True)

    def schedule_model_settings_save(self):
        if self.settings.model_name not in self.models_by_name:
            return
        self.pending_model_settings_name = self.settings.model_name
        self.model_settings_save_due = time.monotonic() + MODEL_SETTINGS_SAVE_DELAY_SECONDS

    def flush_model_settings_save(self, force=False):
        model_name = self.pending_model_settings_name
        if not model_name:
            return
        if not force and time.monotonic() < self.model_settings_save_due:
            return
        model = self.models_by_name.get(model_name)
        if model is not None:
            try:
                save_model_settings(str(model.directory), self.settings)
            except OSError as error:
                printt("Could not save model settings: %s", error)
        self.pending_model_settings_name = None
        self.model_settings_save_due = 0.0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def log_startup_diagnostics(self):
        """Log release-relevant checks without opening any audio stream."""
        printt("=== Startup diagnostics ===")
        printt("FFmpeg: %s (found=%s)", self.ffmpeg_path, os.path.isfile(self.ffmpeg_path))
        printt("Models discovered: %s", len(self.models))
        printt("CUDA available: %s", torch.cuda.is_available())
        if torch.cuda.is_available():
            printt("CUDA devices: %s", torch.cuda.device_count())
            printt("Selected GPU: %s", self.settings.gpu)
            printt("CUDA Graph enabled: %s", cuda_graph_enabled(self.config.device))
        try:
            os.makedirs(self.settings.recording_folder, exist_ok=True)
            printt(
                "Recording folder writable: %s",
                os.access(self.settings.recording_folder, os.W_OK),
            )
        except OSError as error:
            printt("Recording folder unavailable: %s", error)
        printt(
            "Configured route: input=%s / output=%s / monitor=%s",
            self.settings.input_device,
            self.settings.output_device,
            self.settings.monitor_device or "Disabled",
        )
        printt("=== Startup diagnostics complete ===")

    def log_active_audio_route(self):
        printt("=== Active audio route ===")
        printt(
            "Input=%s / Output=%s / Monitor=%s",
            self.input_api or "File",
            self.output_api,
            self.monitor_api or "Disabled",
        )
        printt(
            "Rate=%s Hz / Chunk=%.3f sec / PIPEWIRE_QUANTUM=%s",
            self.samplerate,
            self.settings.block_time,
            os.environ.get("PIPEWIRE_QUANTUM", "(server default)"),
        )
        printt(
            "Channel selectors: input=%s output=%s monitor=%s",
            self.input_selectors,
            self.output_selectors,
            self.monitor_selectors,
        )
        printt("=== Active audio route complete ===")

    # ------------------------------------------------------------------
    # Periodic work
    # ------------------------------------------------------------------
    def tick(self):
        """Called about 20 times per second from the command thread."""
        self.flush_model_settings_save()
        if self.running and not self.streams_active():
            printt("Audio stream stopped unexpectedly; returning to the idle state.")
            self.stop()
            self.status("stream_stopped")
        source = self.file_audio_source
        if source is not None:
            if source.error:
                error, source.error = source.error, None
                self.emit("error", {"code": "file_error", "message": str(error)})
            file_state = (bool(source.playing and not source.finished), source.finished)
            if file_state != self.file_state:
                self.file_state = file_state
                self.emit_state()
        if self.running or (source is not None and source.playing):
            self.emit("meters", self.meters())

    def streams_active(self):
        def active(stream):
            return stream is not None and bool(getattr(stream, "active", False))

        if self.file_source_active:
            # File input owns no PortAudio input stream: decoded blocks are
            # fed into audio_callback by FileAudioSource.
            return active(self.output_stream)
        return active(self.input_stream) and active(self.output_stream)

    def meters(self):
        recording_seconds = None
        if self.recorder.active and self.recorder.started_at is not None:
            recording_seconds = (
                time.time() - self.recorder.started_at.timestamp()
            )
        source = self.file_audio_source
        return {
            "input": self.latest_input_meter if self.running else 0.0,
            "output": self.latest_output_meter if self.running else 0.0,
            "monitor": self.latest_monitor_meter if self.running else 0.0,
            "infer_ms": self.latest_infer_time if self.running else None,
            "recording_seconds": recording_seconds,
            "file_position": source.position if source else None,
        }

    def shutdown(self):
        self.flush_model_settings_save(force=True)
        self.stop()

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------
    @property
    def file_source_active(self):
        return self.settings.input_source == "file"

    def start(self, function="vc"):
        if function not in FUNCTIONS:
            raise EngineError("bad_request", "function must be vc or passthrough")
        if self.running:
            self.stop()
        self.function = FUNCTIONS[function]
        model = self.validate_start()
        self.status("preparing")
        try:
            self.start_vc(model)
        except EngineError:
            self.stop()
            raise
        except Exception as error:
            printt(traceback.format_exc())
            self.stop()
            raise EngineError("audio_start_failed", str(error)) from error
        if self.stream is not None:
            stream_latency = self.stream.latency
            if isinstance(stream_latency, (tuple, list)):
                stream_latency = stream_latency[-1]
            self.delay_time = (
                stream_latency
                + self.settings.block_time
                + self.settings.crossfade_time
                + 0.01
            )
        if self.settings.input_denoise:
            self.delay_time += min(self.settings.crossfade_time, 0.04)
        save_settings(self.settings)
        self.status(
            "passthrough_started" if self.function == "im" else "conversion_started"
        )
        self.emit_state()

    def validate_start(self):
        model = self.models_by_name.get(self.settings.model_name)
        if model is None:
            raise EngineError("no_model", "Select a model from the models folder.")
        if not model.model_path.is_file():
            raise EngineError(
                "model_file_missing",
                f"Model file not found: {model.model_path}",
                path=str(model.model_path),
            )
        if self.settings.index_rate > 0 and model.index_path is None:
            raise EngineError(
                "index_missing",
                "The selected model has no .index file. "
                "Set Index to 0 or add an .index file to its model folder.",
            )
        self.apply_selected_gpu()
        if self.file_source_active:
            if not self.file_input_path:
                raise EngineError("no_audio_file", "Select an audio file first.")
            if not os.path.isfile(self.ffmpeg_path):
                raise EngineError("ffmpeg_missing", "FFmpeg was not found.")
        self.set_devices()
        return model

    def set_devices(self):
        settings = self.settings
        output = self.devices.outputs.get(settings.output_device)
        if output is None:
            raise EngineError("bad_device", "Select an output device.", key="output_device")
        if self.file_source_active:
            # A decoded file has no PortAudio input endpoint.  Binding the
            # unused input side to the output endpoint prevents a stale or
            # disabled microphone from vetoing conversion startup.
            input_index = output.index
            self.input_api = "File"
            self.input_selectors = []
        else:
            endpoint = self.devices.inputs.get(settings.input_device)
            if endpoint is None:
                raise EngineError("bad_device", "Select an input device.", key="input_device")
            input_index = endpoint.index
            self.input_api = endpoint.api
            self.input_selectors = endpoint.selectors
        sd.default.device = (input_index, output.index)
        self.output_api = output.api
        self.output_selectors = output.selectors
        monitor = self.devices.outputs.get(settings.monitor_device or "")
        self.monitor_device_index = monitor.index if monitor else None
        self.monitor_selectors = monitor.selectors if monitor else []
        if (
            self.monitor_device_index == output.index
            and self.monitor_selectors == self.output_selectors
        ):
            printt(
                "Monitor output is the same as the main output; "
                "the duplicate stream was omitted."
            )
            self.monitor_device_index = None
            self.monitor_selectors = []
        self.monitor_api = monitor.api if self.monitor_device_index is not None else ""
        printt(
            "Audio route: Input [%s] %s / Output [%s] %s / Monitor [%s] %s",
            self.input_api,
            settings.input_device,
            self.output_api,
            settings.output_device,
            self.monitor_api or "Disabled",
            settings.monitor_device or "Disabled",
        )

    def start_vc(self, model):
        self.status("loading_model")
        torch.cuda.empty_cache()
        self.rvc = rvc_for_realtime.RVC(
            self.settings.pitch,
            self.settings.formant,
            str(model.model_path),
            str(model.index_path) if model.index_path is not None else "",
            self.settings.index_rate,
            self.config,
            self.rvc if hasattr(self, "rvc") else None,
        )
        # Decoded files are always supplied as mono float32 blocks.  Do not
        # inherit a JACK input endpoint's channel layout for file input.
        self.input_channels = 1 if self.file_source_active else self.get_device_channels()
        self.output_channels = self.get_output_channels()
        self.samplerate = self.get_automatic_samplerate(self.rvc.tgt_sr)
        self.zc = self.samplerate // 100
        self.block_frame = (
            int(np.round(self.settings.block_time * self.samplerate / self.zc)) * self.zc
        )
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = (
            int(np.round(self.settings.crossfade_time * self.samplerate / self.zc))
            * self.zc
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = (
            int(np.round(self.settings.extra_time * self.samplerate / self.zc)) * self.zc
        )
        self.input_wav = torch.zeros(
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc,
            device=self.config.device,
            dtype=torch.float32,
        )
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
        )
        self.sola_den_kernel = torch.ones(
            1, 1, self.sola_buffer_frame, device=self.config.device, dtype=torch.float32
        )
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.fade_in_window = (
            torch.sin(
                0.5
                * np.pi
                * torch.linspace(
                    0.0,
                    1.0,
                    steps=self.sola_buffer_frame,
                    device=self.config.device,
                    dtype=torch.float32,
                )
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        self.resampler = tat.Resample(
            orig_freq=self.samplerate, new_freq=16000, dtype=torch.float32
        ).to(self.config.device)
        if self.rvc.tgt_sr != self.samplerate:
            self.resampler2 = tat.Resample(
                orig_freq=self.rvc.tgt_sr, new_freq=self.samplerate, dtype=torch.float32
            ).to(self.config.device)
        else:
            self.resampler2 = None
        self.tg = TorchGate(
            sr=self.samplerate, n_fft=4 * self.zc, prop_decrease=0.9
        ).to(self.config.device)
        self.status("preparing_inference")
        self.prewarm_cuda_graph()
        self.status("starting_audio")
        if self.output_stream is not None:
            self.stop()
        self.start_stream()

    def prewarm_cuda_graph(self):
        if not cuda_graph_enabled(self.config.device):
            return
        try:
            samples = self.input_wav_res.shape[0]
            phase = torch.arange(samples, device=self.config.device, dtype=torch.float32)
            probe = 0.05 * torch.sin(2 * np.pi * 220.0 * phase / 16000.0)
            self.input_wav_res.copy_(probe)

            if self.settings.input_denoise:
                short = self.input_wav[-self.sola_buffer_frame - self.block_frame :].unsqueeze(0)
                self.tg(short, self.input_wav.unsqueeze(0))

            resample_input = self.input_wav[-self.block_frame - 2 * self.zc :]
            run_cuda_graph(
                self.resampler,
                "realtime-input-resample",
                lambda audio: self.resampler(audio),
                resample_input,
            )

            inferred = self.rvc.infer(
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
                self.settings.f0method,
            )
            if self.resampler2 is not None:
                inferred = run_cuda_graph(
                    self.resampler2,
                    "realtime-output-resample",
                    lambda audio: self.resampler2(audio),
                    inferred,
                )
            if self.settings.output_denoise:
                self.tg(inferred.unsqueeze(0), self.output_buffer.unsqueeze(0))
            torch.cuda.synchronize(self.config.device)
        except Exception:
            printt(traceback.format_exc())
        finally:
            self.input_wav.zero_()
            self.input_wav_denoise.zero_()
            self.input_wav_res.zero_()
            self.output_buffer.zero_()
            self.sola_buffer.zero_()
            self.nr_buffer.zero_()
            self.rvc.cache_pitch.zero_()
            self.rvc.cache_pitchf.zero_()

    def start_stream(self):
        if self.running:
            return
        self.running = True
        self.log_active_audio_route()
        if self.file_source_active:
            self.start_file_stream()
            return
        if is_native_api(self.input_api):
            self.start_native_input_route()
            return
        self.start_output_stream()
        self.input_stream = sd.InputStream(
            callback=self.audio_callback,
            blocksize=self.block_frame,
            samplerate=self.samplerate,
            channels=self.input_channels,
            device=sd.default.device[0],
            dtype="float32",
        )
        self.input_stream.start()
        self.stream = self.output_stream
        self.start_monitor_stream()

    def start_file_stream(self):
        """Start only the output side; file decoding feeds audio_callback."""
        self.start_output_stream()
        self.stream = self.output_stream
        self.start_monitor_stream()

    def start_output_stream(self, native=None):
        """Open the main output.

        JACK owns its period size, so a JACK output (or any output fed by the
        native input worker) reads from a frame FIFO with blocksize=0.  ALSA
        outputs keep the fixed RVC block queue.
        """
        if native is None:
            native = is_native_api(self.output_api)
        self.output_queue = (
            AudioFrameFifo(1, max_frames=self.block_frame * 4)
            if native
            else queue.Queue(maxsize=3)
        )
        self.output_stream = sd.OutputStream(
            callback=self.output_audio_callback,
            blocksize=0 if native else self.block_frame,
            samplerate=self.samplerate,
            channels=self.output_channels,
            device=sd.default.device[1],
            dtype="float32",
        )
        self.output_stream.start()

    def start_native_input_route(self):
        """Keep JACK's native period size separate from the RVC chunk."""
        # These are capacity limits, not an added latency preset.  JACK
        # callbacks feed their native 64/128/256-frame periods into the input
        # FIFO; the worker consumes exact RVC chunks.
        fifo_capacity = self.block_frame * 4
        self.native_input_fifo = AudioFrameFifo(
            len(self.input_selectors), max_frames=fifo_capacity
        )
        self.native_worker_stop = threading.Event()
        self.native_worker_wakeup = threading.Event()
        self.native_worker = threading.Thread(
            target=self.native_inference_worker,
            name="rvc-native-inference",
            daemon=True,
        )
        self.start_output_stream(native=True)
        self.start_monitor_stream()
        self.input_stream = sd.InputStream(
            callback=self.native_fifo_input_callback,
            blocksize=0,
            samplerate=self.samplerate,
            channels=self.input_channels,
            device=sd.default.device[0],
            dtype="float32",
        )
        self.native_worker.start()
        self.input_stream.start()
        self.stream = self.input_stream

    def native_inference_worker(self):
        while not self.native_worker_stop.is_set():
            self.native_worker_wakeup.wait(0.05)
            self.native_worker_wakeup.clear()
            while not self.native_worker_stop.is_set():
                input_block = self.native_input_fifo.read(self.block_frame, exact=True)
                if input_block is None:
                    break
                try:
                    self.audio_callback(input_block, self.block_frame, None, None)
                except Exception:
                    printt(traceback.format_exc())
                    self.native_worker_stop.set()
                    return

    def native_fifo_input_callback(self, indata, frames, times, status):
        self.native_input_fifo.write(select_channels(indata, self.input_selectors))
        self.native_worker_wakeup.set()

    def output_audio_callback(self, outdata, frames, times, status):
        block = self.read_audio_target(self.output_queue, frames)
        self.write_output_block(outdata, block)
        # Use the real device callback timeline for recording.  The output
        # can be delayed by RVC's buffers, which is intentionally preserved.
        if block is None:
            self.recorder.enqueue("output", np.zeros(frames, dtype=np.float32))
        else:
            self.recorder.enqueue("output", block[:, 0])

    @staticmethod
    def read_audio_target(target, frames):
        if target is None:
            return None
        if isinstance(target, AudioFrameFifo):
            return target.read(frames)
        try:
            block = target.get_nowait()
        except queue.Empty:
            return None
        if block.ndim == 1:
            block = block[:, None]
        return block

    def write_output_block(self, outdata, block):
        outdata.fill(0)
        if block is None:
            return
        sample_count = min(outdata.shape[0], block.shape[0])
        mono = block[:sample_count, 0] if block.ndim == 2 else block[:sample_count]
        scatter_mono(outdata, mono, self.output_selectors)

    def start_monitor_stream(self):
        self.monitor_stream = None
        if self.monitor_device_index is None:
            self.monitor_queue = None
            return
        try:
            self.monitor_channels = self.stream_channels(
                self.monitor_device_index, "output", self.monitor_selectors
            )
            sd.check_output_settings(
                device=self.monitor_device_index,
                channels=self.monitor_channels,
                dtype="float32",
                samplerate=self.samplerate,
            )
            self.monitor_queue = AudioFrameFifo(1, max_frames=self.block_frame * 4)
            self.monitor_stream = sd.OutputStream(
                device=self.monitor_device_index,
                callback=self.monitor_audio_callback,
                blocksize=0,
                samplerate=self.samplerate,
                channels=self.monitor_channels,
                dtype="float32",
            )
            self.monitor_stream.start()
        except Exception as error:
            if self.monitor_stream is not None:
                self.monitor_stream.close()
            self.monitor_stream = None
            self.monitor_queue = None
            # Not fatal: conversion continues on the main output only.
            self.emit("error", {"code": "monitor_failed", "message": str(error)})

    def monitor_audio_callback(self, outdata, frames, times, status):
        outdata.fill(0)
        block = self.read_audio_target(self.monitor_queue, frames)
        if block is None:
            return
        sample_count = min(frames, block.shape[0])
        mono = block[:sample_count, 0] if block.ndim == 2 else block[:sample_count]
        scatter_mono(outdata, mono, self.monitor_selectors)

    def stop(self):
        was_running = self.running
        self.running = False
        self.stop_file_playback()
        if self.recorder.active:
            self.stop_recording()
        if self.native_worker_stop is not None:
            self.native_worker_stop.set()
        if self.native_worker_wakeup is not None:
            self.native_worker_wakeup.set()
        for stream_name in ("input_stream", "output_stream", "monitor_stream"):
            stream = getattr(self, stream_name, None)
            if stream is not None:
                stream.abort()
                stream.close()
                setattr(self, stream_name, None)
        self.stream = None
        self.output_queue = None
        self.monitor_queue = None
        self.native_input_fifo = None
        if self.native_worker is not None and self.native_worker.is_alive():
            self.native_worker.join(timeout=0.5)
        self.native_worker = None
        self.native_worker_stop = None
        self.native_worker_wakeup = None
        self.latest_input_meter = 0.0
        self.latest_output_meter = 0.0
        self.latest_monitor_meter = 0.0
        if was_running:
            self.status(
                "passthrough_stopped" if self.function == "im" else "conversion_stopped"
            )
            self.emit_state()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def audio_callback(self, indata, frames, times, status):
        """Process one input-audio callback block."""
        start_time = time.perf_counter()
        settings = self.settings
        indata = librosa.to_mono(indata.T)
        indata *= np.float32(settings.input_gain)
        np.clip(indata, -1.0, 1.0, out=indata)
        self.recorder.enqueue("input", indata)
        meter_now = time.perf_counter()
        if meter_now - self.last_input_meter_update >= METER_INTERVAL_SECONDS:
            self.latest_input_meter = peak_meter(indata)
            self.last_input_meter_update = meter_now
        if settings.noise_gate_db > -60:
            indata = np.append(self.rms_buffer, indata)
            rms = librosa.feature.rms(
                y=indata, frame_length=4 * self.zc, hop_length=self.zc
            )[:, 2:]
            self.rms_buffer[:] = indata[-4 * self.zc :]
            indata = indata[2 * self.zc - self.zc // 2 :]
            db_threhold = (
                librosa.amplitude_to_db(rms, ref=1.0)[0] < settings.noise_gate_db
            )
            for i in range(db_threhold.shape[0]):
                if db_threhold[i]:
                    indata[i * self.zc : (i + 1) * self.zc] = 0
            indata = indata[self.zc // 2 :]
        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(self.config.device)
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[
            self.block_frame_16k :
        ].clone()
        # input noise reduction and resampling
        if settings.input_denoise:
            self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[
                self.block_frame :
            ].clone()
            input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame :]
            input_wav = self.tg(
                input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)
            ).squeeze(0)
            input_wav[: self.sola_buffer_frame] *= self.fade_in_window
            input_wav[: self.sola_buffer_frame] += self.nr_buffer * self.fade_out_window
            self.input_wav_denoise[-self.block_frame :] = input_wav[: self.block_frame]
            self.nr_buffer[:] = input_wav[self.block_frame :]
            resample_input = self.input_wav_denoise[-self.block_frame - 2 * self.zc :]
            self.input_wav_res[-self.block_frame_16k - 160 :] = run_cuda_graph(
                self.resampler,
                "realtime-input-resample",
                lambda audio: self.resampler(audio),
                resample_input,
            )[160:]
        else:
            resample_input = self.input_wav[-indata.shape[0] - 2 * self.zc :]
            self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = run_cuda_graph(
                self.resampler,
                "realtime-input-resample",
                lambda audio: self.resampler(audio),
                resample_input,
            )[160:]
        # infer
        if self.function == "vc":
            infer_wav = self.rvc.infer(
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
                settings.f0method,
            )
            if self.resampler2 is not None:
                infer_wav = run_cuda_graph(
                    self.resampler2,
                    "realtime-output-resample",
                    lambda audio: self.resampler2(audio),
                    infer_wav,
                )
        elif settings.input_denoise:
            infer_wav = self.input_wav_denoise[self.extra_frame :].clone()
        else:
            infer_wav = self.input_wav[self.extra_frame :].clone()
        # output noise reduction
        if settings.output_denoise and self.function == "vc":
            self.output_buffer[: -self.block_frame] = self.output_buffer[
                self.block_frame :
            ].clone()
            self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
            infer_wav = self.tg(
                infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)
            ).squeeze(0)
        # volume envelop mixing
        if settings.rms_mix_rate < 1 and self.function == "vc":
            if settings.input_denoise:
                input_wav = self.input_wav_denoise[self.extra_frame :]
            else:
                input_wav = self.input_wav[self.extra_frame :]
            rms1 = librosa.feature.rms(
                y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms1 = torch.from_numpy(rms1).to(self.config.device)
            rms1 = F.interpolate(
                rms1.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = librosa.feature.rms(
                y=infer_wav[:].cpu().numpy(),
                frame_length=4 * self.zc,
                hop_length=self.zc,
            )
            rms2 = torch.from_numpy(rms2).to(self.config.device)
            rms2 = F.interpolate(
                rms2.unsqueeze(0),
                size=infer_wav.shape[0] + 1,
                mode="linear",
                align_corners=True,
            )[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(rms1 / rms2, 1.0 - settings.rms_mix_rate)
        # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
        conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input**2, self.sola_den_kernel) + 1e-8)
        if sys.platform == "darwin":
            _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])
            sola_offset = sola_offset.item()
        else:
            sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
        infer_wav = infer_wav[sola_offset:]
        infer_wav[: self.sola_buffer_frame] *= self.fade_in_window
        infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = infer_wav[
            self.block_frame : self.block_frame + self.sola_buffer_frame
        ]
        output_block = torch.clamp(
            infer_wav[: self.block_frame] * settings.output_gain, -1.0, 1.0
        )
        output_mono = output_block.cpu().numpy()
        # RVC can restore the source loudness internally (for example through
        # volume-envelope mixing).  Apply the file-player volume after
        # inference so each 1% step controls the audible result.
        if self.file_source_active:
            output_mono *= np.float32(np.clip(settings.file_input_volume, 0.0, 1.0))
        monitor_mono = np.clip(
            output_mono * np.float32(settings.monitor_gain), -1.0, 1.0
        )
        meter_now = time.perf_counter()
        if meter_now - self.last_output_meter_update >= METER_INTERVAL_SECONDS:
            self.latest_output_meter = peak_meter(output_mono)
            self.latest_monitor_meter = (
                0.0 if self.monitor_device_index is None else peak_meter(monitor_mono)
            )
            self.last_output_meter_update = meter_now
        self.enqueue_audio_target(self.output_queue, output_mono)
        self.enqueue_audio_target(self.monitor_queue, monitor_mono)
        if self.running:
            self.latest_infer_time = int((time.perf_counter() - start_time) * 1000)
        return output_mono

    @staticmethod
    def enqueue_audio_target(target, block):
        if target is None:
            return
        if isinstance(target, AudioFrameFifo):
            target.write(block)
        else:
            enqueue_latest(target, block.copy())

    # ------------------------------------------------------------------
    # Routing helpers
    # ------------------------------------------------------------------
    def get_automatic_samplerate(self, model_rate):
        """Use the active audio route's clock; RVC resamples as needed.

        JACK endpoints run at the PipeWire graph rate (usually 48 kHz), so
        the devices' own rates are tried first and the model rate is only a
        final fallback.
        """
        return self.get_routing_samplerate(preferred_rates=[model_rate])

    def get_routing_samplerate(self, preferred_rates=()):
        """Find a sample rate that every active stream can actually open."""
        input_info = sd.query_devices(device=sd.default.device[0])
        output_info = sd.query_devices(device=sd.default.device[1])

        candidates = []
        for rate in (
            *(() if self.file_source_active else (input_info["default_samplerate"],)),
            output_info["default_samplerate"],
            48000,
            44100,
            *preferred_rates,
            40000,
        ):
            rate = int(round(rate))
            if rate > 0 and rate not in candidates:
                candidates.append(rate)

        checks = [(sd.check_output_settings, sd.default.device[1], self.output_channels)]
        if not self.file_source_active:
            checks.append((sd.check_input_settings, sd.default.device[0], self.input_channels))
        if self.monitor_device_index is not None:
            checks.append(
                (
                    sd.check_output_settings,
                    self.monitor_device_index,
                    self.stream_channels(
                        self.monitor_device_index, "output", self.monitor_selectors
                    ),
                )
            )

        errors = []
        for rate in candidates:
            try:
                for checker, device, channels in checks:
                    checker(device=device, channels=channels, dtype="float32", samplerate=rate)
            except sd.PortAudioError as error:
                errors.append(f"{rate} Hz: {error}")
                continue
            printt("Selected common sample rate: %s", rate)
            return rate

        raise EngineError(
            "no_common_samplerate",
            "The selected input, output, and monitor devices have no common sample rate. "
            "Set the devices to the same rate (usually 48 kHz or 44.1 kHz).\n"
            + "\n".join(errors[-3:]),
        )

    @staticmethod
    def stream_channels(device, direction, selectors):
        """Channels to open: enough to reach every selected JACK port."""
        if selectors:
            return max(selectors) + 1
        max_key = "max_input_channels" if direction == "input" else "max_output_channels"
        return min(int(sd.query_devices(device=device)[max_key]), 2)

    def get_device_channels(self):
        return self.stream_channels(sd.default.device[0], "input", self.input_selectors)

    def get_output_channels(self):
        return self.stream_channels(sd.default.device[1], "output", self.output_selectors)

    # ------------------------------------------------------------------
    # File input
    # ------------------------------------------------------------------
    def file_select(self, path):
        if not isinstance(path, str) or not os.path.isfile(path):
            raise EngineError("no_audio_file", f"Audio file not found: {path}")
        if not os.path.isfile(self.ffmpeg_path):
            raise EngineError(
                "ffmpeg_missing",
                "FFmpeg was not found. Install ffmpeg and make sure it is on PATH.",
            )
        self.stop_file_playback()
        self.file_input_path = path
        self.status("file_selected", name=os.path.basename(path))
        self.emit_state()

    def file_play(self):
        if not self.file_source_active or not self.file_input_path:
            raise EngineError("no_audio_file", "Select an audio file first.")
        if not os.path.isfile(self.ffmpeg_path):
            raise EngineError("ffmpeg_missing", "FFmpeg was not found.")
        if self.output_stream is None:
            try:
                self.prepare_file_playback_output()
            except EngineError:
                raise
            except Exception as error:
                raise EngineError("audio_start_failed", str(error)) from error
        if self.file_audio_source is None:
            settings = self.settings
            tail_blocks = max(
                2,
                int((settings.extra_time + settings.crossfade_time) / settings.block_time) + 2,
            )
            try:
                self.file_audio_source = FileAudioSource(
                    self.ffmpeg_path,
                    self.file_input_path,
                    self.samplerate,
                    self.block_frame,
                    self.handle_file_audio_block,
                    tail_blocks=tail_blocks,
                )
            except OSError as error:
                raise EngineError("file_error", str(error)) from error
        self.file_audio_source.play()
        self.status("playing")
        self.emit_state()

    def file_pause(self):
        if self.file_audio_source is not None:
            self.file_audio_source.pause()
        self.status("paused")
        self.emit_state()

    def file_stop(self):
        self.stop_file_playback()
        self.status("playback_stopped")
        self.emit_state()

    def file_seek(self, seconds):
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise EngineError("bad_request", "seconds must be a number")
        if self.file_audio_source is not None:
            self.file_audio_source.seek(max(0.0, float(seconds)))

    def prepare_file_playback_output(self):
        """Open an output-only stream to preview a file without converting."""
        self.set_devices()
        self.input_channels = 1
        self.output_channels = self.get_output_channels()
        self.samplerate = self.get_automatic_samplerate(48000)
        self.zc = self.samplerate // 100
        self.block_frame = max(
            self.zc,
            int(round(self.settings.block_time * self.samplerate / self.zc)) * self.zc,
        )
        self.start_file_stream()

    def handle_file_audio_block(self, block):
        if self.running:
            self.audio_callback(block[:, None], self.block_frame, None, None)
            return
        volume = float(np.clip(self.settings.file_input_volume, 0.0, 1.0))
        block = np.clip(block * np.float32(volume), -1.0, 1.0)
        self.enqueue_audio_target(self.output_queue, block)
        self.enqueue_audio_target(
            self.monitor_queue, block * np.float32(self.settings.monitor_gain)
        )

    def stop_file_playback(self):
        if self.file_audio_source is not None:
            self.file_audio_source.stop()
            self.file_audio_source = None
        self.file_state = (False, False)
        if not self.running and self.output_stream is not None:
            # Close the preview-only output opened by file_play.
            for stream_name in ("output_stream", "monitor_stream"):
                stream = getattr(self, stream_name)
                if stream is not None:
                    stream.abort()
                    stream.close()
                    setattr(self, stream_name, None)
            self.stream = None

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def start_recording(self):
        if not self.running:
            raise EngineError("record_requires_running", "Start conversion before recording.")
        if self.recorder.active:
            return
        folder = self.settings.recording_folder or DEFAULT_RECORDING_FOLDER
        try:
            self.recorder.start(folder, self.samplerate, self.settings.recording_mode)
        except OSError as error:
            raise EngineError("recording_failed", str(error)) from error
        self.status("recording_started")
        self.emit_state()

    def stop_recording(self):
        if not self.recorder.active:
            return []
        self.recorder.stop()
        saved_paths = list(getattr(self.recorder, "last_saved_paths", []))
        if saved_paths:
            self.status("recording_saved", paths=saved_paths)
        self.emit_state()
        return saved_paths
