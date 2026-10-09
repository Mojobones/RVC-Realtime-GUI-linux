"""Engine settings, defaults, and their JSON persistence.

Global settings live in ``configs/engine.json``.  The per-model settings
(gains, noise gate, pitch, formant, index rate, protect, measured source
pitch) live beside each model in ``realtime_settings.json`` using the same
keys as the Tk build, so both front ends share them.
"""

import json
import math
import os
from dataclasses import asdict, dataclass, fields

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "engine.json")
DEFAULT_RECORDING_FOLDER = os.path.join(PROJECT_ROOT, "recordings")

F0_METHODS = ("pm", "rmvpe", "fcpe")
INPUT_SOURCES = ("microphone", "file")
RECORDING_MODES = ("separate", "mix", "stereo")
#: How context hold decides a block is silent: by loudness, or (experimental)
#: by RNNoise's voice-activity probability.
HOLD_DETECTORS = ("level", "voice")
#: Output-level indicator targets (tools/loudness.py TARGETS_LUFS).
LOUDNESS_TARGETS = ("voice_chat", "streaming")

MODEL_SETTINGS_FILENAME = "realtime_settings.json"
# Protocol name -> key stored in each model's realtime_settings.json.
MODEL_SETTING_FILE_KEYS = {
    "input_gain_db": "input_gain_db",
    "output_gain_db": "output_gain_db",
    "monitor_gain_db": "monitor_gain_db",
    "noise_gate_db": "threhold",
    "pitch": "pitch",
    "formant": "formant",
    "index_rate": "index_rate",
    "protect": "protect",
    "source_pitch_hz": "source_pitch_hz",
}
MODEL_SETTING_KEYS = tuple(MODEL_SETTING_FILE_KEYS)
#: "Reset model settings" keeps the measured source pitch: it is a fact
#: about the model's voice, not a preference.
RESETTABLE_MODEL_SETTING_KEYS = tuple(
    key for key in MODEL_SETTING_KEYS if key != "source_pitch_hz"
)

# Settings that only matter when the next stream starts; changing one stops
# a running stream, as the Tk build did.
RESTART_SETTING_KEYS = frozenset(
    (
        "model_name",
        "gpu",
        "input_source",
        "input_device",
        "output_device",
        "monitor_device",
        "block_time",
        "extra_time",
    )
)


@dataclass
class EngineSettings:
    # Global settings (engine.json).
    model_name: str = ""
    gpu: str = "auto"
    input_source: str = "microphone"
    input_device: str = ""
    output_device: str = ""
    monitor_device: str | None = None
    block_time: float = 0.25
    extra_time: float = 2.5
    input_denoise: bool = False
    #: Freeze the model's past context during silence (see engine/core.py).
    hold_context: bool = True
    hold_detector: str = "level"
    #: What the output-level indicator aims for: voice chat (-21 LUFS) or
    #: streaming (-16 LUFS).
    loudness_target: str = "voice_chat"
    output_denoise: bool = False
    # Keep the original real-time RVC defaults: no envelope post-processing
    # or index blending until the user enables them.
    rms_mix_rate: float = 0.0
    f0method: str = "rmvpe"
    #: Viterbi pitch tracking for RMVPE (infer/rmvpe.py): the most likely
    #: continuous pitch path instead of each 10 ms frame's own peak.
    pitch_smoothing: bool = True
    recording_folder: str = DEFAULT_RECORDING_FOLDER
    recording_mode: str = "separate"
    file_input_volume: float = 1.0
    # Per-model settings (realtime_settings.json beside the model).
    input_gain_db: float = 0.0
    output_gain_db: float = 0.0
    monitor_gain_db: float = 0.0
    noise_gate_db: float = -60.0
    pitch: float = 0.0
    formant: float = 0.0
    index_rate: float = 0.0
    #: RVC's consonant protection (0.5 = off); only matters with index_rate > 0.
    protect: float = 0.33
    #: Median pitch of the model's source speaker (Hz, 0 = not measured),
    #: from a clip analysed with ``analyze_source_pitch``.
    source_pitch_hz: float = 0.0

    @property
    def input_gain(self):
        return db_to_linear(self.input_gain_db)

    @property
    def output_gain(self):
        return db_to_linear(self.output_gain_db)

    @property
    def monitor_gain(self):
        return db_to_linear(self.monitor_gain_db)

    def to_dict(self):
        return asdict(self)

    def global_dict(self):
        return {
            key: value
            for key, value in asdict(self).items()
            if key not in MODEL_SETTING_KEYS
        }

    def model_dict(self):
        return {key: getattr(self, key) for key in MODEL_SETTING_KEYS}


SETTING_TYPES = {field.name: field.type for field in fields(EngineSettings)}
PERFORMANCE_DEFAULT_KEYS = (
    "block_time",
    "extra_time",
    "input_denoise",
    "output_denoise",
    "hold_context",
    "hold_detector",
    "rms_mix_rate",
    "f0method",
    "pitch_smoothing",
)


def db_to_linear(db):
    return 10.0 ** (float(db) / 20.0)


class SettingError(ValueError):
    def __init__(self, key, message):
        super().__init__(message)
        self.key = key


def coerce_setting(key, value):
    """Validate one protocol value and convert it to the stored type."""
    if key not in SETTING_TYPES:
        raise SettingError(key, f"Unknown setting: {key}")
    kind = SETTING_TYPES[key]
    if kind is bool:
        if not isinstance(value, bool):
            raise SettingError(key, f"{key} must be true or false")
        return value
    if kind is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingError(key, f"{key} must be a number")
        value = float(value)
        if not math.isfinite(value):
            raise SettingError(key, f"{key} must be finite")
        if key in ("block_time",):
            value = max(0.02, value)
        if key in ("extra_time", "index_rate"):
            value = max(0.0, value)
        if key in ("rms_mix_rate", "index_rate", "file_input_volume"):
            value = min(max(value, 0.0), 1.0)
        if key == "protect":
            value = min(max(value, 0.0), 0.5)
        if key == "source_pitch_hz":
            value = max(value, 0.0)
        return value
    if key == "monitor_device":
        if value is not None and not isinstance(value, str):
            raise SettingError(key, "monitor_device must be a string or null")
        return value or None
    if not isinstance(value, str):
        raise SettingError(key, f"{key} must be a string")
    allowed = {
        "f0method": F0_METHODS,
        "input_source": INPUT_SOURCES,
        "recording_mode": RECORDING_MODES,
        "hold_detector": HOLD_DETECTORS,
        "loudness_target": LOUDNESS_TARGETS,
    }.get(key)
    if allowed and value not in allowed:
        raise SettingError(key, f"{key} must be one of {', '.join(allowed)}")
    return value


def load_settings(path=ENGINE_CONFIG_PATH):
    """Read engine.json; invalid or unknown entries fall back to defaults."""
    settings = EngineSettings()
    try:
        with open(path, "r", encoding="utf-8") as config_file:
            saved = json.load(config_file)
    except (OSError, ValueError):
        return settings
    if not isinstance(saved, dict):
        return settings
    for key, value in saved.items():
        if key in MODEL_SETTING_KEYS:
            continue
        try:
            setattr(settings, key, coerce_setting(key, value))
        except SettingError:
            continue
    return settings


def save_settings(settings, path=ENGINE_CONFIG_PATH):
    write_json_atomic(path, settings.global_dict())


def load_model_settings(directory):
    """Return one model's per-model settings, or the defaults."""
    defaults = EngineSettings()
    values = {key: getattr(defaults, key) for key in MODEL_SETTING_KEYS}
    if directory is None:
        return values
    path = os.path.join(directory, MODEL_SETTINGS_FILENAME)
    try:
        with open(path, "r", encoding="utf-8") as settings_file:
            saved = json.load(settings_file)
    except FileNotFoundError:
        return values
    except (OSError, ValueError):
        print(f"Could not read model settings: {path}")
        return values
    if not isinstance(saved, dict):
        return values
    for key, file_key in MODEL_SETTING_FILE_KEYS.items():
        value = saved.get(file_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            values[key] = float(value)
    return values


def save_model_settings(directory, settings):
    path = os.path.join(directory, MODEL_SETTINGS_FILENAME)
    write_json_atomic(
        path,
        {
            file_key: float(getattr(settings, key))
            for key, file_key in MODEL_SETTING_FILE_KEYS.items()
        },
    )


def write_json_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = path + ".tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8") as output:
            json.dump(data, output, ensure_ascii=False, indent=2)
        os.replace(temporary_path, path)
    except OSError:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise
