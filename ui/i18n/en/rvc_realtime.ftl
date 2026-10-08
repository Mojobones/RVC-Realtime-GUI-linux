app-title = RVC Realtime

## Drag and drop
drop-title = Drop to add models
drop-detail = Each .pth becomes a model; drop its .index file along with it.
model-imported = Added model: { $names }

## Navigation
page-model = Model
page-audio = Audio
page-performance = Performance
page-recording = Recording
page-log = Log

## Engine connection
engine-starting = Starting the engine…
engine-starting-detail = Loading PyTorch, models, and audio devices.
engine-unavailable = The engine is not available
engine-retrying = Retrying automatically.

## Model page
section-models = Models
model = Model
models-empty = No models yet. Drag a .pth file (and its .index, if you have one) onto this window to add it.
model-files = Model: { $model } / Index: { $index }
none = None
reload = Reload
section-voice = Voice
pitch = Pitch (semitones)
formant = Formant
index-rate = Index rate
section-levels = Levels
input-gain = Input gain
output-gain = Output gain
monitor-gain = Monitor gain
noise-gate = Noise gate
off = Off
reset-model-settings = Reset model settings

## Audio page
section-input = Input
input-source = Source
source-microphone = Microphone / audio device
source-file = Audio file
audio-file = File
no-file = No file selected
browse = Browse…
pick-audio-file = Choose an audio file
file-volume = File volume
input-device = Input device
section-output = Output
output-device = Output device
monitor-device = Monitor device
monitor-disabled = Disabled
reload-devices = Reload devices
sample-rate-value = Sample rate: { $rate } Hz
sample-rate-idle = Sample rate: — (not running)
jack-hint = [JACK] devices give the lowest latency. Set the period with PIPEWIRE_QUANTUM, e.g. 128/48000.

## Performance page
section-buffering = Buffering
chunk = Chunk
crossfade = Crossfade
extra = Extra inference buffer
section-inference = Inference
pitch-detector = Pitch detector
f0-rmvpe = RMVPE (recommended)
f0-fcpe = FCPE
f0-pm = PM (fastest)
volume-envelope = Volume envelope
gpu = GPU
gpu-automatic = Automatic (recommended)
section-noise = Noise reduction
input-denoise = Input noise reduction
output-denoise = Output noise reduction
reset-performance = Reset performance settings

## Recording page
section-recording = Recording
recording-mode = Mode
recording-separate = Separate files (input + output)
recording-mix = Mix (input + output)
recording-stereo = Split L/R (left: input / right: output)
save-folder = Save folder
pick-recording-folder = Choose a recording folder
change = Change…
open-folder = Open folder
record = Record
stop-recording = Stop recording
record-hint = Start conversion before recording.

## Log page
save-log = Save log
clear-log = Clear log
log-saved = Log saved: { $path }

## Control bar
start = Start
starting = Starting…
converting = Converting
passthrough = Passthrough
passthrough-active = Passthrough active
meter-in = In
meter-out = Out
meter-mon = Mon
latency = Latency: { $value } ms
inference = Inference: { $value } ms

## Engine status
status-preparing = Preparing…
status-loading-model = Loading model…
status-preparing-inference = Preparing inference…
status-starting-audio = Starting audio devices…
status-conversion-started = Conversion started
status-passthrough-started = Passthrough started
status-conversion-stopped = Conversion stopped
status-passthrough-stopped = Passthrough stopped
status-settings-changed = Stopped for a settings change
status-stream-stopped = The audio device stopped unexpectedly
status-file-selected = Audio file selected: { $name }
status-playing = Playing
status-paused = Paused
status-playback-stopped = Playback stopped
status-recording-started = Recording started
status-importing-model = Adding model…
status-recording-saved = Recording saved:
    { $paths }

## Engine errors
error-no-model = Select a model from the models folder.
error-model-file-missing = The model file was not found.
error-index-missing = The selected model has no .index file. Set Index rate to 0 or add an .index file to its folder.
error-no-audio-file = Select an audio file first.
error-ffmpeg-missing = FFmpeg was not found. Install ffmpeg and make sure it is on PATH.
error-record-requires-running = Start conversion before recording.
error-no-common-samplerate = The input, output, and monitor devices have no common sample rate. Set them to the same rate (usually 48 kHz).
error-audio-start-failed = Could not start the audio devices: { $detail }
error-monitor-failed = The monitor device could not start; continuing on the main output only. ({ $detail })
error-not-ready = The engine is still starting.
error-import-no-model-file = Drop an RVC model's .pth file. Its .index file can be dropped along with it.
error-import-failed = Could not add the model: { $detail }
