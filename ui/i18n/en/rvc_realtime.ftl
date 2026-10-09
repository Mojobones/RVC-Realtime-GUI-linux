app-title = RVC Realtime

## Drag and drop
drop-title = Drop to add models or measure a voice
drop-detail = Each .pth becomes a model; drop its .index file along with it. Drop audio clips or a folder of them to measure the selected model's voice pitch.
model-imported = Added model: { $names }

## Typed values
value-click-to-type = Click to type a value
value-invalid = Enter a number

## Navigation
page-model = Model
page-library = Library
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
assets-missing-banner = Inference assets are missing: { $paths }. Copy the assets/ folder from the CUDA 12.8 release package into the project folder, then restart.
section-models = Models
model = Model
models-empty = No models yet. Drag a .pth file (and its .index, if you have one) onto this window to add it.
model-files = Model: { $model } / Index: { $index }
none = None
reload = Reload
section-voice = Voice
pitch = Pitch (semitones)
pitch-detail = How far to shift your voice: +12 is one octave up. Pitch match below can suggest a value.
pitch-match = Pitch match
pitch-match-no-source = Model's voice: not measured. Drop clips of it here, ideally the audio it was trained on.
pitch-match-analyzing = Model's voice: measuring…
pitch-match-source = Model's voice: { $hz } Hz median
pitch-match-voice = Your voice: { $hz } Hz median ({ $minutes } min of speech)
pitch-match-listening = Your voice: talk while converting to measure it ({ $seconds } of { $needed } s)
pitch-match-apply = Apply
pitch-match-measure = Measure clips…
pitch-match-reset-voice = Forget your measured voice and start over
pick-pitch-clips = Choose clips of the model's voice
output-level = Output level
output-level-idle = While converting, shows how loud you sound to others. Voice chat aims for -21 LUFS; streaming for -16 LUFS, the usual level of streamed voice.
output-level-listening = Talk to measure your level.
output-level-measuring = { $lufs } LUFS so far; measuring ({ $seconds } of 5 s of speech)
output-level-reading = { $lufs } LUFS (target { $target }): { $verdict }
loudness-target-voice-chat = Voice chat (-21)
loudness-target-streaming = Streaming (-16)
output-level-good = a good level.
output-level-quiet = too quiet; raise the output gain.
output-level-quiet-peaky = quiet, but your peaks already reach full scale, so raising the gain would squash them.
output-level-loud = too loud; lower the output gain.
output-level-squashed = too loud; peaks are being squashed. Lower the output gain.
output-level-apply = Apply
level-good = Level: good
level-quiet = Level: too quiet
level-loud = Level: too loud
level-measuring = Level: measuring…
library-search = Search models
library-sort-name = By name
library-sort-newest = Newest first
library-count = { $shown } of { $total } models
library-no-match = No models match your search.
library-voice = { $voice } ({ $count })
library-checkpoint = { $detail } · { $epochs } epochs
library-epochs = { $epochs } epochs
library-index-yes = has index
library-index-no = no index
library-use = Use
library-selected = Selected
library-open-folder = Open folder
library-delete = Move to Trash
library-delete-in-use = Stop converting to delete this model
delete-model-title = Move model to the Trash?
delete-model-body = { $name } and its saved settings (pitch, gains) will be moved to the Trash. You can restore it from there.
delete-model-confirm = Move to Trash
cancel = Cancel
status-model-deleted = Moved { $name } to the Trash
error-model-in-use = Stop converting before deleting this model.
error-delete-failed = Could not move the model to the Trash: { $detail }
library-rename = Rename
library-rename-in-use = Stop converting to rename this model
rename-model-title = Rename model
rename-model-body = This renames the model's folder; its saved settings stay with it.
rename-model-confirm = Rename
status-model-renamed = Renamed { $name } to { $new_name }
error-model-name-taken = Another model already has that name.
error-bad-model-name = That name can't be used: it can't be empty, start with a dot, or contain / \ : * ? " < > |
error-rename-failed = Could not rename the model: { $detail }
formant = Formant
formant-detail = Shifts the voice's resonance, how big the speaker sounds, without changing pitch. Higher sounds smaller and brighter, lower sounds bigger and deeper. 0 keeps the model's own.
index-rate = Index rate
index-rate-detail = Pulls your voice toward the model's own recordings using its .index file. Higher can sound closer to the speaker, but also brings in their accent and quirks. 0 is off.
protect = Consonant protection
protect-detail = Only matters when Index rate is above 0: keeps breaths and consonants (s, t, k) closer to your own so they don't smear. Lower protects more.
section-levels = Levels
input-gain = Input gain
input-gain-detail = Boosts or cuts your mic before conversion. The converted voice follows your input level, so this also changes how loud you come out.
output-gain = Output gain
output-gain-detail = Volume of the converted voice on the output device, which is what others hear. Output level below suggests a value.
monitor-gain = Monitor gain
monitor-gain-detail = Volume of the copy you hear yourself on the monitor device. Doesn't change what others hear.
noise-gate = Noise gate
noise-gate-detail = Mutes your mic whenever it's quieter than this, so background sound between words isn't converted. Off at -60 dB.
off = Off
reset-model-settings = Reset model settings

## Audio page
section-input = Input
input-source = Source
input-source-detail = Convert your microphone live, or play an audio file through the model.
source-microphone = Microphone / audio device
source-file = Audio file
audio-file = File
no-file = No file selected
browse = Browse…
pick-audio-file = Choose an audio file
file-volume = File volume
file-volume-detail = Playback volume of the converted file.
input-device = Input device
input-device-detail = Your microphone.
section-output = Output
output-device = Output device
output-device-detail = Where the converted voice goes, usually the virtual mic that Discord or OBS listens to.
monitor-device = Monitor device
monitor-device-detail = An optional second output so you can hear yourself, such as your headphones.
monitor-disabled = Disabled
reload-devices = Reload devices
sample-rate-value = Sample rate: { $rate } Hz
sample-rate-idle = Sample rate: — (not running)
jack-hint = [JACK] devices give the lowest latency. Set the period with PIPEWIRE_QUANTUM, e.g. 128/48000.

## Performance page
section-buffering = Buffering
chunk = Chunk
chunk-detail = How much audio is converted at a time. Every bit adds the same amount of delay; shorter chunks lower the delay but can stutter if the GPU can't keep up (watch Inference in the bottom bar). Changing it restarts conversion.
extra = Extra inference buffer
extra-detail = How much of your recent speech the model looks back at for each chunk. More context gives steadier tone and pitch at a small GPU cost, with no added delay. Changing it restarts conversion.
section-inference = Inference
pitch-detector = Pitch detector
pitch-detector-detail = How your pitch is measured. RMVPE is the most accurate; FCPE is lighter; PM is fastest but roughest.
pitch-smoothing = Pitch smoothing
pitch-smoothing-detail = Follows your pitch as one continuous line instead of judging each 10 ms on its own: steadier held notes and no octave flips. No added delay.
pitch-smoothing-rmvpe-only = Only applies to the RMVPE pitch detector.
f0-rmvpe = RMVPE (recommended)
f0-fcpe = FCPE
f0-pm = PM (fastest)
volume-envelope = Volume envelope
volume-envelope-detail = How closely the converted voice's loudness follows yours from moment to moment. 0 matches your volume exactly; 1 keeps the model's own loudness.
gpu = GPU
gpu-detail = The graphics card that runs the model. Changing it restarts conversion.
gpu-automatic = Automatic (recommended)
hold-context = Hold context during silence
hold-context-detail = Keeps the model's memory of your last words while you're quiet, so the first word after a pause isn't slurred.
input-denoise-rnnoise = Removes background noise like keyboards, fans and traffic from your mic before conversion. Uses RNNoise; adds 20 ms of delay.
input-denoise-spectral = Removes background noise from your mic before conversion. Uses spectral gating, which adds 40 ms of delay; install RNNoise (sudo pacman -S rnnoise) for better removal with less delay.
hold-detector = Silence detection
hold-detector-level = Loudness
hold-detector-voice = Voice detection (experimental)
hold-detector-level-detail = Silence is anything below -50 dBFS (or the noise gate). Best in quiet rooms.
hold-detector-voice-detail = RNNoise judges whether you are speaking, so fans and hum don't count as speech. Best in noisy rooms.
section-noise = Noise reduction
input-denoise = Input noise reduction
output-denoise = Output noise reduction
output-denoise-detail = Cleans hiss from the converted voice with spectral gating. Can make the voice slightly duller; leave off unless you hear hiss.
reset-performance = Reset performance settings

## Recording page
section-recording = Recording
recording-mode = Mode
recording-mode-detail = Recordings capture your raw mic and the converted voice; choose how they're saved.
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
latency-detail = Delay from your mic to the output: chunk, crossfade and device buffers, plus the audio queued right now. If it jumps, the engine fell behind for a moment; it catches up during your next pause. The real end-to-end delay is still somewhat higher.
inference = Inference: { $value } ms
inference-detail = Time to convert each chunk. It must stay well below the chunk length, or the audio stutters.

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
status-analyzing-pitch = Measuring the voice pitch for { $name }…
status-pitch-analyzed = { $name }: { $hz } Hz median over { $minutes } min of speech
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
error-assets-missing = Can't start: inference assets are missing ({ $paths }). Copy the assets/ folder from the release package into the project folder.
error-model-load-failed = Could not load the model: { $detail }
error-no-voice-found = No voiced speech was found in those clips.
error-pitch-analysis-busy = Clips are already being measured; wait for that to finish.
error-pitch-analysis-failed = Could not measure the clips: { $detail }
