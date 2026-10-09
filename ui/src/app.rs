//! Application state and message handling.

use std::collections::HashMap;

use cosmic::app::{Core, Task};
use cosmic::dialog::file_chooser;
use cosmic::iced::Subscription;
use cosmic::prelude::*;
use cosmic::widget::{self, nav_bar, toaster};
use serde_json::{Value, json};

use crate::drop::DroppedFiles;
use crate::engine::{self, ErrorInfo, Handle, Meters, Settings, State};
use crate::fl;
use crate::pages;

pub const APP_ID: &str = "io.github.rvc_realtime.RvcRealtime";
const MAX_LOG_CHARS: usize = 40_000;
/// Clicking away from a typed value applies it after this long, unless the
/// unfocus came from Escape (whose key event arrives just after).
const BLUR_APPLY_DELAY: std::time::Duration = std::time::Duration::from_millis(40);

/// The one inline field used to type a slider value.
pub static VALUE_INPUT_ID: std::sync::LazyLock<widget::Id> =
    std::sync::LazyLock::new(|| widget::Id::new("slider-value-input"));
/// The name field of the rename dialog.
pub static RENAME_INPUT_ID: std::sync::LazyLock<widget::Id> =
    std::sync::LazyLock::new(|| widget::Id::new("rename-model-input"));

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Page {
    Model,
    Library,
    Audio,
    Performance,
    Recording,
    Log,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Function {
    Convert,
    Passthrough,
}

impl Function {
    pub fn id(self) -> &'static str {
        match self {
            Function::Convert => "vc",
            Function::Passthrough => "passthrough",
        }
    }
}

/// Numeric settings edited with sliders.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Num {
    InputGain,
    OutputGain,
    MonitorGain,
    NoiseGate,
    Pitch,
    Formant,
    IndexRate,
    Protect,
    RmsMix,
    BlockTime,
    Extra,
    FileVolume,
}

impl Num {
    pub fn key(self) -> &'static str {
        match self {
            Num::InputGain => "input_gain_db",
            Num::OutputGain => "output_gain_db",
            Num::MonitorGain => "monitor_gain_db",
            Num::NoiseGate => "noise_gate_db",
            Num::Pitch => "pitch",
            Num::Formant => "formant",
            Num::IndexRate => "index_rate",
            Num::Protect => "protect",
            Num::RmsMix => "rms_mix_rate",
            Num::BlockTime => "block_time",
            Num::Extra => "extra_time",
            Num::FileVolume => "file_input_volume",
        }
    }

    /// Slider range and step, matching the Tk build's controls.
    pub fn range(self) -> (f32, f32, f32) {
        match self {
            Num::InputGain | Num::OutputGain | Num::MonitorGain => (-24.0, 24.0, 0.5),
            Num::NoiseGate => (-60.0, 0.0, 1.0),
            // 0.1-semitone steps allow fine tuning such as +11.5.
            Num::Pitch => (-24.0, 24.0, 0.1),
            Num::Formant => (-5.0, 5.0, 0.01),
            Num::IndexRate | Num::RmsMix => (0.0, 1.0, 0.01),
            Num::Protect => (0.0, 0.5, 0.01),
            Num::BlockTime => (0.02, 1.5, 0.01),
            Num::Extra => (0.05, 5.0, 0.01),
            Num::FileVolume => (0.0, 1.0, 0.01),
        }
    }

    /// The value as plain text for the edit field (no unit or sign padding).
    pub fn edit_text(self, value: f32) -> String {
        let shown = if self == Num::FileVolume {
            value * 100.0
        } else {
            value
        };
        let text = format!("{shown:.3}");
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    }

    /// Parse a typed value: units, a leading `+`, and a decimal comma are
    /// accepted, `off` means -60 dB for the noise gate, and the file volume is
    /// typed in percent.  The result is clamped to the slider's range.
    pub fn parse(self, text: &str) -> Option<f32> {
        let (min, max, _) = self.range();
        let text = text.trim().to_lowercase();
        let off = text == "off" || text == fl!("off").to_lowercase();
        if self == Num::NoiseGate && off {
            return Some(min);
        }
        if self == Num::Protect && off {
            return Some(max);
        }
        let number = text
            .trim_end_matches(['%', 's'])
            .trim_end_matches("db")
            .trim()
            .trim_start_matches('+')
            .replace(',', ".");
        let mut value: f32 = number.parse().ok().filter(|v: &f32| v.is_finite())?;
        if self == Num::FileVolume {
            value /= 100.0;
        }
        Some(value.clamp(min, max))
    }

    /// Changing these stops a running stream, so they are sent on release.
    pub fn needs_restart(self) -> bool {
        matches!(self, Num::BlockTime | Num::Extra)
    }

    pub fn get(self, settings: &Settings) -> f32 {
        match self {
            Num::InputGain => settings.input_gain_db,
            Num::OutputGain => settings.output_gain_db,
            Num::MonitorGain => settings.monitor_gain_db,
            Num::NoiseGate => settings.noise_gate_db,
            Num::Pitch => settings.pitch,
            Num::Formant => settings.formant,
            Num::IndexRate => settings.index_rate,
            Num::Protect => settings.protect,
            Num::RmsMix => settings.rms_mix_rate,
            Num::BlockTime => settings.block_time,
            Num::Extra => settings.extra_time,
            Num::FileVolume => settings.file_input_volume,
        }
    }

    fn set(self, settings: &mut Settings, value: f32) {
        let field = match self {
            Num::InputGain => &mut settings.input_gain_db,
            Num::OutputGain => &mut settings.output_gain_db,
            Num::MonitorGain => &mut settings.monitor_gain_db,
            Num::NoiseGate => &mut settings.noise_gate_db,
            Num::Pitch => &mut settings.pitch,
            Num::Formant => &mut settings.formant,
            Num::IndexRate => &mut settings.index_rate,
            Num::Protect => &mut settings.protect,
            Num::RmsMix => &mut settings.rms_mix_rate,
            Num::BlockTime => &mut settings.block_time,
            Num::Extra => &mut settings.extra_time,
            Num::FileVolume => &mut settings.file_input_volume,
        };
        *field = value;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Toggle {
    InputDenoise,
    OutputDenoise,
    HoldContext,
    PitchSmoothing,
}

impl Toggle {
    fn key(self) -> &'static str {
        match self {
            Toggle::InputDenoise => "input_denoise",
            Toggle::OutputDenoise => "output_denoise",
            Toggle::HoldContext => "hold_context",
            Toggle::PitchSmoothing => "pitch_smoothing",
        }
    }
}

pub const F0_METHODS: [&str; 3] = ["rmvpe", "fcpe", "pm"];
pub const RECORDING_MODES: [&str; 3] = ["separate", "mix", "stereo"];
pub const HOLD_DETECTORS: [&str; 2] = ["level", "voice"];
pub const LOUDNESS_TARGETS: [&str; 2] = ["voice_chat", "streaming"];
pub const INPUT_SOURCES: [&str; 2] = ["microphone", "file"];

/// Connection to the engine process, as shown to the user.
#[derive(Clone, Debug)]
pub enum Connection {
    Connecting,
    /// Connected; the engine is loading PyTorch and the models.
    Starting,
    Ready,
    Failed(String),
}

/// What to do with the response to a request we sent.
#[derive(Clone, Copy, Debug)]
enum Pending {
    Start,
    Import,
    RecordStop,
    SaveLog,
}

#[derive(Clone, Debug)]
pub enum Message {
    Engine(engine::Event),
    Num(Num, f32),
    NumReleased(Num),
    /// Clicked a slider's value to type one instead.
    EditValue(Num),
    EditInput(String),
    EditSubmit,
    /// The value field lost focus (clicked away, Tab, or Escape).
    EditBlur,
    /// Apply the value typed before the field lost focus.
    EditBlurApply,
    EscapePressed,
    Toggle(Toggle, bool),
    SelectModel(usize),
    SelectGpu(usize),
    SelectInputSource(usize),
    SelectInput(usize),
    SelectOutput(usize),
    SelectMonitor(usize),
    SelectF0(usize),
    SelectRecordingMode(usize),
    SelectHoldDetector(usize),
    SelectLoudnessTarget(usize),
    Run(Function),
    Reset(&'static str),
    ReloadModels,
    ReloadDevices,
    PickAudioFile,
    PickRecordingFolder,
    Picked(PickTarget, Option<String>),
    FilePlay,
    FilePause,
    FileStop,
    FileSeek(f32),
    FileSeekReleased,
    ToggleRecording,
    OpenRecordingFolder,
    SaveLog,
    ClearLog,
    CloseToast(toaster::ToastId),
    /// Files are being dragged over the window (true) or left it (false).
    DragHover(bool),
    FilesDropped(Option<DroppedFiles>),
    /// Choose clips of the model's voice to measure its pitch.
    PickPitchClips,
    PitchClipsPicked(Vec<String>),
    /// Set the pitch to the recommendation.
    ApplyRecommendedPitch,
    /// Forget the measured voice pitch and start over.
    ResetVoicePitch,
    /// Change the output gain by the output-level suggestion.
    ApplyLoudness,
    OpenLibrary,
    LibrarySearch(String),
    LibrarySort(usize),
    SelectModelByName(String),
    OpenModelFolder(String),
    /// Ask before moving a model to the Trash.
    AskDeleteModel(String),
    ConfirmDeleteModel,
    CancelDeleteModel,
    AskRenameModel(String),
    RenameInput(String),
    ConfirmRenameModel,
    CancelRenameModel,
}

#[derive(Clone, Copy, Debug)]
pub enum PickTarget {
    AudioFile,
    RecordingFolder,
}

pub struct App {
    core: Core,
    nav: nav_bar::Model,
    engine: Option<Handle>,
    pub connection: Connection,
    pub state: Option<State>,
    pub meters: Meters,
    pub log: String,
    pub status: String,
    /// Slider being dragged; its local value wins over incoming state.
    dragging: Option<Num>,
    pub seek_preview: Option<f32>,
    /// Slider value being typed in, with its text so far.
    pub editing: Option<(Num, String)>,
    pub starting: bool,
    pub drag_hover: bool,
    pending: HashMap<u64, Pending>,
    /// Request id of the log history fetched after connecting.
    log_request: Option<u64>,
    pub toasts: toaster::Toasts<Message>,
    // Dropdown labels, rebuilt when the state changes.
    pub model_names: Vec<String>,
    pub gpu_labels: Vec<String>,
    pub input_labels: Vec<String>,
    pub output_labels: Vec<String>,
    pub monitor_labels: Vec<String>,
    pub source_labels: Vec<String>,
    pub f0_labels: Vec<String>,
    pub recording_mode_labels: Vec<String>,
    pub hold_detector_labels: Vec<String>,
    pub loudness_target_labels: Vec<String>,
    /// Library page: search text, sort (index into LIBRARY_SORTS) and the
    /// model awaiting delete confirmation.
    pub library_query: String,
    pub library_sort: usize,
    pub library_sort_labels: Vec<String>,
    pub confirm_delete: Option<String>,
    /// Model being renamed and the new name typed so far.
    pub renaming: Option<(String, String)>,
    /// A typed value to apply once the unfocus is known not to be Escape.
    blur_pending: Option<(Num, String)>,
}

/// Library sort orders: by name, or newest first.
pub const LIBRARY_SORTS: [&str; 2] = ["name", "newest"];

impl cosmic::Application for App {
    type Executor = cosmic::executor::Default;
    type Flags = ();
    type Message = Message;
    const APP_ID: &'static str = APP_ID;

    fn core(&self) -> &Core {
        &self.core
    }

    fn core_mut(&mut self) -> &mut Core {
        &mut self.core
    }

    fn init(core: Core, _flags: ()) -> (Self, Task<Message>) {
        let mut nav = nav_bar::Model::default();
        for (page, title, icon) in [
            (Page::Model, fl!("page-model"), "avatar-default-symbolic"),
            (Page::Library, fl!("page-library"), "view-list-symbolic"),
            (Page::Audio, fl!("page-audio"), "audio-card-symbolic"),
            (
                Page::Performance,
                fl!("page-performance"),
                "speedometer-symbolic",
            ),
            (
                Page::Recording,
                fl!("page-recording"),
                "media-record-symbolic",
            ),
            (Page::Log, fl!("page-log"), "utilities-terminal-symbolic"),
        ] {
            nav.insert()
                .text(title)
                .icon(widget::icon::from_name(icon))
                .data(page);
        }
        nav.activate_position(0);
        let mut app = App {
            core,
            nav,
            engine: None,
            connection: Connection::Connecting,
            state: None,
            meters: Meters::default(),
            log: String::new(),
            status: String::new(),
            dragging: None,
            seek_preview: None,
            editing: None,
            starting: false,
            drag_hover: false,
            pending: HashMap::new(),
            log_request: None,
            toasts: toaster::Toasts::new(Message::CloseToast),
            model_names: Vec::new(),
            gpu_labels: Vec::new(),
            input_labels: Vec::new(),
            output_labels: Vec::new(),
            monitor_labels: Vec::new(),
            source_labels: vec![fl!("source-microphone"), fl!("source-file")],
            f0_labels: vec![fl!("f0-rmvpe"), fl!("f0-fcpe"), fl!("f0-pm")],
            hold_detector_labels: vec![fl!("hold-detector-level"), fl!("hold-detector-voice")],
            loudness_target_labels: vec![
                fl!("loudness-target-voice-chat"),
                fl!("loudness-target-streaming"),
            ],
            recording_mode_labels: vec![
                fl!("recording-separate"),
                fl!("recording-mix"),
                fl!("recording-stereo"),
            ],
            library_query: String::new(),
            library_sort: 0,
            library_sort_labels: vec![fl!("library-sort-name"), fl!("library-sort-newest")],
            confirm_delete: None,
            renaming: None,
            blur_pending: None,
        };
        let task = app.update_title();
        (app, task)
    }

    fn nav_model(&self) -> Option<&nav_bar::Model> {
        Some(&self.nav)
    }

    fn on_nav_select(&mut self, id: nav_bar::Id) -> Task<Message> {
        self.nav.activate(id);
        self.update_title()
    }

    fn subscription(&self) -> Subscription<Message> {
        use cosmic::iced::keyboard::{self, Key, key::Named};
        use cosmic::iced::{Event, event};
        Subscription::batch([
            Subscription::run(engine::connection).map(Message::Engine),
            // Every Escape press, also ones a text field already handled.
            event::listen_with(|event, _status, _window| match event {
                Event::Keyboard(keyboard::Event::KeyPressed {
                    key: Key::Named(Named::Escape),
                    ..
                }) => Some(Message::EscapePressed),
                _ => None,
            }),
        ])
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Engine(event) => return self.handle_engine(event),
            Message::Num(num, value) => {
                self.dragging = Some(num);
                if let Some(state) = &mut self.state {
                    num.set(&mut state.settings, value);
                }
                if !num.needs_restart() {
                    self.send_setting(num.key(), json!(value));
                }
            }
            Message::NumReleased(num) => {
                self.dragging = None;
                if num.needs_restart()
                    && let Some(value) = self.state.as_ref().map(|s| num.get(&s.settings))
                {
                    self.send_setting(num.key(), json!(value));
                }
            }
            Message::EditValue(num) => {
                let Some(state) = &self.state else {
                    return Task::none();
                };
                self.editing = Some((num, num.edit_text(num.get(&state.settings))));
                let id = VALUE_INPUT_ID.clone();
                return Task::batch([
                    widget::text_input::focus(id.clone()),
                    widget::text_input::select_all(id),
                ]);
            }
            Message::EditInput(text) => {
                if let Some((_, current)) = &mut self.editing {
                    *current = text;
                }
            }
            Message::EditSubmit => {
                let Some((num, text)) = self.editing.take() else {
                    return Task::none();
                };
                match num.parse(&text) {
                    Some(value) => self.apply_typed(num, value),
                    // Keep the field open so the typo can be fixed.
                    None => self.editing = Some((num, text)),
                }
            }
            Message::EditBlur => {
                // Escape also unfocuses; its key event follows within a frame
                // and cancels this before it applies.
                self.blur_pending = self.editing.clone();
                return cosmic::task::future(async {
                    tokio::time::sleep(BLUR_APPLY_DELAY).await;
                    Message::EditBlurApply
                });
            }
            Message::EditBlurApply => {
                let Some((num, text)) = self.blur_pending.take() else {
                    return Task::none();
                };
                // Another value may have been opened meanwhile: close only this one.
                if self.editing.as_ref().is_some_and(|(editing, _)| *editing == num) {
                    self.editing = None;
                }
                // Invalid text is dropped: clicking away leaves the value as it was.
                if let Some(value) = num.parse(&text) {
                    self.apply_typed(num, value);
                }
            }
            Message::EscapePressed => {
                self.blur_pending = None;
                self.editing = None;
                self.confirm_delete = None;
                self.renaming = None;
            }
            Message::Toggle(toggle, value) => self.send_setting(toggle.key(), json!(value)),
            Message::SelectModel(index) => {
                if let Some(name) = self.model_names.get(index).cloned() {
                    self.send_setting("model_name", json!(name));
                }
            }
            Message::SelectGpu(index) => {
                let id = self
                    .state
                    .as_ref()
                    .and_then(|state| state.gpus.get(index))
                    .map(|gpu| gpu.id.clone());
                if let Some(id) = id {
                    self.send_setting("gpu", json!(id));
                }
            }
            Message::SelectInputSource(index) => {
                self.send_setting("input_source", json!(INPUT_SOURCES[index]));
            }
            Message::SelectInput(index) => {
                if let Some(label) = self.input_labels.get(index).cloned() {
                    self.send_setting("input_device", json!(label));
                }
            }
            Message::SelectOutput(index) => {
                if let Some(label) = self.output_labels.get(index).cloned() {
                    self.send_setting("output_device", json!(label));
                }
            }
            Message::SelectMonitor(index) => {
                // Index 0 is "Disabled"; the rest mirror the output list.
                let value = match index {
                    0 => Value::Null,
                    _ => self.output_labels.get(index - 1).cloned().into(),
                };
                self.send_setting("monitor_device", value);
            }
            Message::SelectF0(index) => self.send_setting("f0method", json!(F0_METHODS[index])),
            Message::SelectLoudnessTarget(index) => {
                // The next meters event carries the reading for the new target.
                self.meters.loudness = None;
                self.send_setting("loudness_target", json!(LOUDNESS_TARGETS[index]));
            }
            Message::SelectHoldDetector(index) => {
                self.send_setting("hold_detector", json!(HOLD_DETECTORS[index]));
            }
            Message::SelectRecordingMode(index) => {
                self.send_setting("recording_mode", json!(RECORDING_MODES[index]));
            }
            Message::Run(function) => {
                let running_same = self
                    .state
                    .as_ref()
                    .is_some_and(|s| s.running && s.function == function.id());
                if running_same {
                    self.send("stop", json!({}));
                } else {
                    self.starting = true;
                    self.status = fl!("status-preparing");
                    if let Some(id) = self.send("start", json!({"function": function.id()})) {
                        self.pending.insert(id, Pending::Start);
                    }
                }
            }
            Message::Reset(group) => {
                self.send("reset_settings", json!({"group": group}));
            }
            Message::ReloadModels => {
                self.send("reload_models", json!({}));
            }
            Message::ReloadDevices => {
                self.send("reload_devices", json!({}));
            }
            Message::PickAudioFile => {
                return pick(PickTarget::AudioFile, fl!("pick-audio-file"));
            }
            Message::PickRecordingFolder => {
                return pick(PickTarget::RecordingFolder, fl!("pick-recording-folder"));
            }
            Message::Picked(_, None) => {}
            Message::Picked(PickTarget::AudioFile, Some(path)) => {
                self.send("file_select", json!({"path": path}));
            }
            Message::Picked(PickTarget::RecordingFolder, Some(path)) => {
                self.send_setting("recording_folder", json!(path));
            }
            Message::FilePlay => {
                self.send("file_play", json!({}));
            }
            Message::FilePause => {
                self.send("file_pause", json!({}));
            }
            Message::FileStop => {
                self.send("file_stop", json!({}));
            }
            Message::FileSeek(seconds) => self.seek_preview = Some(seconds),
            Message::FileSeekReleased => {
                if let Some(seconds) = self.seek_preview.take() {
                    self.send("file_seek", json!({"seconds": seconds}));
                }
            }
            Message::ToggleRecording => {
                let recording = self.state.as_ref().is_some_and(|s| s.recording);
                if recording {
                    if let Some(id) = self.send("record_stop", json!({})) {
                        self.pending.insert(id, Pending::RecordStop);
                    }
                } else {
                    self.send("record_start", json!({}));
                }
            }
            Message::OpenRecordingFolder => {
                if let Some(state) = &self.state {
                    let folder = state.settings.recording_folder.clone();
                    let _ = std::fs::create_dir_all(&folder);
                    if let Err(error) = std::process::Command::new("xdg-open").arg(&folder).spawn()
                    {
                        return self.toast(error.to_string());
                    }
                }
            }
            Message::SaveLog => {
                if let Some(id) = self.send("save_log", json!({})) {
                    self.pending.insert(id, Pending::SaveLog);
                }
            }
            Message::ClearLog => {
                self.log.clear();
                self.send("clear_log", json!({}));
            }
            Message::CloseToast(id) => self.toasts.remove(id),
            Message::DragHover(hovering) => self.drag_hover = hovering,
            Message::FilesDropped(files) => {
                self.drag_hover = false;
                let paths = files.map(|files| files.0).unwrap_or_default();
                if paths.is_empty() {
                    return Task::none();
                }
                if self.engine.is_none() || self.state.is_none() {
                    return self.toast(fl!("error-not-ready"));
                }
                // Model files are imported; anything else is a voice clip
                // (or folder of clips) for the selected model's pitch.
                if !paths.iter().any(|path| is_model_file(path)) {
                    self.send("analyze_source_pitch", json!({"paths": paths}));
                    return Task::none();
                }
                self.status = fl!("status-importing-model");
                if let Some(id) = self.send("import_model", json!({"paths": paths})) {
                    self.pending.insert(id, Pending::Import);
                }
            }
            Message::PickPitchClips => return pick_clips(fl!("pick-pitch-clips")),
            Message::PitchClipsPicked(paths) => {
                if !paths.is_empty() {
                    self.send("analyze_source_pitch", json!({"paths": paths}));
                }
            }
            Message::ApplyRecommendedPitch => {
                let recommended = self.state.as_ref().and_then(recommended_pitch);
                if let (Some(value), Some(state)) = (recommended, &mut self.state) {
                    state.settings.pitch = value;
                    self.send_setting(Num::Pitch.key(), json!(value));
                }
            }
            Message::ResetVoicePitch => {
                self.send("reset_voice_pitch", json!({}));
            }
            Message::OpenLibrary => {
                let library = self
                    .nav
                    .iter()
                    .find(|&id| self.nav.data::<Page>(id) == Some(&Page::Library));
                if let Some(id) = library {
                    return self.on_nav_select(id);
                }
            }
            Message::LibrarySearch(text) => self.library_query = text,
            Message::LibrarySort(index) => self.library_sort = index.min(LIBRARY_SORTS.len() - 1),
            Message::SelectModelByName(name) => self.send_setting("model_name", json!(name)),
            Message::OpenModelFolder(name) => {
                let Some(root) = engine::project_dir() else {
                    return Task::none();
                };
                let folder = root.join("models").join(&name);
                if let Err(error) = std::process::Command::new("xdg-open").arg(&folder).spawn() {
                    return self.toast(error.to_string());
                }
            }
            Message::AskDeleteModel(name) => self.confirm_delete = Some(name),
            Message::CancelDeleteModel => self.confirm_delete = None,
            Message::AskRenameModel(name) => {
                self.renaming = Some((name.clone(), name));
                let id = RENAME_INPUT_ID.clone();
                return Task::batch([
                    widget::text_input::focus(id.clone()),
                    widget::text_input::select_all(id),
                ]);
            }
            Message::RenameInput(text) => {
                if let Some((_, current)) = &mut self.renaming {
                    *current = text;
                }
            }
            Message::CancelRenameModel => self.renaming = None,
            Message::ConfirmRenameModel => {
                if let Some((name, new_name)) = self.renaming.take() {
                    let new_name = new_name.trim().to_string();
                    if !new_name.is_empty() && new_name != name {
                        self.send("rename_model", json!({"name": name, "new_name": new_name}));
                    }
                }
            }
            Message::ConfirmDeleteModel => {
                if let Some(name) = self.confirm_delete.take() {
                    self.send("delete_model", json!({"name": name}));
                }
            }
            Message::ApplyLoudness => {
                let adjust = self.meters.loudness.as_ref().map_or(0.0, |l| l.adjust_db);
                if adjust != 0.0
                    && let Some(state) = &mut self.state
                {
                    let (min, max, _) = Num::OutputGain.range();
                    let value = (state.settings.output_gain_db + adjust).clamp(min, max);
                    state.settings.output_gain_db = value;
                    self.send_setting(Num::OutputGain.key(), json!(value));
                    // The next meters event (within 50 ms) brings the reading
                    // at the new gain; until then nothing can be applied twice.
                    self.meters.loudness = None;
                }
            }
        }
        Task::none()
    }

    fn view(&self) -> Element<'_, Message> {
        let page = self
            .nav
            .active_data::<Page>()
            .copied()
            .unwrap_or(Page::Model);
        let mut content = cosmic::iced::widget::Stack::new().push(pages::view(self, page));
        if self.drag_hover {
            content = content.push(pages::drop_overlay());
        }
        // Dropping .pth files anywhere on the window imports them as models.
        let drop_target = widget::dnd_destination::dnd_destination_for_data::<DroppedFiles, _>(
            content,
            |files, _action| Message::FilesDropped(files),
        )
        .on_enter(|_, _, _| Message::DragHover(true))
        .on_leave(|| Message::DragHover(false));
        widget::toaster(&self.toasts, drop_target)
    }

    fn dialog(&self) -> Option<Element<'_, Message>> {
        if let Some((name, new_name)) = &self.renaming {
            let trimmed = new_name.trim();
            let can_rename = !trimmed.is_empty() && trimmed != name;
            return Some(
                widget::dialog()
                    .title(fl!("rename-model-title"))
                    .body(fl!("rename-model-body"))
                    .control(
                        widget::text_input("", new_name.as_str())
                            .id(RENAME_INPUT_ID.clone())
                            .on_input(Message::RenameInput)
                            .on_submit(|_| Message::ConfirmRenameModel),
                    )
                    .primary_action(
                        widget::button::suggested(fl!("rename-model-confirm"))
                            .on_press_maybe(can_rename.then_some(Message::ConfirmRenameModel)),
                    )
                    .secondary_action(
                        widget::button::standard(fl!("cancel"))
                            .on_press(Message::CancelRenameModel),
                    )
                    .into(),
            );
        }
        let name = self.confirm_delete.as_ref()?;
        Some(
            widget::dialog()
                .title(fl!("delete-model-title"))
                .icon(widget::icon::from_name("user-trash-symbolic").size(48))
                .body(fl!("delete-model-body", name = name.clone()))
                .primary_action(
                    widget::button::destructive(fl!("delete-model-confirm"))
                        .on_press(Message::ConfirmDeleteModel),
                )
                .secondary_action(
                    widget::button::standard(fl!("cancel")).on_press(Message::CancelDeleteModel),
                )
                .into(),
        )
    }

    fn footer(&self) -> Option<Element<'_, Message>> {
        self.state
            .as_ref()
            .map(|state| pages::control_bar(self, state))
    }
}

impl App {
    fn update_title(&mut self) -> Task<Message> {
        let page = self
            .nav
            .text(self.nav.active())
            .unwrap_or_default()
            .to_string();
        self.set_header_title(page.clone());
        let title = format!("{page} — {}", fl!("app-title"));
        match self.core.main_window_id() {
            Some(id) => self.set_window_title(title, id),
            None => Task::none(),
        }
    }

    fn send(&mut self, cmd: &str, args: Value) -> Option<u64> {
        let handle = self.engine.as_ref()?;
        Some(handle.send(cmd, args))
    }

    fn send_setting(&mut self, key: &str, value: Value) {
        self.send("update_settings", json!({"settings": {key: value}}));
    }

    /// Apply a typed slider value (Enter, or clicking away).
    fn apply_typed(&mut self, num: Num, value: f32) {
        if let Some(state) = &mut self.state {
            num.set(&mut state.settings, value);
        }
        self.send_setting(num.key(), json!(value));
    }

    fn toast(&mut self, text: String) -> Task<Message> {
        self.toasts
            .push(toaster::Toast::new(text))
            .map(cosmic::Action::App)
    }

    fn apply_state(&mut self, mut state: State) {
        if let (Some(num), Some(old)) = (self.dragging, &self.state) {
            num.set(&mut state.settings, num.get(&old.settings));
        }
        self.model_names = state.models.iter().map(|m| m.name.clone()).collect();
        self.gpu_labels = state
            .gpus
            .iter()
            .map(|gpu| gpu.label.clone().unwrap_or_else(|| fl!("gpu-automatic")))
            .collect();
        self.input_labels = state
            .devices
            .inputs
            .iter()
            .map(|d| d.label.clone())
            .collect();
        self.output_labels = state
            .devices
            .outputs
            .iter()
            .map(|d| d.label.clone())
            .collect();
        self.monitor_labels = std::iter::once(fl!("monitor-disabled"))
            .chain(self.output_labels.iter().cloned())
            .collect();
        if !state.running {
            self.meters = Meters {
                file_position: self.meters.file_position.filter(|_| state.file.playing),
                ..Meters::default()
            };
        }
        self.state = Some(state);
    }

    fn handle_engine(&mut self, event: engine::Event) -> Task<Message> {
        use engine::Event;
        match event {
            Event::Connected(handle) => {
                self.engine = Some(handle);
                self.connection = Connection::Starting;
                self.pending.clear();
                self.starting = false;
            }
            Event::Disconnected(reason) => {
                self.engine = None;
                self.state = None;
                self.connection = Connection::Failed(reason);
            }
            Event::Hello { state, init_error } => {
                if let Some(error) = init_error {
                    self.connection = Connection::Failed(error);
                } else if let Some(state) = state {
                    self.connection = Connection::Ready;
                    self.apply_state(*state);
                }
                // The log history arrives as a plain response.
                self.log_request = self.send("get_log", json!({}));
            }
            Event::Ready(state) => {
                self.connection = Connection::Ready;
                self.apply_state(*state);
            }
            Event::Fatal(message) => self.connection = Connection::Failed(message),
            Event::State(state) => self.apply_state(*state),
            Event::Meters(meters) => {
                // The live voice measurement rides along with the meters.
                if let (Some(voice), Some(state)) = (meters.voice_pitch, &mut self.state) {
                    state.voice_pitch = voice;
                }
                self.meters = meters;
            }
            Event::Status { code, data } => {
                let text = status_text(&code, &data);
                if !text.is_empty() {
                    self.status = text;
                }
                if matches!(
                    code.as_str(),
                    "recording_saved"
                        | "stream_stopped"
                        | "pitch_analyzed"
                        | "model_deleted"
                        | "model_renamed"
                ) {
                    return self.toast(self.status.clone());
                }
            }
            Event::Error(error) => {
                let text = error_text(&error);
                self.status = text.clone();
                return self.toast(text);
            }
            Event::Log(text) => self.append_log(&text),
            Event::Response { id, result } => return self.handle_response(id, result),
        }
        Task::none()
    }

    fn handle_response(&mut self, id: u64, result: Result<Value, ErrorInfo>) -> Task<Message> {
        if self.log_request == Some(id) {
            self.log_request = None;
            if let Ok(value) = &result {
                let history = value["text"].as_str().unwrap_or_default().to_string();
                // Keep lines that arrived as events after the request.
                let recent = std::mem::take(&mut self.log);
                self.log = history;
                if !self.log.ends_with(&recent) {
                    self.append_log(&recent);
                }
            }
            return Task::none();
        }
        let pending = self.pending.remove(&id);
        if matches!(pending, Some(Pending::Start)) {
            self.starting = false;
        }
        match (pending, result) {
            (_, Err(error)) => {
                let text = error_text(&error);
                self.status = text.clone();
                self.toast(text)
            }
            (Some(Pending::Import), Ok(value)) => {
                let names: Vec<&str> = value["models"]
                    .as_array()
                    .map(|names| names.iter().filter_map(Value::as_str).collect())
                    .unwrap_or_default();
                let text = fl!("model-imported", names = names.join(", "));
                self.status = text.clone();
                self.toast(text)
            }
            (Some(Pending::SaveLog), Ok(value)) => {
                let path = value["path"].as_str().unwrap_or_default().to_string();
                self.toast(fl!("log-saved", path = path))
            }
            (Some(Pending::RecordStop), Ok(_)) | (_, Ok(_)) => Task::none(),
        }
    }

    fn append_log(&mut self, text: &str) {
        self.log.push_str(text);
        if self.log.len() > MAX_LOG_CHARS {
            let mut cut = self.log.len() - MAX_LOG_CHARS;
            while !self.log.is_char_boundary(cut) {
                cut += 1;
            }
            self.log.drain(..cut);
        }
    }
}

fn pick_clips(title: String) -> Task<Message> {
    cosmic::task::future(async move {
        let dialog = file_chooser::open::Dialog::new().title(title);
        let paths = dialog
            .open_files()
            .await
            .map(|response| {
                response
                    .urls()
                    .iter()
                    .filter_map(|url| url.to_file_path().ok())
                    .map(|path| path.to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        Message::PitchClipsPicked(paths)
    })
}

/// `.pth` models and their `.index` files are imported; other drops are clips.
fn is_model_file(path: &str) -> bool {
    let lower = path.to_lowercase();
    lower.ends_with(".pth") || lower.ends_with(".index")
}

/// Semitones that move the user's median pitch onto the model speaker's,
/// in the slider's 0.1 steps (see tools/pitch_match.py).
pub fn recommended_shift(voice_hz: f32, source_hz: f32) -> Option<f32> {
    if !(voice_hz > 0.0 && source_hz > 0.0) {
        return None;
    }
    let shift = 12.0 * (source_hz / voice_hz).log2();
    Some(((shift * 10.0).round() / 10.0).clamp(-24.0, 24.0))
}

pub fn recommended_pitch(state: &State) -> Option<f32> {
    recommended_shift(state.voice_pitch.median_hz?, state.settings.source_pitch_hz)
}

fn pick(target: PickTarget, title: String) -> Task<Message> {
    cosmic::task::future(async move {
        let dialog = file_chooser::open::Dialog::new().title(title);
        let response = match target {
            PickTarget::AudioFile => dialog.open_file().await,
            PickTarget::RecordingFolder => dialog.open_folder().await,
        };
        let path = response
            .ok()
            .and_then(|response| response.url().to_file_path().ok())
            .map(|path| path.to_string_lossy().into_owned());
        Message::Picked(target, path)
    })
}

/// Translate an engine status code.
pub fn status_text(code: &str, data: &Value) -> String {
    let text = |key: &str| {
        data.get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    match code {
        "preparing" => fl!("status-preparing"),
        "loading_model" => fl!("status-loading-model"),
        "preparing_inference" => fl!("status-preparing-inference"),
        "starting_audio" => fl!("status-starting-audio"),
        "conversion_started" => fl!("status-conversion-started"),
        "passthrough_started" => fl!("status-passthrough-started"),
        "conversion_stopped" => fl!("status-conversion-stopped"),
        "passthrough_stopped" => fl!("status-passthrough-stopped"),
        "settings_changed" => fl!("status-settings-changed"),
        "stream_stopped" => fl!("status-stream-stopped"),
        "file_selected" => fl!("status-file-selected", name = text("name")),
        "playing" => fl!("status-playing"),
        "paused" => fl!("status-paused"),
        "playback_stopped" => fl!("status-playback-stopped"),
        "recording_started" => fl!("status-recording-started"),
        "importing_model" => fl!("status-importing-model"),
        "model_deleted" => fl!("status-model-deleted", name = text("name")),
        "model_renamed" => fl!(
            "status-model-renamed",
            name = text("name"),
            new_name = text("new_name")
        ),
        "analyzing_pitch" => fl!("status-analyzing-pitch", name = text("name")),
        "pitch_analyzed" => fl!(
            "status-pitch-analyzed",
            name = text("name"),
            hz = format!("{:.1}", data.get("median_hz").and_then(Value::as_f64).unwrap_or(0.0)),
            minutes = format!("{:.1}", data.get("seconds").and_then(Value::as_f64).unwrap_or(0.0) / 60.0)
        ),
        // The toast from the import response names the model.
        "model_imported" => String::new(),
        "recording_saved" => {
            let paths: Vec<&str> = data
                .get("paths")
                .and_then(Value::as_array)
                .map(|paths| paths.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            fl!("status-recording-saved", paths = paths.join("\n"))
        }
        other => other.to_string(),
    }
}

/// Translate an engine error; unknown codes fall back to its message.
pub fn error_text(error: &ErrorInfo) -> String {
    match error.code.as_str() {
        "no_model" => fl!("error-no-model"),
        "model_file_missing" => fl!("error-model-file-missing"),
        "index_missing" => fl!("error-index-missing"),
        "no_audio_file" => fl!("error-no-audio-file"),
        "ffmpeg_missing" => fl!("error-ffmpeg-missing"),
        "record_requires_running" => fl!("error-record-requires-running"),
        "no_common_samplerate" => fl!("error-no-common-samplerate"),
        "audio_start_failed" => fl!("error-audio-start-failed", detail = error.message.clone()),
        "monitor_failed" => fl!("error-monitor-failed", detail = error.message.clone()),
        "not_ready" => fl!("error-not-ready"),
        "import_no_model_file" => fl!("error-import-no-model-file"),
        "import_failed" => fl!("error-import-failed", detail = error.message.clone()),
        "assets_missing" => {
            let paths: Vec<&str> = error.details["paths"]
                .as_array()
                .map(|paths| paths.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            fl!("error-assets-missing", paths = paths.join(", "))
        }
        "model_load_failed" => fl!("error-model-load-failed", detail = error.message.clone()),
        "no_voice_found" => fl!("error-no-voice-found"),
        "model_in_use" => fl!("error-model-in-use"),
        "delete_failed" => fl!("error-delete-failed", detail = error.message.clone()),
        "model_name_taken" => fl!("error-model-name-taken"),
        "bad_model_name" => fl!("error-bad-model-name"),
        "rename_failed" => fl!("error-rename-failed", detail = error.message.clone()),
        "pitch_analysis_busy" => fl!("error-pitch-analysis-busy"),
        "pitch_analysis_failed" => {
            fl!("error-pitch-analysis-failed", detail = error.message.clone())
        }
        _ => error.message.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::{Num, is_model_file, recommended_shift};

    #[test]
    fn recommends_the_shift_onto_the_source_median_in_tenths() {
        // Matches tools/pitch_match.py (tests/test_pitch_match.py).
        assert_eq!(recommended_shift(114.0, 212.0), Some(10.7));
        assert_eq!(recommended_shift(114.0, 220.0), Some(11.4));
        assert_eq!(recommended_shift(220.0, 110.0), Some(-12.0));
        assert_eq!(recommended_shift(100.0, 5000.0), Some(24.0));
        assert_eq!(recommended_shift(114.0, 0.0), None);
        assert_eq!(recommended_shift(0.0, 212.0), None);
    }

    #[test]
    fn model_files_are_told_apart_from_voice_clips() {
        assert!(is_model_file("/a/Voice.PTH"));
        assert!(is_model_file("/a/added_Voice.index"));
        assert!(!is_model_file("/a/dataset.wav"));
        assert!(!is_model_file("/a/dataset-folder"));
    }

    #[test]
    fn parses_typed_values_with_units_signs_and_commas() {
        assert_eq!(Num::Pitch.parse("+11.5"), Some(11.5));
        assert_eq!(Num::Pitch.parse(" -3,25 "), Some(-3.25));
        assert_eq!(Num::OutputGain.parse("-6 dB"), Some(-6.0));
        assert_eq!(Num::BlockTime.parse("0.3s"), Some(0.3));
        assert_eq!(Num::FileVolume.parse("50%"), Some(0.5));
        assert_eq!(Num::NoiseGate.parse("Off"), Some(-60.0));
    }

    #[test]
    fn clamps_to_the_slider_range_and_rejects_garbage() {
        assert_eq!(Num::Pitch.parse("40"), Some(24.0));
        assert_eq!(Num::IndexRate.parse("-1"), Some(0.0));
        assert_eq!(Num::Protect.parse("Off"), Some(0.5));
        assert_eq!(Num::Protect.parse("0.9"), Some(0.5));
        assert_eq!(Num::Pitch.parse("high"), None);
        assert_eq!(Num::Pitch.parse(""), None);
        assert_eq!(Num::Pitch.parse("NaN"), None);
    }

    #[test]
    fn edit_text_is_plain_and_round_trips() {
        assert_eq!(Num::Pitch.edit_text(11.5), "11.5");
        assert_eq!(Num::Pitch.edit_text(12.0), "12");
        assert_eq!(Num::FileVolume.edit_text(0.25), "25");
        for (num, value) in [
            (Num::Pitch, -7.3),
            (Num::Formant, 0.05),
            (Num::FileVolume, 0.8),
        ] {
            let parsed = num.parse(&num.edit_text(value)).unwrap();
            assert!((parsed - value).abs() < 1e-4, "{num:?} {value} -> {parsed}");
        }
    }
}
