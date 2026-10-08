//! Application state and message handling.

use std::collections::HashMap;

use cosmic::app::{Core, Task};
use cosmic::dialog::file_chooser;
use cosmic::iced::Subscription;
use cosmic::prelude::*;
use cosmic::widget::{self, nav_bar, toaster};
use serde_json::{Value, json};

use crate::engine::{self, ErrorInfo, Handle, Meters, Settings, State};
use crate::fl;
use crate::pages;

pub const APP_ID: &str = "io.github.rvc_realtime.RvcRealtime";
const MAX_LOG_CHARS: usize = 40_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Page {
    Model,
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
    RmsMix,
    BlockTime,
    Crossfade,
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
            Num::RmsMix => "rms_mix_rate",
            Num::BlockTime => "block_time",
            Num::Crossfade => "crossfade_time",
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
            Num::BlockTime => (0.02, 1.5, 0.01),
            Num::Crossfade => (0.01, 0.15, 0.01),
            Num::Extra => (0.05, 5.0, 0.01),
            Num::FileVolume => (0.0, 1.0, 0.01),
        }
    }

    /// Changing these stops a running stream, so they are sent on release.
    pub fn needs_restart(self) -> bool {
        matches!(self, Num::BlockTime | Num::Crossfade | Num::Extra)
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
            Num::RmsMix => settings.rms_mix_rate,
            Num::BlockTime => settings.block_time,
            Num::Crossfade => settings.crossfade_time,
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
            Num::RmsMix => &mut settings.rms_mix_rate,
            Num::BlockTime => &mut settings.block_time,
            Num::Crossfade => &mut settings.crossfade_time,
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
}

impl Toggle {
    fn key(self) -> &'static str {
        match self {
            Toggle::InputDenoise => "input_denoise",
            Toggle::OutputDenoise => "output_denoise",
        }
    }
}

pub const F0_METHODS: [&str; 3] = ["rmvpe", "fcpe", "pm"];
pub const RECORDING_MODES: [&str; 3] = ["separate", "mix", "stereo"];
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
    RecordStop,
    SaveLog,
}

#[derive(Clone, Debug)]
pub enum Message {
    Engine(engine::Event),
    Num(Num, f32),
    NumReleased(Num),
    Toggle(Toggle, bool),
    SelectModel(usize),
    SelectGpu(usize),
    SelectInputSource(usize),
    SelectInput(usize),
    SelectOutput(usize),
    SelectMonitor(usize),
    SelectF0(usize),
    SelectRecordingMode(usize),
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
    pub starting: bool,
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
}

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
            (Page::Audio, fl!("page-audio"), "audio-card-symbolic"),
            (Page::Performance, fl!("page-performance"), "speedometer-symbolic"),
            (Page::Recording, fl!("page-recording"), "media-record-symbolic"),
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
            starting: false,
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
            recording_mode_labels: vec![
                fl!("recording-separate"),
                fl!("recording-mix"),
                fl!("recording-stereo"),
            ],
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
        Subscription::run(engine::connection).map(Message::Engine)
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
                    if let Err(error) = std::process::Command::new("xdg-open").arg(&folder).spawn() {
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
        }
        Task::none()
    }

    fn view(&self) -> Element<'_, Message> {
        let page = self.nav.active_data::<Page>().copied().unwrap_or(Page::Model);
        let content = pages::view(self, page);
        widget::toaster(&self.toasts, content)
    }

    fn footer(&self) -> Option<Element<'_, Message>> {
        self.state.as_ref().map(|state| pages::control_bar(self, state))
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

    fn toast(&mut self, text: String) -> Task<Message> {
        self.toasts.push(toaster::Toast::new(text)).map(cosmic::Action::App)
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
        self.input_labels = state.devices.inputs.iter().map(|d| d.label.clone()).collect();
        self.output_labels = state.devices.outputs.iter().map(|d| d.label.clone()).collect();
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
            Event::Meters(meters) => self.meters = meters,
            Event::Status { code, data } => {
                self.status = status_text(&code, &data);
                if code == "recording_saved" || code == "stream_stopped" {
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
    let text = |key: &str| data.get(key).and_then(Value::as_str).unwrap_or_default().to_string();
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
        _ => error.message.clone(),
    }
}
