//! Connection to the Python engine (`python -m engine.server`).
//!
//! The protocol is documented in `docs/engine-protocol.md`: newline-delimited
//! JSON over a Unix socket.  [`connection`] is an iced subscription that
//! starts the engine when nothing is listening, reconnects when the socket
//! drops, and turns incoming lines into [`Event`]s.  Commands go out through
//! the [`Handle`] delivered in [`Event::Connected`].

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use cosmic::iced::futures::{SinkExt, Stream};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;
use tokio::sync::mpsc;

/// Seconds the engine keeps running after its last client disconnects.
const ENGINE_IDLE_EXIT_SECONDS: &str = "10";
const SPAWN_CONNECT_TIMEOUT: Duration = Duration::from_secs(20);
const RECONNECT_DELAY: Duration = Duration::from_secs(1);

#[derive(Clone, Debug, Deserialize)]
pub struct State {
    pub running: bool,
    pub function: String,
    pub settings: Settings,
    pub models: Vec<Model>,
    pub devices: Devices,
    pub gpus: Vec<Gpu>,
    pub samplerate: Option<u32>,
    pub delay_ms: Option<i64>,
    pub recording: bool,
    pub file: FileState,
    pub ffmpeg: bool,
    /// Asset files the engine needs but cannot find (see README).
    #[serde(default)]
    pub missing_assets: Vec<String>,
    /// Whether input noise reduction can use RNNoise (otherwise spectral gating).
    #[serde(default)]
    pub rnnoise: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Settings {
    pub model_name: String,
    pub gpu: String,
    pub input_source: String,
    pub input_device: String,
    pub output_device: String,
    pub monitor_device: Option<String>,
    pub block_time: f32,
    pub extra_time: f32,
    pub input_denoise: bool,
    pub output_denoise: bool,
    pub hold_context: bool,
    #[serde(default = "default_hold_detector")]
    pub hold_detector: String,
    pub rms_mix_rate: f32,
    pub f0method: String,
    pub recording_folder: String,
    pub recording_mode: String,
    pub file_input_volume: f32,
    pub input_gain_db: f32,
    pub output_gain_db: f32,
    pub monitor_gain_db: f32,
    pub noise_gate_db: f32,
    pub pitch: f32,
    pub formant: f32,
    pub index_rate: f32,
}

fn default_hold_detector() -> String {
    "level".to_string()
}

#[derive(Clone, Debug, Deserialize)]
pub struct Model {
    pub name: String,
    pub model_file: String,
    pub index_file: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Devices {
    pub inputs: Vec<Device>,
    pub outputs: Vec<Device>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Device {
    pub label: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Gpu {
    pub id: String,
    pub label: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FileState {
    pub path: Option<String>,
    pub duration: Option<f64>,
    pub playing: bool,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Meters {
    pub input: f32,
    pub output: f32,
    pub monitor: f32,
    pub infer_ms: Option<u32>,
    pub recording_seconds: Option<f64>,
    pub file_position: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ErrorInfo {
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub details: Value,
}

#[derive(Clone, Debug)]
pub enum Event {
    /// Connected; commands can be sent through the handle.
    Connected(Handle),
    /// The connection dropped or the engine could not be started.
    Disconnected(String),
    /// Sent once per connection; `state` is `None` while the engine starts.
    Hello {
        state: Option<Box<State>>,
        init_error: Option<String>,
    },
    Ready(Box<State>),
    Fatal(String),
    State(Box<State>),
    Meters(Meters),
    Status {
        code: String,
        data: Value,
    },
    Error(ErrorInfo),
    Log(String),
    Response {
        id: u64,
        result: Result<Value, ErrorInfo>,
    },
}

/// Sends commands to the engine; cheap to clone.
#[derive(Clone, Debug)]
pub struct Handle {
    sender: mpsc::UnboundedSender<String>,
    next_id: Arc<AtomicU64>,
}

impl Handle {
    /// Queue one command; returns its request id for matching the response.
    pub fn send(&self, cmd: &str, args: Value) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let line = json!({"id": id, "cmd": cmd, "args": args}).to_string();
        // A closed channel means the connection is gone; the subscription
        // reports that separately as Event::Disconnected.
        let _ = self.sender.send(line);
        id
    }
}

pub fn socket_path() -> PathBuf {
    let runtime_dir = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            // Matches engine.protocol.default_socket_path().
            let uid = current_uid();
            PathBuf::from(format!("/tmp/rvc-realtime-{uid}"))
        });
    runtime_dir.join("rvc-realtime.sock")
}

fn current_uid() -> u32 {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata("/proc/self")
        .map(|meta| meta.uid())
        .unwrap_or(0)
}

/// The repository root: `RVC_PROJECT_DIR`, or the first ancestor of the
/// executable or working directory that contains `engine/server.py`.
pub fn project_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("RVC_PROJECT_DIR") {
        return Some(PathBuf::from(dir));
    }
    let starts = [std::env::current_exe().ok(), std::env::current_dir().ok()];
    starts.into_iter().flatten().find_map(|start| {
        start
            .ancestors()
            .find(|dir| dir.join("engine").join("server.py").is_file())
            .map(Path::to_path_buf)
    })
}

fn spawn_engine(socket: &Path) -> Result<(), String> {
    let root = project_dir().ok_or_else(|| {
        "Could not find the project folder (engine/server.py). Set RVC_PROJECT_DIR.".to_string()
    })?;
    let python = root.join(".venv").join("bin").join("python");
    if !python.is_file() {
        return Err(format!(
            "Python environment not found: {}",
            python.display()
        ));
    }
    tokio::process::Command::new(&python)
        .args([
            "-m",
            "engine.server",
            "--exit-when-idle",
            ENGINE_IDLE_EXIT_SECONDS,
            "--socket",
        ])
        .arg(socket)
        .current_dir(&root)
        .stdin(std::process::Stdio::null())
        .spawn()
        .map(|_| ())
        .map_err(|error| format!("Could not start the engine ({}): {error}", python.display()))
}

/// Connect to a running engine, or start one and wait for its socket.
async fn connect() -> Result<UnixStream, String> {
    let socket = socket_path();
    if let Ok(stream) = UnixStream::connect(&socket).await {
        return Ok(stream);
    }
    spawn_engine(&socket)?;
    let deadline = tokio::time::Instant::now() + SPAWN_CONNECT_TIMEOUT;
    loop {
        tokio::time::sleep(Duration::from_millis(100)).await;
        match UnixStream::connect(&socket).await {
            Ok(stream) => return Ok(stream),
            Err(error) if tokio::time::Instant::now() >= deadline => {
                return Err(format!(
                    "The engine did not open {}: {error}",
                    socket.display()
                ));
            }
            Err(_) => {}
        }
    }
}

fn parse_line(line: &str) -> Option<Event> {
    let message: Value = serde_json::from_str(line).ok()?;
    if let Some(id) = message.get("id").and_then(Value::as_u64) {
        let result = if message.get("ok").and_then(Value::as_bool) == Some(true) {
            Ok(message.get("result").cloned().unwrap_or(Value::Null))
        } else {
            Err(serde_json::from_value(message.get("error")?.clone()).ok()?)
        };
        return Some(Event::Response { id, result });
    }
    let event = message.get("event")?.as_str()?;
    let data = message.get("data").cloned().unwrap_or(Value::Null);
    let state = |value: Value| serde_json::from_value::<State>(value).ok().map(Box::new);
    Some(match event {
        "hello" => Event::Hello {
            state: data.get("state").cloned().and_then(state),
            init_error: data
                .get("init_error")
                .and_then(Value::as_str)
                .map(String::from),
        },
        "ready" => Event::Ready(state(data)?),
        "state" => Event::State(state(data)?),
        "fatal" => Event::Fatal(data.get("message")?.as_str()?.to_string()),
        "meters" => Event::Meters(serde_json::from_value(data).ok()?),
        "status" => Event::Status {
            code: data.get("code")?.as_str()?.to_string(),
            data,
        },
        "error" => Event::Error(serde_json::from_value(data).ok()?),
        "log" => Event::Log(data.get("text")?.as_str()?.to_string()),
        _ => return None,
    })
}

/// Subscription stream: one engine connection at a time, reconnecting forever.
pub fn connection() -> impl Stream<Item = Event> {
    cosmic::iced::stream::channel(256, async |mut output| {
        let next_id = Arc::new(AtomicU64::new(1));
        loop {
            let stream = match connect().await {
                Ok(stream) => stream,
                Err(error) => {
                    let _ = output.send(Event::Disconnected(error)).await;
                    tokio::time::sleep(RECONNECT_DELAY * 3).await;
                    continue;
                }
            };
            let (read_half, mut write_half) = stream.into_split();
            let (sender, mut receiver) = mpsc::unbounded_channel::<String>();
            let handle = Handle {
                sender,
                next_id: next_id.clone(),
            };
            let _ = output.send(Event::Connected(handle)).await;

            let writer = tokio::spawn(async move {
                while let Some(mut line) = receiver.recv().await {
                    line.push('\n');
                    if write_half.write_all(line.as_bytes()).await.is_err() {
                        break;
                    }
                }
            });
            let mut lines = BufReader::new(read_half).lines();
            let reason = loop {
                match lines.next_line().await {
                    Ok(Some(line)) => {
                        if let Some(event) = parse_line(&line) {
                            let _ = output.send(event).await;
                        }
                    }
                    Ok(None) => break "The engine closed the connection.".to_string(),
                    Err(error) => break error.to_string(),
                }
            };
            writer.abort();
            let _ = output.send(Event::Disconnected(reason)).await;
            tokio::time::sleep(RECONNECT_DELAY).await;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_responses() {
        match parse_line(r#"{"id":3,"ok":true,"result":{"paths":["a.wav"]}}"#) {
            Some(Event::Response {
                id: 3,
                result: Ok(value),
            }) => {
                assert_eq!(value["paths"][0], "a.wav");
            }
            other => panic!("unexpected {other:?}"),
        }
        match parse_line(
            r#"{"id":4,"ok":false,"error":{"code":"no_model","message":"m","details":{}}}"#,
        ) {
            Some(Event::Response {
                id: 4,
                result: Err(error),
            }) => assert_eq!(error.code, "no_model"),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn parses_events_and_ignores_unknown_ones() {
        assert!(matches!(
            parse_line(r#"{"event":"meters","data":{"input":0.5,"output":0,"monitor":0,"infer_ms":null,"recording_seconds":null,"file_position":1.5}}"#),
            Some(Event::Meters(Meters { input, file_position: Some(_), .. })) if input == 0.5
        ));
        assert!(matches!(
            parse_line(
                r#"{"event":"hello","data":{"protocol":1,"ready":false,"init_error":null,"state":null}}"#
            ),
            Some(Event::Hello {
                state: None,
                init_error: None
            })
        ));
        assert!(parse_line(r#"{"event":"future_thing","data":{}}"#).is_none());
        assert!(parse_line("not json").is_none());
    }
}
