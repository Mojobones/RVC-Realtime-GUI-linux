//! Page and control-bar views.

use cosmic::iced::{Alignment, Length};
use cosmic::prelude::*;
use cosmic::widget::{self, settings};

use crate::app::{
    App, Connection, F0_METHODS, Function, HOLD_DETECTORS, INPUT_SOURCES, Message, Num, Page,
    RECORDING_MODES, Toggle, VALUE_INPUT_ID,
};
use crate::engine::State;
use crate::fl;

const SLIDER_WIDTH: f32 = 240.0;
const VALUE_WIDTH: f32 = 72.0;
const METER_WIDTH: f32 = 90.0;

pub fn view(app: &App, page: Page) -> Element<'_, Message> {
    let Some(state) = app
        .state
        .as_ref()
        .filter(|_| matches!(app.connection, Connection::Ready))
    else {
        return connection_view(app);
    };
    let content = match page {
        Page::Model => model_page(app, state),
        Page::Audio => audio_page(app, state),
        Page::Performance => performance_page(app, state),
        Page::Recording => recording_page(app, state),
        Page::Log => return log_page(app),
    };
    widget::scrollable(
        widget::container(content)
            .max_width(900)
            .padding([0, cosmic::theme::spacing().space_s]),
    )
    .width(Length::Fill)
    .into()
}

fn connection_view(app: &App) -> Element<'_, Message> {
    let space = cosmic::theme::spacing();
    let column = match &app.connection {
        Connection::Failed(reason) => widget::column::with_capacity(4)
            .push(widget::icon::from_name("dialog-error-symbolic").size(48))
            .push(widget::text::title3(fl!("engine-unavailable")))
            .push(widget::text::body(fl!("engine-retrying")))
            .push(
                widget::container(widget::text::monotext(reason.as_str()))
                    .class(cosmic::theme::Container::Card)
                    .padding(space.space_s)
                    .max_width(720),
            ),
        _ => widget::column::with_capacity(4)
            .push(widget::progress_bar::circular::Circular::new().size(40.0))
            .push(widget::text::title3(fl!("engine-starting")))
            .push(widget::text::caption(fl!("engine-starting-detail"))),
    };
    widget::container(column.spacing(space.space_s).align_x(Alignment::Center))
        .center(Length::Fill)
        .into()
}

fn page_column<'a>(sections: Vec<Element<'a, Message>>) -> Element<'a, Message> {
    settings::view_column(sections).into()
}

/// A labelled slider with its formatted value.  Clicking the value opens a
/// field to type an exact one (Enter applies, Escape or clicking away cancels).
fn slider_item<'a>(
    app: &'a App,
    title: String,
    num: Num,
    state: &State,
    format: impl Fn(f32) -> String,
) -> Element<'a, Message> {
    let (min, max, step) = num.range();
    let value = num.get(&state.settings);
    let value_widget: Element<_> = match &app.editing {
        Some((editing, text)) if *editing == num => {
            let invalid = num.parse(text).is_none();
            let mut input = widget::text_input::inline_input("", text.as_str())
                .id(VALUE_INPUT_ID.clone())
                .on_input(Message::EditInput)
                .on_submit(|_| Message::EditSubmit)
                .on_unfocus(Message::EditCancel)
                .width(Length::Fixed(VALUE_WIDTH));
            if invalid {
                input = input.error(fl!("value-invalid"));
            }
            input.into()
        }
        _ => widget::tooltip(
            widget::button::custom(
                widget::text::body(format(value))
                    .width(Length::Fill)
                    .align_x(Alignment::End),
            )
            .class(cosmic::theme::Button::Text)
            .padding([2, 4])
            .width(Length::Fixed(VALUE_WIDTH))
            .on_press(Message::EditValue(num)),
            widget::text::caption(fl!("value-click-to-type")),
            widget::tooltip::Position::Top,
        )
        .into(),
    };
    settings::item(
        title,
        widget::row::with_capacity(2)
            .push(
                widget::slider(min..=max, value.clamp(min, max), move |v| {
                    Message::Num(num, v)
                })
                .step(step)
                .on_release(Message::NumReleased(num))
                .width(Length::Fixed(SLIDER_WIDTH)),
            )
            .push(value_widget)
            .spacing(cosmic::theme::spacing().space_xs)
            .align_y(Alignment::Center),
    )
    .into()
}

fn dropdown_item<'a>(
    title: String,
    labels: &'a [String],
    selected: Option<usize>,
    on_select: fn(usize) -> Message,
) -> Element<'a, Message> {
    settings::item(title, widget::dropdown(labels, selected, on_select)).into()
}

fn position_of(labels: &[String], value: &str) -> Option<usize> {
    labels.iter().position(|label| label == value)
}

fn signed(decimals: usize) -> impl Fn(f32) -> String {
    move |value| format!("{value:+.decimals$}")
}

fn decibels(value: f32) -> String {
    format!("{value:+.1} dB")
}

fn seconds(value: f32) -> String {
    format!("{value:.2} s")
}

fn ratio(value: f32) -> String {
    format!("{value:.2}")
}

pub fn mmss(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    format!("{:02}:{:02}", total / 60, total % 60)
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

fn model_page<'a>(app: &'a App, state: &'a State) -> Element<'a, Message> {
    let selected = state.settings.model_name.as_str();
    let files = state
        .models
        .iter()
        .find(|model| model.name == selected)
        .map(|model| {
            fl!(
                "model-files",
                model = model.model_file.clone(),
                index = model.index_file.clone().unwrap_or_else(|| fl!("none"))
            )
        })
        .unwrap_or_default();

    let mut sections = Vec::with_capacity(5);
    if !state.missing_assets.is_empty() {
        sections.push(
            widget::warning(fl!(
                "assets-missing-banner",
                paths = state.missing_assets.join(", ")
            ))
            .into_widget()
            .into(),
        );
    }
    sections.extend([
        settings::section()
            .title(fl!("section-models"))
            .add(if state.models.is_empty() {
                Element::from(widget::text::body(fl!("models-empty")))
            } else {
                dropdown_item(
                    fl!("model"),
                    &app.model_names,
                    position_of(&app.model_names, selected),
                    Message::SelectModel,
                )
            })
            .add(
                widget::row::with_capacity(4)
                    .push(widget::text::caption(files).width(Length::Fill))
                    .push(
                        widget::button::standard(fl!("reload"))
                            .leading_icon(widget::icon::from_name("view-refresh-symbolic"))
                            .on_press(Message::ReloadModels),
                    )
                    .align_y(Alignment::Center),
            )
            .into(),
        settings::section()
            .title(fl!("section-voice"))
            .add(slider_item(app, fl!("pitch"), Num::Pitch, state, signed(1)))
            .add(slider_item(
                app,
                fl!("formant"),
                Num::Formant,
                state,
                signed(2),
            ))
            .add(slider_item(
                app,
                fl!("index-rate"),
                Num::IndexRate,
                state,
                ratio,
            ))
            .into(),
        settings::section()
            .title(fl!("section-levels"))
            .add(slider_item(
                app,
                fl!("input-gain"),
                Num::InputGain,
                state,
                decibels,
            ))
            .add(slider_item(
                app,
                fl!("output-gain"),
                Num::OutputGain,
                state,
                decibels,
            ))
            .add(slider_item(
                app,
                fl!("monitor-gain"),
                Num::MonitorGain,
                state,
                decibels,
            ))
            .add(slider_item(
                app,
                fl!("noise-gate"),
                Num::NoiseGate,
                state,
                |value| {
                    if value <= -60.0 {
                        fl!("off")
                    } else {
                        format!("{value:.0} dB")
                    }
                },
            ))
            .into(),
        reset_row(fl!("reset-model-settings"), "general"),
    ]);
    page_column(sections)
}

fn reset_row<'a>(label: String, group: &'static str) -> Element<'a, Message> {
    widget::row::with_capacity(4)
        .push(widget::space::horizontal())
        .push(widget::button::standard(label).on_press(Message::Reset(group)))
        .into()
}

// ---------------------------------------------------------------------------
// Audio
// ---------------------------------------------------------------------------

fn audio_page<'a>(app: &'a App, state: &'a State) -> Element<'a, Message> {
    let space = cosmic::theme::spacing();
    let settings_ = &state.settings;
    let file_source = settings_.input_source == "file";
    let source_index = INPUT_SOURCES
        .iter()
        .position(|s| *s == settings_.input_source);

    let mut input = settings::section()
        .title(fl!("section-input"))
        .add(dropdown_item(
            fl!("input-source"),
            &app.source_labels,
            source_index,
            Message::SelectInputSource,
        ));
    if file_source && !state.ffmpeg {
        input = input.add(widget::text::body(fl!("error-ffmpeg-missing")));
    }
    if file_source {
        let file_name = state
            .file
            .path
            .as_deref()
            .and_then(|path| std::path::Path::new(path).file_name())
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| fl!("no-file"));
        input = input.add(settings::item(
            fl!("audio-file"),
            widget::row::with_capacity(4)
                .push(widget::text::body(file_name))
                .push(widget::button::standard(fl!("browse")).on_press(Message::PickAudioFile))
                .spacing(space.space_s)
                .align_y(Alignment::Center),
        ));
        input = input.add(file_player(app, state));
        input = input.add(slider_item(
            app,
            fl!("file-volume"),
            Num::FileVolume,
            state,
            |value| format!("{:.0}%", value * 100.0),
        ));
    } else {
        input = input.add(dropdown_item(
            fl!("input-device"),
            &app.input_labels,
            position_of(&app.input_labels, &settings_.input_device),
            Message::SelectInput,
        ));
    }

    let monitor_index = match &settings_.monitor_device {
        None => Some(0),
        Some(label) => position_of(&app.output_labels, label).map(|i| i + 1),
    };
    let output = settings::section()
        .title(fl!("section-output"))
        .add(dropdown_item(
            fl!("output-device"),
            &app.output_labels,
            position_of(&app.output_labels, &settings_.output_device),
            Message::SelectOutput,
        ))
        .add(dropdown_item(
            fl!("monitor-device"),
            &app.monitor_labels,
            monitor_index,
            Message::SelectMonitor,
        ));

    let rate = state
        .samplerate
        .map(|rate| fl!("sample-rate-value", rate = rate))
        .unwrap_or_else(|| fl!("sample-rate-idle"));
    page_column(vec![
        input.into(),
        output.into(),
        widget::row::with_capacity(4)
            .push(
                widget::text::caption(format!("{}\n{}", rate, fl!("jack-hint")))
                    .width(Length::Fill),
            )
            .push(
                widget::button::standard(fl!("reload-devices"))
                    .leading_icon(widget::icon::from_name("view-refresh-symbolic"))
                    .on_press(Message::ReloadDevices),
            )
            .spacing(space.space_s)
            .align_y(Alignment::Center)
            .into(),
    ])
}

fn file_player<'a>(app: &'a App, state: &'a State) -> Element<'a, Message> {
    let space = cosmic::theme::spacing();
    let has_file = state.file.path.is_some();
    let duration = state.file.duration.unwrap_or(0.0).max(1.0) as f32;
    let position = app
        .seek_preview
        .unwrap_or(app.meters.file_position.unwrap_or(0.0) as f32);
    let play_pause = if state.file.playing {
        widget::button::icon(widget::icon::from_name("media-playback-pause-symbolic"))
            .on_press(Message::FilePause)
    } else {
        widget::button::icon(widget::icon::from_name("media-playback-start-symbolic"))
            .on_press_maybe(has_file.then_some(Message::FilePlay))
    };
    widget::row::with_capacity(4)
        .push(play_pause)
        .push(
            widget::button::icon(widget::icon::from_name("media-playback-stop-symbolic"))
                .on_press_maybe(has_file.then_some(Message::FileStop)),
        )
        .push(
            widget::slider(0.0..=duration, position.min(duration), Message::FileSeek)
                .on_release(Message::FileSeekReleased)
                .width(Length::Fill),
        )
        .push(widget::text::body(format!(
            "{} / {}",
            mmss(position as f64),
            mmss(state.file.duration.unwrap_or(0.0))
        )))
        .spacing(space.space_xs)
        .align_y(Alignment::Center)
        .into()
}

// ---------------------------------------------------------------------------
// Performance
// ---------------------------------------------------------------------------

fn performance_page<'a>(app: &'a App, state: &'a State) -> Element<'a, Message> {
    let settings_ = &state.settings;
    let gpu_index = state.gpus.iter().position(|gpu| gpu.id == settings_.gpu);
    let f0_index = F0_METHODS
        .iter()
        .position(|method| *method == settings_.f0method);
    page_column(vec![
        settings::section()
            .title(fl!("section-buffering"))
            .add(slider_item(
                app,
                fl!("chunk"),
                Num::BlockTime,
                state,
                seconds,
            ))
            .add(slider_item(app, fl!("extra"), Num::Extra, state, seconds))
            .into(),
        settings::section()
            .title(fl!("section-inference"))
            .add(dropdown_item(
                fl!("pitch-detector"),
                &app.f0_labels,
                f0_index,
                Message::SelectF0,
            ))
            .add(slider_item(
                app,
                fl!("volume-envelope"),
                Num::RmsMix,
                state,
                ratio,
            ))
            .add(dropdown_item(
                fl!("gpu"),
                &app.gpu_labels,
                gpu_index,
                Message::SelectGpu,
            ))
            .add(
                settings::item::builder(fl!("hold-context"))
                    .description(fl!("hold-context-detail"))
                    .toggler(settings_.hold_context, |value| {
                        Message::Toggle(Toggle::HoldContext, value)
                    }),
            )
            .add(
                settings::item::builder(fl!("hold-detector"))
                    .description(if settings_.hold_detector == "voice" {
                        fl!("hold-detector-voice-detail")
                    } else {
                        fl!("hold-detector-level-detail")
                    })
                    .control(widget::dropdown(
                        &app.hold_detector_labels,
                        HOLD_DETECTORS
                            .iter()
                            .position(|detector| *detector == settings_.hold_detector),
                        Message::SelectHoldDetector,
                    )),
            )
            .into(),
        settings::section()
            .title(fl!("section-noise"))
            .add(
                settings::item::builder(fl!("input-denoise"))
                    .description(if state.rnnoise {
                        fl!("input-denoise-rnnoise")
                    } else {
                        fl!("input-denoise-spectral")
                    })
                    .toggler(settings_.input_denoise, |value| {
                        Message::Toggle(Toggle::InputDenoise, value)
                    }),
            )
            .add(settings::item(
                fl!("output-denoise"),
                widget::toggler(settings_.output_denoise)
                    .on_toggle(|value| Message::Toggle(Toggle::OutputDenoise, value)),
            ))
            .into(),
        reset_row(fl!("reset-performance"), "performance"),
    ])
}

// ---------------------------------------------------------------------------
// Recording
// ---------------------------------------------------------------------------

fn recording_page<'a>(app: &'a App, state: &'a State) -> Element<'a, Message> {
    let space = cosmic::theme::spacing();
    let settings_ = &state.settings;
    let mode_index = RECORDING_MODES
        .iter()
        .position(|mode| *mode == settings_.recording_mode);
    let record_button = if state.recording {
        widget::button::destructive(fl!("stop-recording"))
            .leading_icon(widget::icon::from_name("media-playback-stop-symbolic"))
            .on_press(Message::ToggleRecording)
    } else {
        widget::button::suggested(fl!("record"))
            .leading_icon(widget::icon::from_name("media-record-symbolic"))
            .on_press_maybe(state.running.then_some(Message::ToggleRecording))
    };
    let elapsed = match app.meters.recording_seconds {
        Some(seconds) if state.recording => {
            let total = seconds as u64;
            format!(
                "● REC {:02}:{:02}:{:02}",
                total / 3600,
                (total / 60) % 60,
                total % 60
            )
        }
        _ if !state.running => fl!("record-hint"),
        _ => String::new(),
    };
    page_column(vec![
        settings::section()
            .title(fl!("section-recording"))
            .add(dropdown_item(
                fl!("recording-mode"),
                &app.recording_mode_labels,
                mode_index,
                Message::SelectRecordingMode,
            ))
            .add(settings::item(
                fl!("save-folder"),
                widget::row::with_capacity(4)
                    .push(widget::text::caption(settings_.recording_folder.as_str()))
                    .push(
                        widget::button::standard(fl!("change"))
                            .on_press(Message::PickRecordingFolder),
                    )
                    .push(
                        widget::button::standard(fl!("open-folder"))
                            .on_press(Message::OpenRecordingFolder),
                    )
                    .spacing(space.space_xs)
                    .align_y(Alignment::Center),
            ))
            .into(),
        widget::row::with_capacity(4)
            .push(widget::text::body(elapsed).width(Length::Fill))
            .push(record_button)
            .align_y(Alignment::Center)
            .into(),
    ])
}

// ---------------------------------------------------------------------------
// Log
// ---------------------------------------------------------------------------

fn log_page(app: &App) -> Element<'_, Message> {
    let space = cosmic::theme::spacing();
    widget::column::with_capacity(4)
        .push(
            widget::row::with_capacity(4)
                .push(widget::space::horizontal())
                .push(widget::button::standard(fl!("save-log")).on_press(Message::SaveLog))
                .push(widget::button::standard(fl!("clear-log")).on_press(Message::ClearLog))
                .spacing(space.space_xs),
        )
        .push(
            widget::container(
                widget::scrollable(widget::text::monotext(app.log.as_str()).width(Length::Fill))
                    .anchor_bottom()
                    .height(Length::Fill),
            )
            .class(cosmic::theme::Container::Card)
            .padding(space.space_s)
            .height(Length::Fill),
        )
        .spacing(space.space_s)
        .padding([0, space.space_s, space.space_s, space.space_s])
        .height(Length::Fill)
        .into()
}

// ---------------------------------------------------------------------------
// Control bar
// ---------------------------------------------------------------------------

pub fn control_bar<'a>(app: &'a App, state: &'a State) -> Element<'a, Message> {
    let space = cosmic::theme::spacing();
    let converting = state.running && state.function == "vc";
    let passthrough = state.running && state.function == "passthrough";
    let enabled = !app.starting;

    let start = if converting {
        widget::button::destructive(fl!("converting"))
            .leading_icon(widget::icon::from_name("media-playback-stop-symbolic"))
    } else {
        widget::button::suggested(if app.starting {
            fl!("starting")
        } else {
            fl!("start")
        })
        .leading_icon(widget::icon::from_name("media-playback-start-symbolic"))
    }
    .on_press_maybe(enabled.then_some(Message::Run(Function::Convert)));
    let passthrough_button = if passthrough {
        widget::button::destructive(fl!("passthrough-active"))
    } else {
        widget::button::standard(fl!("passthrough"))
    }
    .on_press_maybe(enabled.then_some(Message::Run(Function::Passthrough)));

    let meter = |label: String, value: f32| {
        widget::row::with_capacity(4)
            .push(widget::text::caption(label).width(Length::Fixed(32.0)))
            .push(
                widget::progress_bar::linear::Linear::new()
                    .progress(value)
                    .girth(6.0)
                    .width(Length::Fixed(METER_WIDTH)),
            )
            .spacing(space.space_xxs)
            .align_y(Alignment::Center)
    };
    let meters = widget::column::with_capacity(4)
        .push(meter(fl!("meter-in"), app.meters.input))
        .push(meter(fl!("meter-out"), app.meters.output))
        .push(meter(fl!("meter-mon"), app.meters.monitor))
        .spacing(2);

    let dash = || "—".to_string();
    let stats = widget::column::with_capacity(4)
        .push(widget::text::caption(fl!(
            "latency",
            value = state.delay_ms.map(|ms| ms.to_string()).unwrap_or_else(dash)
        )))
        .push(widget::text::caption(fl!(
            "inference",
            value = app
                .meters
                .infer_ms
                .map(|ms| ms.to_string())
                .unwrap_or_else(dash)
        )))
        .width(Length::Fixed(120.0));

    widget::container(
        widget::row::with_capacity(4)
            .push(start)
            .push(passthrough_button)
            .push(widget::text::body(app.status.as_str()).width(Length::Fill))
            .push(meters)
            .push(stats)
            .spacing(space.space_s)
            .align_y(Alignment::Center),
    )
    .class(cosmic::theme::Container::Card)
    .padding(space.space_s)
    .into()
}

/// Shown over the page while files are dragged onto the window.
pub fn drop_overlay<'a>() -> Element<'a, Message> {
    let space = cosmic::theme::spacing();
    widget::container(
        widget::container(
            widget::column::with_capacity(3)
                .push(widget::icon::from_name("document-open-symbolic").size(48))
                .push(widget::text::title3(fl!("drop-title")))
                .push(widget::text::body(fl!("drop-detail")))
                .spacing(space.space_xs)
                .align_x(Alignment::Center),
        )
        .class(cosmic::theme::Container::Dialog(true))
        .padding(space.space_l),
    )
    .center(Length::Fill)
    .class(cosmic::theme::Container::Transparent)
    .into()
}

#[cfg(test)]
mod tests {
    use super::mmss;

    #[test]
    fn formats_minutes_and_seconds() {
        assert_eq!(mmss(0.0), "00:00");
        assert_eq!(mmss(75.9), "01:15");
        assert_eq!(mmss(-3.0), "00:00");
    }
}
