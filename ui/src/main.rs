//! COSMIC front end for the RVC real-time engine (`engine/` in the repository).

mod app;
mod drop;
mod engine;
mod i18n;
mod pages;

use cosmic::app::Settings;
use cosmic::iced::Size;

fn main() -> cosmic::iced::Result {
    i18n::init();
    let settings = Settings::default()
        .size(Size::new(960.0, 760.0))
        .size_limits(
            cosmic::iced::Limits::NONE
                .min_width(640.0)
                .min_height(480.0),
        );
    cosmic::app::run::<app::App>(settings, ())
}
