//! Fluent translations embedded in the binary (`i18n/<lang>/rvc_realtime.ftl`).

use std::sync::LazyLock;

use i18n_embed::{
    DefaultLocalizer, LanguageLoader, Localizer,
    fluent::{FluentLanguageLoader, fluent_language_loader},
};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "i18n/"]
struct Localizations;

pub static LANGUAGE_LOADER: LazyLock<FluentLanguageLoader> = LazyLock::new(|| {
    let loader: FluentLanguageLoader = fluent_language_loader!();
    loader
        .load_fallback_language(&Localizations)
        .expect("Error while loading fallback language");
    loader
});

/// Translate a message by id: `fl!("start")`, `fl!("status-file-selected", name = file)`.
#[macro_export]
macro_rules! fl {
    ($message_id:literal) => {{
        i18n_embed_fl::fl!($crate::i18n::LANGUAGE_LOADER, $message_id)
    }};
    ($message_id:literal, $($args:expr),*) => {{
        i18n_embed_fl::fl!($crate::i18n::LANGUAGE_LOADER, $message_id, $($args), *)
    }};
}

/// Select the languages to use; `RVC_UI_LANGUAGE` overrides the desktop locale.
pub fn init() {
    let requested = match std::env::var("RVC_UI_LANGUAGE") {
        Ok(language) if !language.is_empty() => language
            .parse()
            .map(|identifier| vec![identifier])
            .unwrap_or_default(),
        _ => i18n_embed::DesktopLanguageRequester::requested_languages(),
    };
    let localizer = DefaultLocalizer::new(&*LANGUAGE_LOADER, &Localizations);
    if let Err(error) = localizer.select(&requested) {
        eprintln!("Error while loading languages: {error}");
    }
}
