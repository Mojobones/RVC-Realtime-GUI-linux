//! Files dragged onto the window from a file manager (`text/uri-list`).

use std::borrow::Cow;

use cosmic::iced::clipboard::mime::AllowedMimeTypes;

const URI_LIST: &str = "text/uri-list";

/// Local file paths from a drop; non-file URIs are ignored.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DroppedFiles(pub Vec<String>);

impl AllowedMimeTypes for DroppedFiles {
    fn allowed() -> Cow<'static, [String]> {
        Cow::Owned(vec![URI_LIST.to_string()])
    }
}

impl TryFrom<(Vec<u8>, String)> for DroppedFiles {
    type Error = ();

    fn try_from((data, mime): (Vec<u8>, String)) -> Result<Self, Self::Error> {
        if mime != URI_LIST {
            return Err(());
        }
        Ok(parse_uri_list(&String::from_utf8_lossy(&data)))
    }
}

/// Parse RFC 2483 `text/uri-list`: one URI per line, `#` lines are comments.
pub fn parse_uri_list(text: &str) -> DroppedFiles {
    DroppedFiles(
        text.lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .filter_map(|line| url::Url::parse(line).ok())
            .filter(|url| url.scheme() == "file")
            .filter_map(|url| url.to_file_path().ok())
            .map(|path| path.to_string_lossy().into_owned())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_file_uris_and_skips_comments_and_other_schemes() {
        let list = "# dragged from COSMIC Files\r\n\
                    file:///home/me/Downloads/My%20Voice.pth\r\n\
                    file:///home/me/Downloads/added_My%20Voice.index\r\n\
                    https://example.com/model.pth\r\n";

        assert_eq!(
            parse_uri_list(list).0,
            vec![
                "/home/me/Downloads/My Voice.pth".to_string(),
                "/home/me/Downloads/added_My Voice.index".to_string(),
            ]
        );
    }

    #[test]
    fn rejects_other_mime_types() {
        assert!(DroppedFiles::try_from((b"file:///a.pth".to_vec(), "text/plain".into())).is_err());
        assert_eq!(
            DroppedFiles::try_from((b"file:///a.pth\n".to_vec(), URI_LIST.into())),
            Ok(DroppedFiles(vec!["/a.pth".to_string()]))
        );
    }
}
