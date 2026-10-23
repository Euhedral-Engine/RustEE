//! Bounded reads of small kernel files, with a memory of which sources have been seen.

use crate::monitor::sampling::{Signal, SignalValue};
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, Read as _};
use std::path::{Path, PathBuf};

/// Kernel text files are far smaller; a larger file is treated as unreadable rather than parsed in
/// part.
pub(crate) const MAX_FILE_BYTES: u64 = 1 << 20;

/// Reads at most [`MAX_FILE_BYTES`]; a larger file is an error.
pub(crate) fn read_bounded(path: &Path) -> io::Result<String> {
    let mut text = String::new();
    File::open(path)?
        .take(MAX_FILE_BYTES + 1)
        .read_to_string(&mut text)?;
    if text.len() as u64 > MAX_FILE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "file exceeds the read bound",
        ));
    }
    Ok(text)
}

/// A bounded read, trimmed, or `None`.
pub(crate) fn read_trimmed(path: &Path) -> Option<String> {
    read_bounded(path).ok().map(|text| text.trim().to_owned())
}

/// The outcome of reading one source.
#[derive(Debug)]
pub(crate) enum Read {
    Text(String),
    /// The host does not provide it.
    Absent,
    /// It exists, or existed, but could not be read.
    Failed,
}

/// Reads sources and remembers which were ever read, so one that vanishes is a failure rather than
/// a missing feature.
#[derive(Debug, Default)]
pub(crate) struct Sources {
    seen: HashSet<PathBuf>,
}

impl Sources {
    pub(crate) fn read(&mut self, path: Option<&Path>) -> Read {
        let Some(path) = path else {
            return Read::Absent;
        };
        match read_bounded(path) {
            Ok(text) => {
                if !self.seen.contains(path) {
                    self.seen.insert(path.to_path_buf());
                }
                Read::Text(text)
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound && !self.seen.contains(path) => {
                Read::Absent
            }
            Err(error) => {
                log::debug!("cannot read {}: {error}", path.display());
                Read::Failed
            }
        }
    }

    /// Reads and parses a source into a signal: absent is unsupported; unreadable or unparsable is a
    /// transient failure.
    pub(crate) fn signal<T: SignalValue>(
        &mut self,
        path: Option<&Path>,
        at: i64,
        parse: impl FnOnce(&str) -> Option<T>,
    ) -> Signal<T> {
        signal_of(&self.read(path), at, parse)
    }
}

/// The signal for an already-read source.
pub(crate) fn signal_of<T: SignalValue>(
    read: &Read,
    at: i64,
    parse: impl FnOnce(&str) -> Option<T>,
) -> Signal<T> {
    match read {
        Read::Text(text) => parse(text).map_or_else(
            || Signal::transient_failure(at),
            |value| Signal::valid(value, at),
        ),
        Read::Absent => Signal::unsupported(at),
        Read::Failed => Signal::transient_failure(at),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::monitor::sampling::SignalValidity;
    use crate::platform::testing::TempDir;

    #[test]
    fn reads_are_bounded() {
        let dir = TempDir::new("bounded");
        let big = dir.path().join("big");
        std::fs::write(&big, vec![b'1'; MAX_FILE_BYTES as usize + 1]).unwrap();
        assert!(read_bounded(&big).is_err());
        std::fs::write(&big, vec![b'1'; MAX_FILE_BYTES as usize]).unwrap();
        assert_eq!(read_bounded(&big).unwrap().len(), MAX_FILE_BYTES as usize);
    }

    #[test]
    fn a_vanished_source_fails_and_a_never_seen_one_is_absent() {
        let dir = TempDir::new("sources");
        let file = dir.path().join("value");
        let mut sources = Sources::default();
        let parse = |text: &str| text.trim().parse::<u64>().ok();
        assert_eq!(
            sources.signal(Some(&file), 1, parse).validity(),
            SignalValidity::Unsupported
        );
        std::fs::write(&file, "42\n").unwrap();
        assert_eq!(sources.signal(Some(&file), 2, parse).get(), Some(42));
        std::fs::write(&file, "x").unwrap();
        assert_eq!(
            sources.signal(Some(&file), 3, parse).validity(),
            SignalValidity::TransientFailure
        );
        std::fs::remove_file(&file).unwrap();
        assert_eq!(
            sources.signal(Some(&file), 4, parse).validity(),
            SignalValidity::TransientFailure
        );
        assert_eq!(
            sources.signal(None, 5, parse).validity(),
            SignalValidity::Unsupported
        );
    }
}
