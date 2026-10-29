//! The hot-path lock ratchet: every hot-path lock the audit lists carries a `HOT-FIX(<id>)` tag, and the count of
//! distinct ids only goes down.
//!
//! A change that removes a lock removes its tag and lowers [`HOT_FIX_TAGS`] by the tags it removed. A change that adds
//! a lock must use `LifecycleMutex` or `LifecycleRwLock` for start, stop and rebalance state; the workspace lint
//! (`clippy.toml`) refuses the standard lock types everywhere else.

// The test reads the workspace's sources.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// The distinct `HOT-FIX(<id>)` ids in the workspace's library sources.
const HOT_FIX_TAGS: usize = 57;

const MARKER: &str = "HOT-FIX(";

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// Whether `path` is test code, which the ratchet does not count.
fn is_test_code(path: &Path) -> bool {
    let name = path.file_name().and_then(|name| name.to_str());
    matches!(name, Some("tests.rs" | "testing.rs" | "tests" | "testing"))
}

fn sources(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if is_test_code(&path) {
            continue;
        }
        if path.is_dir() {
            sources(&path, found);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            found.push(path);
        }
    }
}

/// The distinct ids of every `HOT-FIX(<id>)` in `<workspace>/*/src/**/*.rs`.
fn tags() -> BTreeSet<String> {
    let mut crates: Vec<PathBuf> = fs::read_dir(workspace())
        .expect("the workspace directory")
        .flatten()
        .map(|entry| entry.path().join("src"))
        .filter(|src| src.is_dir())
        .collect();
    crates.sort();
    let mut files = Vec::new();
    for src in &crates {
        sources(src, &mut files);
    }
    let mut ids = BTreeSet::new();
    for file in files {
        let text = fs::read_to_string(&file).expect("a readable source file");
        let mut rest = text.as_str();
        while let Some(at) = rest.find(MARKER) {
            rest = &rest[at + MARKER.len()..];
            let end = rest.find(')').unwrap_or(rest.len());
            ids.insert(rest[..end].to_owned());
        }
    }
    ids
}

/// Whether `id` names a finding of the audit: `C`, `G`, `I`, `H` or `M` and a number, or `S1`, `S2`, `S6a`.
fn names_a_finding(id: &str) -> bool {
    if matches!(id, "S1" | "S2" | "S6a") {
        return true;
    }
    let mut chars = id.chars();
    matches!(chars.next(), Some('C' | 'G' | 'I' | 'H' | 'M'))
        && !chars.as_str().is_empty()
        && chars.all(|digit| digit.is_ascii_digit())
}

#[test]
fn the_hot_fix_tags_are_exactly_the_ratchet() {
    let tags = tags();
    assert_eq!(
        tags.len(),
        HOT_FIX_TAGS,
        "lower HOT_FIX_TAGS by the tags this change removed: {tags:?}"
    );
}

#[test]
fn every_tag_names_a_finding_of_the_audit() {
    for id in tags() {
        assert!(names_a_finding(&id), "`HOT-FIX({id})` names no finding");
    }
}

#[test]
fn the_ids_are_checked_by_shape() {
    for good in ["C1", "C20", "G15", "I9", "H7", "M1", "S1", "S2", "S6a"] {
        assert!(names_a_finding(good), "{good}");
    }
    for bad in ["", "C", "X1", "S3", "S6b", "C1a", "<id>", "c1"] {
        assert!(!names_a_finding(bad), "{bad}");
    }
}
