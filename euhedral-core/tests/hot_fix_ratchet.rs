//! The hot-path lock ratchet. Every hot-path lock the audit lists carries a `HOT-FIX(<id>)` tag; the set of ids only
//! shrinks; and no other `allow` of a disallowed lock type or blocking call enters the library sources unexplained.
//!
//! - A change that replaces a lock deletes its tag and its id from [`HOT_FIX_IDS`].
//! - A change that adds a lock for start, stop or rebalance state uses `LifecycleMutex` or `LifecycleRwLock`; the
//!   workspace lint (`clippy.toml`) refuses the standard lock types everywhere else.
//! - Every `allow(clippy::disallowed_types)` or `allow(clippy::disallowed_methods)` in the library sources carries a
//!   `HOT-FIX(`, `LIFECYCLE(` or `JAVA-BACKOFF` reason, is a test-only `cfg_attr`, or is a file-level allow of a file
//!   on [`EXEMPT_FILES`].
//! - A type alias may not name a lock type: the lint reports the alias, not its uses, so an alias would hide the lock
//!   from every site that uses it.

// The test reads the workspace's sources.
#![allow(clippy::disallowed_methods)]

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// The `HOT-FIX(<id>)` ids that the library sources carry today, one per finding of the audit.
const HOT_FIX_IDS: &[&str] = &[
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C16",
    "C17", "C18", "C19", "C20", "G1", "G2", "G3", "G5", "G6", "G7", "G10", "G11", "G12", "G13",
    "G14", "G15", "I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13",
    "I14", "I15", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "M1", "S1", "S2", "S6a",
];

/// Files whose file-level `allow` of a disallowed type is part of their design.
const EXEMPT_FILES: &[&str] = &[
    // The lifecycle locks wrap the standard types.
    "euhedral-data-structures/src/sync/lifecycle.rs",
    // The simulated device is one state behind a mutex: it stands for hardware, not for a submission path.
    "euhedral-gpu/src/fake/mod.rs",
];

/// The audit the ids are findings of.
const AUDIT: &str = "docs/superpowers/specs/2026-10-28-lock-free-hot-path-audit.md";

const MARKER: &str = "HOT-FIX(";

/// Reasons that explain an `allow` of a disallowed type or call.
const REASONS: &[&str] = &["HOT-FIX(", "LIFECYCLE(", "JAVA-BACKOFF"];

/// Types that make a lock, a wait or a channel.
const LOCK_TYPES: &[&str] = &[
    "Mutex",
    "RwLock",
    "Condvar",
    "Barrier",
    "Sender",
    "SyncSender",
    "Receiver",
];

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// Whether `path` is test code, which the ratchet does not count.
fn is_test_code(path: &Path) -> bool {
    let name = path.file_name().and_then(|name| name.to_str());
    matches!(name, Some("tests.rs" | "testing.rs" | "tests" | "testing"))
}

fn collect(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if is_test_code(&path) {
            continue;
        }
        if path.is_dir() {
            collect(&path, found);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            found.push(path);
        }
    }
}

/// Whether `line` is a `cfg` attribute that holds in test builds only: `#[cfg(test)]`, or an `all(..)` with `test`.
fn is_test_only_cfg(line: &str) -> bool {
    line.trim_end() == "#[cfg(test)]" || (line.starts_with("#[cfg(all(") && line.contains("test"))
}

/// The modules a file declares for test builds only (`#[cfg(test)] mod name;`).
fn test_modules(text: &str) -> Vec<String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut modules = Vec::new();
    for (at, line) in lines.iter().enumerate() {
        if !is_test_only_cfg(line) {
            continue;
        }
        if let Some(next) = lines.get(at + 1)
            && let Some(name) = next.trim_end().strip_prefix("mod ")
            && let Some(name) = name.strip_suffix(';')
        {
            modules.push(name.to_owned());
        }
    }
    modules
}

/// The text before the first inline test module: a test-only `cfg`, any attributes and comments, and `mod name {` at
/// the left margin.
fn library_part(text: &str) -> &str {
    let lines: Vec<&str> = text.lines().collect();
    let mut offset = 0;
    for (at, line) in lines.iter().enumerate() {
        if is_test_only_cfg(line) {
            let module = lines[at + 1..]
                .iter()
                .find(|next| !(next.starts_with("#[") || next.starts_with("//")));
            if module.is_some_and(|next| next.starts_with("mod ") && next.trim_end().ends_with('{'))
            {
                return &text[..offset];
            }
        }
        offset += line.len() + 1;
    }
    text
}

/// Every library source: `<workspace>/*/src/**/*.rs` except the modules declared for test builds, as (path from the
/// workspace root, text up to the first inline test module).
fn sources() -> Vec<(String, String)> {
    let mut crates: Vec<PathBuf> = fs::read_dir(workspace())
        .expect("the workspace directory")
        .flatten()
        .map(|entry| entry.path().join("src"))
        .filter(|src| src.is_dir())
        .collect();
    crates.sort();
    let mut files = Vec::new();
    for src in &crates {
        collect(src, &mut files);
    }
    let texts: Vec<(PathBuf, String)> = files
        .into_iter()
        .map(|file| {
            let text = fs::read_to_string(&file).expect("a readable source file");
            (file, text)
        })
        .collect();
    // The files those declarations name: `dir/name.rs`, `dir/name/mod.rs`, `dir/<this file>/name.rs`.
    let mut test_only = BTreeSet::new();
    for (file, text) in &texts {
        let dir = file.parent().expect("a file has a directory");
        let own = file.file_stem().map(PathBuf::from).unwrap_or_default();
        for name in test_modules(text) {
            test_only.insert(dir.join(format!("{name}.rs")));
            test_only.insert(dir.join(&name).join("mod.rs"));
            test_only.insert(dir.join(&own).join(format!("{name}.rs")));
        }
    }
    texts
        .into_iter()
        .filter(|(file, _)| !test_only.contains(file))
        .map(|(file, text)| {
            let relative = file
                .strip_prefix(workspace())
                .expect("under the workspace")
                .to_string_lossy()
                .replace('\\', "/");
            (relative, library_part(&text).to_owned())
        })
        .collect()
}

/// The distinct ids of every `HOT-FIX(<id>)` in `text`.
fn ids_in(text: &str, ids: &mut BTreeSet<String>) {
    let mut rest = text;
    while let Some(at) = rest.find(MARKER) {
        rest = &rest[at + MARKER.len()..];
        let end = rest.find(')').unwrap_or(rest.len());
        ids.insert(rest[..end].to_owned());
    }
}

fn tags() -> BTreeSet<String> {
    let mut ids = BTreeSet::new();
    for (_, text) in sources() {
        ids_in(&text, &mut ids);
    }
    ids
}

/// The text between the parenthesis that opens at `open` and its match.
fn balanced(text: &str, open: usize) -> &str {
    let mut depth = 0_usize;
    for (at, character) in text[open..].char_indices() {
        match character {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    return &text[open + 1..open + at];
                }
            }
            _ => {}
        }
    }
    &text[open + 1..]
}

/// The problems in one library source: an `allow` of a disallowed type or call with no explanation, and a type alias
/// of a lock type.
fn problems(path: &str, text: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(at) = text[from..].find("allow(") {
        let start = from + at;
        from = start + "allow(".len();
        let body = balanced(text, start + "allow".len());
        if !(body.contains("clippy::disallowed_types")
            || body.contains("clippy::disallowed_methods"))
        {
            continue;
        }
        let before = text[..start].trim_end();
        let file_level = before.ends_with("#![");
        let test_only = before.ends_with("test,") && before.contains("cfg_attr(");
        let line = text[..start].matches('\n').count() + 1;
        if test_only
            || REASONS.iter().any(|reason| body.contains(reason))
            || (file_level && EXEMPT_FILES.contains(&path))
        {
            continue;
        }
        found.push(format!(
            "{path}:{line}: an allow of a disallowed type or call with no HOT-FIX, LIFECYCLE or JAVA-BACKOFF reason"
        ));
    }
    for (index, line) in text.lines().enumerate() {
        let line = line.trim_start();
        let line = line
            .strip_prefix("pub(crate) ")
            .or_else(|| line.strip_prefix("pub(super) "))
            .or_else(|| line.strip_prefix("pub "))
            .unwrap_or(line);
        let Some(alias) = line.strip_prefix("type ") else {
            continue;
        };
        let Some((_, aliased)) = alias.split_once('=') else {
            continue;
        };
        let names_a_lock = aliased
            .split(|character: char| !(character.is_alphanumeric() || character == '_'))
            .any(|word| LOCK_TYPES.contains(&word));
        if names_a_lock {
            found.push(format!(
                "{path}:{}: a type alias of a lock type hides the lock from the lint",
                index + 1
            ));
        }
    }
    found
}

/// Whether `id` is a row of one of the audit's tables.
fn is_a_finding(audit: &str, id: &str) -> bool {
    audit
        .lines()
        .any(|line| line.starts_with(&format!("| {id} |")))
}

#[test]
fn the_hot_fix_tags_are_exactly_the_listed_ids() {
    let found = tags();
    let listed: BTreeSet<String> = HOT_FIX_IDS.iter().map(|id| (*id).to_owned()).collect();
    assert_eq!(listed.len(), HOT_FIX_IDS.len(), "an id is listed twice");
    let removed: Vec<_> = listed.difference(&found).collect();
    let added: Vec<_> = found.difference(&listed).collect();
    assert!(
        removed.is_empty() && added.is_empty(),
        "delete the ids this change removed from HOT_FIX_IDS (tags gone from the sources: {removed:?}); a new id \
         is a new hot lock (tags the list lacks: {added:?})"
    );
}

#[test]
fn every_listed_id_is_a_finding_of_the_audit() {
    let audit = fs::read_to_string(workspace().join(AUDIT)).expect("the audit");
    for id in HOT_FIX_IDS {
        assert!(is_a_finding(&audit, id), "`{id}` is no row of {AUDIT}");
    }
}

#[test]
fn every_disallowed_allow_is_explained_and_no_alias_hides_a_lock() {
    let mut found = Vec::new();
    for (path, text) in sources() {
        found.extend(problems(&path, &text));
    }
    assert!(found.is_empty(), "{}", found.join("\n"));
}

#[test]
fn the_scan_finds_what_it_looks_for() {
    let tagged = "#[allow(clippy::disallowed_types, reason = \"HOT-FIX(C4)\")]\nstruct A;\n\
                  #[allow(\n    clippy::disallowed_methods,\n    reason = \"JAVA-BACKOFF\"\n)]\nfn b() {}\n\
                  #![cfg_attr(\n    test,\n    allow(clippy::disallowed_types, reason = \"fixtures\")\n)]\n";
    assert!(problems("a.rs", tagged).is_empty());

    let bare = "#[allow(clippy::disallowed_types)]\nstruct A;\n";
    assert_eq!(problems("a.rs", bare).len(), 1);
    let reasoned = "#[allow(clippy::disallowed_types, reason = \"it is fine\")]\nstruct A;\n";
    assert_eq!(problems("a.rs", reasoned).len(), 1);

    let file_level = "#![allow(clippy::disallowed_types, reason = \"a double\")]\n";
    assert_eq!(problems("a.rs", file_level).len(), 1);
    assert!(problems("euhedral-gpu/src/fake/mod.rs", file_level).is_empty());

    let alias = "type Queue = Arc<Mutex<Vec<u8>>>;\npub(crate) type Rx = Receiver<u8>;\ntype Fine = Vec<u8>;\n";
    assert_eq!(problems("a.rs", alias).len(), 2);
    let other = "#[allow(clippy::needless_range_loop)]\ntype Mutexes = Vec<u8>;\n";
    assert!(problems("a.rs", other).is_empty());
}

#[test]
fn the_audit_names_findings_by_row() {
    let audit = "| C4 | `x` | y |\n| S6a | `z` |\n";
    assert!(is_a_finding(audit, "C4"));
    assert!(is_a_finding(audit, "S6a"));
    assert!(!is_a_finding(audit, "C5"));
    assert!(!is_a_finding(audit, "C"));
}
