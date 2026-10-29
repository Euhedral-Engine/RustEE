//! The hot-path lock ratchet. Every hot-path lock the audit lists carries a `HOT-FIX(<id>)` tag; the set of ids, and
//! the number of places each id is used, only shrink; and no other `allow` of a disallowed type or blocking call
//! enters the library sources unexplained.
//!
//! - A change that replaces a lock deletes its tags and lowers or deletes its entry in [`HOT_FIX_TAGS`]. A tag used
//!   in one more place, or an id that appears anew, fails until the list says so on purpose.
//! - A change that adds a lock for start, stop or rebalance state uses `LifecycleMutex` or `LifecycleRwLock`; the
//!   workspace lint (`clippy.toml`) refuses the standard lock types everywhere else.
//! - Every `allow(clippy::disallowed_types)` or `allow(clippy::disallowed_methods)` in the library sources begins its
//!   reason with `HOT-FIX(<id>)` (one or more, comma separated, each listed), `LIFECYCLE(<id>)` (a row of the audit)
//!   or, for a call, `JAVA-BACKOFF`; or it is a test-only `cfg_attr`, or a file-level allow of a file on
//!   [`EXEMPT_FILES`].
//! - A file that locks through the `lock!` macro, which names no lock type, has its call sites counted per file
//!   ([`LOCK_MACRO_SITES`]); the count only shrinks.
//! - A type alias may not name a lock type: the lint reports the alias, not its uses, so an alias would hide the lock
//!   from every site that uses it.

// The test reads the workspace's sources.
#![allow(clippy::disallowed_methods)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

/// The `HOT-FIX(<id>)` ids that the library sources carry today, one per finding of the audit, and the number of
/// places each is used (its field, each use the lint reports, its marker comment).
const HOT_FIX_TAGS: &[(&str, usize)] = &[
    ("G5", 3),
    ("G6", 3),
    ("G7", 3),
    ("C16", 3),
    ("C17", 3),
    ("C18", 3),
    ("C19", 3),
    ("C20", 5),
    ("G10", 3),
    ("G11", 4),
    ("G12", 4),
    ("G13", 4),
    ("G14", 4),
    ("G15", 4),
    ("H1", 5),
    ("H2", 4),
    ("H3", 4),
    ("H4", 2),
    ("H5", 4),
    ("H6", 3),
    ("H7", 3),
    ("I3", 3),
    ("I4", 3),
    ("I5", 3),
    ("I6", 3),
    ("I7", 3),
    ("I8", 3),
    ("I9", 3),
    ("I10", 3),
    ("I11", 3),
    ("I12", 3),
    ("I13", 3),
    ("I14", 3),
    ("I15", 3),
    ("M1", 1),
    ("S1", 1),
    ("S2", 1),
    ("S6a", 1),
];

/// The `lock!(..)` call sites by file (the macro names no lock type, so the lint cannot see them), with the ids of the
/// audit the locks it takes belong to. A file's count only shrinks, and is 0 once none of its ids is listed in
/// [`HOT_FIX_TAGS`]; a file not listed here has no site.
const LOCK_MACRO_SITES: &[(&str, &[&str], usize)] = &[
    ("euhedral-io/src/uring.rs", &["I3", "I4", "I5", "I6"], 5),
    (
        "euhedral-io/src/win/mod.rs",
        &["I7", "I8", "I9", "I10", "I11", "I12", "I13", "I14", "I15"],
        31,
    ),
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

const HOT_FIX: &str = "HOT-FIX(";
const LIFECYCLE: &str = "LIFECYCLE(";
const JAVA_BACKOFF: &str = "JAVA-BACKOFF";

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

/// `text` without its inline test modules (a test-only `cfg`, any attributes and comments, `mod name {` at the left
/// margin, and the body up to the closing brace at the left margin). The removed lines stay as blank lines, so line
/// numbers hold, and the library code after a test module is still there.
fn library_part(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut kept = String::new();
    let mut at = 0;
    while at < lines.len() {
        if is_test_only_cfg(lines[at]) {
            let module = (at + 1..lines.len())
                .find(|&next| !(lines[next].starts_with("#[") || lines[next].starts_with("//")));
            if let Some(module) = module
                && lines[module].starts_with("mod ")
                && lines[module].trim_end().ends_with('{')
            {
                let end = (module..lines.len())
                    .find(|&next| lines[next] == "}")
                    .unwrap_or(lines.len() - 1);
                kept.extend((at..=end).map(|_| '\n'));
                at = end + 1;
                continue;
            }
        }
        kept.push_str(lines[at]);
        kept.push('\n');
        at += 1;
    }
    kept
}

/// Every library source: `<workspace>/*/src/**/*.rs` except the modules declared for test builds, as (path from the
/// workspace root, text without its inline test modules).
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
            (relative, library_part(&text))
        })
        .collect()
}

/// How many times each `HOT-FIX(<id>)` appears in `text`, added to `uses`.
fn count_uses(text: &str, uses: &mut BTreeMap<String, usize>) {
    let mut rest = text;
    while let Some(at) = rest.find(HOT_FIX) {
        rest = &rest[at + HOT_FIX.len()..];
        let end = rest.find(')').unwrap_or(rest.len());
        *uses.entry(rest[..end].to_owned()).or_default() += 1;
    }
}

fn uses() -> BTreeMap<String, usize> {
    let mut uses = BTreeMap::new();
    for (_, text) in sources() {
        count_uses(&text, &mut uses);
    }
    uses
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

/// The text of `reason = "..."` in the body of an attribute.
fn reason_of(body: &str) -> Option<&str> {
    let after = &body[body.find("reason")? + "reason".len()..];
    let after = after.trim_start().strip_prefix('=')?.trim_start();
    let after = after.strip_prefix('"')?;
    Some(&after[..after.find('"')?])
}

/// What is wrong with a reason that must begin with an explanation, if anything. `methods_only` says the allow names
/// no disallowed type; `listed` holds the `HOT-FIX` ids; `audit` is the audit's text.
fn reason_problem(
    reason: &str,
    methods_only: bool,
    listed: &BTreeSet<&str>,
    audit: &str,
) -> Option<String> {
    let id_of = |rest: &str| rest.split_once(')').map(|(id, _)| id.to_owned());
    if reason.starts_with(HOT_FIX) {
        let mut rest = reason;
        while let Some(tagged) = rest.strip_prefix(HOT_FIX) {
            let id = id_of(tagged).unwrap_or_default();
            if !listed.contains(id.as_str()) {
                return Some(format!("`HOT-FIX({id})` is no listed id"));
            }
            rest = &tagged[id.len() + 1..];
            match rest.strip_prefix(", ") {
                Some(more) => rest = more,
                None => break,
            }
        }
        None
    } else if let Some(tagged) = reason.strip_prefix(LIFECYCLE) {
        let id = id_of(tagged).unwrap_or_default();
        (!is_a_finding(audit, &id)).then(|| format!("`LIFECYCLE({id})` is no row of the audit"))
    } else if reason.starts_with(JAVA_BACKOFF) {
        (!methods_only).then(|| "`JAVA-BACKOFF` explains a call, not a type".to_owned())
    } else {
        Some(format!(
            "the reason `{reason}` begins with no HOT-FIX, LIFECYCLE or JAVA-BACKOFF tag"
        ))
    }
}

/// The problems in one library source: an `allow` of a disallowed type or call with no explanation, and a type alias
/// of a lock type.
fn problems(path: &str, text: &str, listed: &BTreeSet<&str>, audit: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(at) = text[from..].find("allow(") {
        let start = from + at;
        from = start + "allow(".len();
        let body = balanced(text, start + "allow".len());
        let types = body.contains("clippy::disallowed_types");
        if !(types || body.contains("clippy::disallowed_methods")) {
            continue;
        }
        let before = text[..start].trim_end();
        let file_level = before.ends_with("#![");
        let test_only = before.ends_with("test,") && before.contains("cfg_attr(");
        let line = text[..start].matches('\n').count() + 1;
        if test_only || (file_level && EXEMPT_FILES.contains(&path)) {
            continue;
        }
        let problem = match reason_of(body) {
            None => Some("an allow of a disallowed type or call has no reason".to_owned()),
            Some(reason) => reason_problem(reason, !types, listed, audit),
        };
        if let Some(problem) = problem {
            found.push(format!("{path}:{line}: {problem}"));
        }
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
    !id.is_empty()
        && audit
            .lines()
            .any(|line| line.starts_with(&format!("| {id} |")))
}

fn audit() -> String {
    fs::read_to_string(workspace().join(AUDIT)).expect("the audit")
}

#[test]
fn the_hot_fix_tags_and_their_uses_are_exactly_the_listed_ones() {
    let found = uses();
    let listed: BTreeMap<String, usize> = HOT_FIX_TAGS
        .iter()
        .map(|(id, uses)| ((*id).to_owned(), *uses))
        .collect();
    assert_eq!(listed.len(), HOT_FIX_TAGS.len(), "an id is listed twice");
    let differing: Vec<String> = listed
        .keys()
        .chain(found.keys())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|id| listed.get(*id) != found.get(*id))
        .map(|id| {
            format!(
                "{id}: listed {:?}, found {:?}",
                listed.get(id),
                found.get(id)
            )
        })
        .collect();
    assert!(
        differing.is_empty(),
        "lower or delete the entries of HOT_FIX_TAGS for the tags this change removed; an id or a use that is new is a \
         new hot lock:\n{}",
        differing.join("\n")
    );
}

/// The `lock!(` call sites in `text`.
///
/// A site is `lock!` (not the tail of a longer name) followed by optional whitespace and an opening delimiter.
fn lock_sites(text: &str) -> usize {
    let mut count = 0;
    let mut rest = text;
    while let Some(at) = rest.find("lock!") {
        let before = rest[..at].chars().next_back();
        let after = rest[at + "lock!".len()..].trim_start();
        let part_of_a_name = before.is_some_and(|c| c.is_alphanumeric() || c == '_');
        if !part_of_a_name && after.starts_with(['(', '[', '{']) {
            count += 1;
        }
        rest = &rest[at + "lock!".len()..];
    }
    count
}

#[test]
fn the_lock_macro_sites_are_exactly_the_listed_ones() {
    let listed: BTreeSet<&str> = HOT_FIX_TAGS.iter().map(|(id, _)| *id).collect();
    let mut found: BTreeMap<String, usize> = BTreeMap::new();
    for (path, text) in sources() {
        if path.starts_with("euhedral-io/src/") && lock_sites(&text) > 0 {
            found.insert(path, lock_sites(&text));
        }
    }
    let mut problems = Vec::new();
    for (file, ids, count) in LOCK_MACRO_SITES {
        let actual = found.remove(*file).unwrap_or(0);
        if actual != *count {
            problems.push(format!(
                "{file}: listed {count} `lock!(` sites, found {actual}"
            ));
        }
        if *count > 0 && !ids.iter().any(|id| listed.contains(id)) {
            problems.push(format!(
                "{file}: none of its ids {ids:?} is listed any more, so it may have no `lock!(` site"
            ));
        }
    }
    for (file, actual) in found {
        problems.push(format!(
            "{file}: {actual} `lock!(` sites, and the file is not listed"
        ));
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

#[test]
fn lock_sites_are_found_however_they_are_spelled() {
    assert_eq!(lock_sites("lock!(&a) lock! (&b) lock!\n[c] lock!{d}"), 4);
    assert_eq!(
        lock_sites("unlock!(a) block!(b) my_lock!(c) macro_rules! lock {"),
        0
    );
}

#[test]
fn every_listed_id_is_a_finding_of_the_audit() {
    let audit = audit();
    for (id, _) in HOT_FIX_TAGS {
        assert!(is_a_finding(&audit, id), "`{id}` is no row of {AUDIT}");
    }
}

#[test]
fn every_disallowed_allow_is_explained_and_no_alias_hides_a_lock() {
    let audit = audit();
    let listed: BTreeSet<&str> = HOT_FIX_TAGS.iter().map(|(id, _)| *id).collect();
    let mut found = Vec::new();
    for (path, text) in sources() {
        found.extend(problems(&path, &text, &listed, &audit));
    }
    assert!(found.is_empty(), "{}", found.join("\n"));
}

#[test]
fn the_scan_finds_what_it_looks_for() {
    let audit = "| C25 | `x` |\n| load | `y` |\n";
    let listed: BTreeSet<&str> = ["C4", "C5"].into();
    let check = |path: &str, text: &str| problems(path, text, &listed, audit);

    let tagged = "#[allow(clippy::disallowed_types, reason = \"HOT-FIX(C4), HOT-FIX(C5)\")]\nstruct A;\n\
                  #[allow(\n    clippy::disallowed_methods,\n    reason = \"JAVA-BACKOFF\"\n)]\nfn b() {}\n\
                  #[allow(clippy::disallowed_methods, reason = \"LIFECYCLE(load): it loads\")]\nfn c() {}\n\
                  #![cfg_attr(\n    test,\n    allow(clippy::disallowed_types, reason = \"fixtures\")\n)]\n";
    assert!(check("a.rs", tagged).is_empty());

    for bad in [
        "#[allow(clippy::disallowed_types)]\nstruct A;\n",
        "#[allow(clippy::disallowed_types, reason = \"it is fine\")]\nstruct A;\n",
        // The tag must open the reason.
        "#[allow(clippy::disallowed_types, reason = \"see HOT-FIX(C4)\")]\nstruct A;\n",
        // An id the list lacks, and one the audit lacks.
        "#[allow(clippy::disallowed_types, reason = \"HOT-FIX(C9)\")]\nstruct A;\n",
        "#[allow(clippy::disallowed_methods, reason = \"LIFECYCLE(C99)\")]\nfn a() {}\n",
        // A back-off explains a call, not a type.
        "#[allow(clippy::disallowed_types, reason = \"JAVA-BACKOFF\")]\nstruct A;\n",
        "#[allow(clippy::disallowed_types, clippy::disallowed_methods, reason = \"JAVA-BACKOFF\")]\nfn a() {}\n",
    ] {
        assert_eq!(check("a.rs", bad).len(), 1, "{bad}");
    }

    let file_level = "#![allow(clippy::disallowed_types, reason = \"a double\")]\n";
    assert_eq!(check("a.rs", file_level).len(), 1);
    assert!(check("euhedral-gpu/src/fake/mod.rs", file_level).is_empty());

    let alias = "type Queue = Arc<Mutex<Vec<u8>>>;\npub(crate) type Rx = Receiver<u8>;\ntype Fine = Vec<u8>;\n";
    assert_eq!(check("a.rs", alias).len(), 2);
    let other = "#[allow(clippy::needless_range_loop)]\ntype Mutexes = Vec<u8>;\n";
    assert!(check("a.rs", other).is_empty());
}

#[test]
fn uses_are_counted_per_id() {
    let mut counted = BTreeMap::new();
    count_uses(
        "// HOT-FIX(M1)\nreason = \"HOT-FIX(C4), HOT-FIX(C5)\"\nreason = \"HOT-FIX(C4)\"",
        &mut counted,
    );
    assert_eq!(counted.get("C4"), Some(&2));
    assert_eq!(counted.get("C5"), Some(&1));
    assert_eq!(counted.get("M1"), Some(&1));
}

#[test]
fn library_code_after_an_inline_test_module_is_still_scanned() {
    let text = "fn a() {}\n#[cfg(test)]\n#[allow(clippy::disallowed_types)]\nmod tests {\n    use super::*;\n}\n\
                #[allow(clippy::disallowed_types)]\nstruct Late;\n";
    let kept = library_part(text);
    assert_eq!(
        kept.lines().count(),
        text.lines().count(),
        "line numbers hold"
    );
    assert!(!kept.contains("mod tests") && kept.contains("struct Late"));
    let audit = "";
    let listed = BTreeSet::new();
    let found = problems("a.rs", &kept, &listed, audit);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(found[0].starts_with("a.rs:7:"), "{found:?}");
}

#[test]
fn the_audit_names_findings_by_row() {
    let audit = "| C4 | `x` | y |\n| S6a | `z` |\n";
    assert!(is_a_finding(audit, "C4"));
    assert!(is_a_finding(audit, "S6a"));
    assert!(!is_a_finding(audit, "C5"));
    assert!(!is_a_finding(audit, "C"));
    assert!(!is_a_finding(audit, ""));
}
