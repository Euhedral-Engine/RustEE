//! The workspace's crates form layers, and a crate depends only on lower layers: no cycles, not through a
//! dev-dependency, a build dependency, a target table or a feature.

#![allow(
    clippy::disallowed_methods,
    reason = "the test reads the workspace's manifests"
)]

use std::path::PathBuf;

/// Layer 0 holds the lock-free data structures and the hashing; the hardware utilities use the data structures'
/// lifecycle locks, so they are layer 1. Each layer builds on the ones below it.
const LAYERS: &[(&str, u8)] = &[
    ("euhedral-data-structures", 0),
    ("euhedral-hashing", 0),
    ("euhedral-hardware-utils", 1),
    ("euhedral-core", 2),
    ("euhedral-graph", 3),
    ("euhedral-io", 3),
    ("euhedral-memory", 4),
    ("euhedral-gpu", 5),
];

fn layer(name: &str) -> Option<u8> {
    LAYERS.iter().find(|(n, _)| *n == name).map(|(_, l)| *l)
}

/// The `package = "name"` of a dependency line, if it renames one.
fn package_of(line: &str) -> Option<String> {
    let mut rest = line;
    while let Some(at) = rest.find("package") {
        let (before, after) = rest.split_at(at);
        let boundary = before
            .chars()
            .next_back()
            .is_none_or(|c| !(c.is_alphanumeric() || c == '_' || c == '-'));
        let after = after["package".len()..].trim_start();
        if boundary
            && let Some(value) = after.strip_prefix('=')
            && let Some(value) = value.trim_start().strip_prefix('"')
            && let Some(end) = value.find('"')
        {
            return Some(value[..end].to_owned());
        }
        rest = &rest[at + "package".len()..];
    }
    None
}

/// Every `euhedral-*` crate named by any dependency table of `manifest`, by the package it names, so
/// `alias = { package = "euhedral-core" }` counts as `euhedral-core` and `euhedral-core = { package = "other" }` does
/// not.
fn euhedral_deps(manifest: &str) -> Vec<String> {
    // (the key a dependency is declared under, the package it renames to).
    let mut entries: Vec<(String, Option<String>)> = Vec::new();
    // The dependency a dotted header (`[dependencies.euhedral-memory]`) is the table of.
    let mut dotted: Option<usize> = None;
    let mut in_deps = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(header) = line.strip_prefix('[') {
            let header = header.trim_start_matches('[');
            let header = header.trim_end_matches(']').trim();
            dotted = None;
            in_deps = false;
            if let Some(at) = header.find("dependencies.") {
                // A dotted header names the dependency after `dependencies.`.
                let name = header[at + "dependencies.".len()..]
                    .trim()
                    .trim_matches('"');
                entries.push((name.to_owned(), None));
                dotted = Some(entries.len() - 1);
                in_deps = true;
            } else if header.ends_with("dependencies") {
                in_deps = true;
            }
            continue;
        }
        if !in_deps {
            continue;
        }
        if let Some(entry) = dotted {
            if let Some(package) = package_of(line) {
                entries[entry].1 = Some(package);
            }
            continue;
        }
        let key = line
            .split(['=', '.'])
            .next()
            .unwrap_or_default()
            .trim()
            .trim_matches('"');
        if key.is_empty() {
            continue;
        }
        let package = package_of(line);
        match entries.iter_mut().find(|(alias, _)| alias == key) {
            Some(entry) => entry.1 = entry.1.take().or(package),
            None => entries.push((key.to_owned(), package)),
        }
    }
    let mut deps: Vec<String> = Vec::new();
    for (alias, package) in entries {
        let real = package.unwrap_or(alias);
        if real.starts_with("euhedral-") && !deps.contains(&real) {
            deps.push(real);
        }
    }
    deps
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// The member names listed in the workspace manifest's `members = [...]`.
fn members() -> Vec<String> {
    let text = std::fs::read_to_string(workspace_root().join("Cargo.toml")).unwrap();
    let start = text.find("members").expect("a members list");
    let open = start + text[start..].find('[').expect("a members list");
    let close = open + text[open..].find(']').expect("a closed members list");
    text[open + 1..close]
        .split(',')
        .map(|m| m.trim().trim_matches('"').to_owned())
        .filter(|m| !m.is_empty())
        .collect()
}

#[test]
fn the_parser_finds_every_kind_of_dependency_table() {
    let m = "[package]\nname = \"x\"\n[dependencies]\neuhedral-core = { path = \"../c\" }\nlog = \"1\"\n\
             [dev-dependencies]\neuhedral-graph.workspace = true\n\
             [target.'cfg(unix)'.dependencies]\neuhedral-io = { path = \"../io\" }\n\
             [build-dependencies]\neuhedral-hashing = \"*\"\n[dependencies.euhedral-memory]\npath = \"../m\"\n\
             [features]\neuhedral-gpu = []\n";
    let mut got = euhedral_deps(m);
    got.sort();
    assert_eq!(
        got,
        [
            "euhedral-core",
            "euhedral-graph",
            "euhedral-hashing",
            "euhedral-io",
            "euhedral-memory"
        ]
    );
}

#[test]
fn a_renamed_dependency_counts_as_the_package_it_names() {
    let m = "[dependencies]\n\
             core = { path = \"../c\", package = \"euhedral-core\" }\n\
             euhedral-io = { path = \"../x\", package = \"something-else\" }\n\
             lower = { version = \"1\", features = [\"x\"], package=\"euhedral-hashing\" }\n\
             [dev-dependencies]\n\
             alias.package = \"euhedral-graph\"\n\
             [dependencies.memory]\n\
             path = \"../m\"\n\
             package = \"euhedral-memory\"\n\
             [target.'cfg(unix)'.dependencies.plain]\n\
             version = \"1\"\n\
             [features]\n\
             default = [\"package\"]\n";
    let mut got = euhedral_deps(m);
    got.sort();
    assert_eq!(
        got,
        [
            "euhedral-core",
            "euhedral-graph",
            "euhedral-hashing",
            "euhedral-memory"
        ]
    );
}

#[test]
fn every_member_has_a_layer() {
    for member in members() {
        assert!(layer(&member).is_some(), "add {member} to LAYERS");
    }
}

#[test]
fn crates_depend_only_on_lower_layers() {
    for member in members() {
        let own = layer(&member).unwrap_or_else(|| panic!("add {member} to LAYERS"));
        let manifest =
            std::fs::read_to_string(workspace_root().join(&member).join("Cargo.toml")).unwrap();
        for dep in euhedral_deps(&manifest) {
            if dep == member {
                // A crate may name itself to turn on its own features for its tests.
                continue;
            }
            let theirs = layer(&dep).unwrap_or_else(|| panic!("add {dep} to LAYERS"));
            assert!(
                theirs < own,
                "{member} (layer {own}) depends on {dep} (layer {theirs})"
            );
        }
    }
}
