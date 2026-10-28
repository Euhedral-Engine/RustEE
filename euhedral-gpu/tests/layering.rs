//! The workspace's crates form layers, and a crate depends only on lower layers: no cycles, not through a
//! dev-dependency, a build dependency, a target table or a feature.

#![allow(
    clippy::disallowed_methods,
    reason = "the test reads the workspace's manifests"
)]

use std::path::PathBuf;

/// Layer 0 holds the utilities; each layer builds on the ones below it.
const LAYERS: &[(&str, u8)] = &[
    ("euhedral-data-structures", 0),
    ("euhedral-hashing", 0),
    ("euhedral-hardware-utils", 0),
    ("euhedral-core", 1),
    ("euhedral-graph", 2),
    ("euhedral-io", 2),
    ("euhedral-memory", 3),
    ("euhedral-gpu", 4),
];

fn layer(name: &str) -> Option<u8> {
    LAYERS.iter().find(|(n, _)| *n == name).map(|(_, l)| *l)
}

/// Every `euhedral-*` name declared in any dependency table of `manifest`.
fn euhedral_deps(manifest: &str) -> Vec<String> {
    let mut deps = Vec::new();
    // The name from a dotted header (`[dependencies.euhedral-memory]`), if the current table is one.
    let mut dotted: Option<String> = None;
    let mut in_deps = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(header) = line.strip_prefix('[') {
            let header = header.trim_start_matches('[');
            let header = header.trim_end_matches(']').trim_end_matches(']').trim();
            dotted = None;
            in_deps = false;
            if let Some(at) = header.find("dependencies.") {
                // A dotted header names the dependency after `dependencies.`.
                let name = header[at + "dependencies.".len()..]
                    .trim()
                    .trim_matches('"');
                dotted = Some(name.to_owned());
                in_deps = true;
                if name.starts_with("euhedral-") {
                    deps.push(name.to_owned());
                }
            } else if header.ends_with("dependencies") {
                in_deps = true;
            }
            continue;
        }
        if in_deps && dotted.is_none() {
            let key = line
                .split(['=', '.'])
                .next()
                .unwrap_or_default()
                .trim()
                .trim_matches('"');
            if key.starts_with("euhedral-") && !deps.iter().any(|d| d == key) {
                deps.push(key.to_owned());
            }
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
