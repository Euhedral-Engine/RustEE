//! The Rust toolchain is pinned in `mise.toml` and `rust-toolchain.toml`; the two must name the same version and
//! profile.

#![allow(
    clippy::disallowed_methods,
    reason = "the test reads the workspace's pin files"
)]

use std::path::PathBuf;

/// The text after `key =` on the first line of `text` that sets `key`, with quotes removed.
fn value_of(text: &str, key: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let rest = line
            .trim()
            .strip_prefix(key)?
            .trim_start()
            .strip_prefix('=')?;
        Some(rest.trim().trim_matches('"').to_owned())
    })
}

/// The `field` of the inline table `rust = { ... }` in a `mise.toml`.
fn mise_field(mise: &str, field: &str) -> Option<String> {
    let line = mise.lines().find(|l| l.trim_start().starts_with("rust"))?;
    let table = line.split_once('{')?.1.split_once('}')?.0;
    table.split(',').find_map(|part| value_of(part, field))
}

fn read(name: &str) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::fs::read_to_string(root.join(name)).unwrap_or_else(|e| panic!("read {name}: {e}"))
}

fn check(mise: &str, toolchain: &str) -> Result<(), String> {
    for (mise_key, file_key) in [("version", "channel"), ("profile", "profile")] {
        let pinned =
            mise_field(mise, mise_key).ok_or(format!("mise.toml has no rust {mise_key}"))?;
        let file = value_of(toolchain, file_key)
            .ok_or(format!("rust-toolchain.toml has no {file_key}"))?;
        if pinned != file {
            return Err(format!(
                "mise.toml rust {mise_key} {pinned} differs from rust-toolchain.toml {file_key} {file}"
            ));
        }
    }
    Ok(())
}

#[test]
fn the_toolchain_pins_agree() {
    if let Err(e) = check(&read("mise.toml"), &read("rust-toolchain.toml")) {
        panic!("{e}");
    }
    // The Bazel pin (`MODULE.bazel`) is compared here once that file exists; until then there is nothing to compare.
}

#[test]
fn a_differing_pin_is_reported() {
    let mise = "[tools]\nrust = { version = \"1.99.1\", profile = \"default\" }\n";
    let same = "[toolchain]\nchannel = \"1.99.1\"\nprofile = \"default\"\n";
    assert_eq!(check(mise, same), Ok(()));
    let other = same.replace("1.99.1", "1.99.2");
    assert!(check(mise, &other).is_err());
}
