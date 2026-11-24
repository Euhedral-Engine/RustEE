//! The Rust toolchain is pinned in `mise.toml` and `rust-toolchain.toml`, which must name the same version and profile,
//! and twice in `MODULE.bazel` (`rust.toolchain` and the host tools), which must name that version.

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

/// The text inside the parentheses of the first call that starts with `opener` (for example `rust.toolchain(`).
fn call<'a>(module: &'a str, opener: &str) -> Option<&'a str> {
    module.split_once(opener)?.1.split(')').next()
}

/// The quoted value of the line in `call` that starts with `key`.
fn quoted(call: &str, key: &str) -> Option<String> {
    call.lines().find_map(|l| {
        let rest = l.trim().strip_prefix(key)?;
        Some(rest.split('"').nth(1)?.to_owned())
    })
}

/// The versions `MODULE.bazel` pins: the one entry of `versions = [...]` in `rust.toolchain(` and the `version` of
/// `host_tools(`, which `from_cargo` must name through `host_tools = "@rust_host_tools_pinned"`.
fn module_versions(module: &str) -> Result<(String, String), String> {
    let toolchain_call =
        call(module, "rust.toolchain(").ok_or("MODULE.bazel has no rust.toolchain( call")?;
    let list = toolchain_call
        .split_once("versions")
        .and_then(|(_, rest)| rest.split_once('[')?.1.split_once(']'))
        .ok_or("MODULE.bazel rust.toolchain has no versions = [\"...\"] line")?
        .0;
    let entries: Vec<&str> = list
        .split(',')
        .map(str::trim)
        .filter(|e| !e.is_empty())
        .collect();
    let [entry] = entries.as_slice() else {
        return Err(format!(
            "MODULE.bazel rust.toolchain versions must hold exactly one version, found {}",
            entries.len()
        ));
    };
    let toolchain = entry.trim_matches('"').to_owned();
    let host_call = call(module, ".host_tools(").ok_or("MODULE.bazel has no host_tools( call")?;
    let host = quoted(host_call, "version")
        .ok_or("MODULE.bazel has no host_tools version = \"...\" line")?;
    let name = quoted(host_call, "name").ok_or("MODULE.bazel host_tools has no name")?;
    let from_cargo =
        call(module, "crate.from_cargo(").ok_or("MODULE.bazel has no crate.from_cargo( call")?;
    if quoted(from_cargo, "host_tools") != Some(format!("@{name}")) {
        return Err(format!(
            "MODULE.bazel crate.from_cargo must set host_tools = \"@{name}\", the pinned host tools repo"
        ));
    }
    Ok((toolchain, host))
}

/// Compares the three files; `module` is `None` while `MODULE.bazel` is absent.
fn check(mise: &str, toolchain: &str, module: Option<&str>) -> Result<(), String> {
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
    if let Some(module) = module {
        let want = value_of(toolchain, "channel").ok_or("rust-toolchain.toml has no channel")?;
        let (toolchain_version, host_version) = module_versions(module)?;
        for (line, got) in [
            ("rust.toolchain versions", toolchain_version),
            ("host_tools version", host_version),
        ] {
            if got != want {
                return Err(format!(
                    "MODULE.bazel {line} {got} differs from rust-toolchain.toml channel {want}; \
                     edit both MODULE.bazel lines, then run `bazel mod deps --lockfile_mode=update` and commit MODULE.bazel.lock"
                ));
            }
        }
    }
    Ok(())
}

#[test]
fn the_toolchain_pins_agree() {
    let module = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../MODULE.bazel")
        .exists()
        .then(|| read("MODULE.bazel"));
    if let Err(e) = check(
        &read("mise.toml"),
        &read("rust-toolchain.toml"),
        module.as_deref(),
    ) {
        panic!("{e}");
    }
}

const MISE: &str = "[tools]\nrust = { version = \"1.99.1\", profile = \"default\" }\n";
const TOOLCHAIN: &str = "[toolchain]\nchannel = \"1.99.1\"\nprofile = \"default\"\n";
const MODULE: &str = "rust.toolchain(\n    edition = \"2024\",\n    versions = [\"1.99.1\"],\n)\n\
    rust_host_tools.host_tools(\n    name = \"x\",\n    version = \"1.99.1\",\n)\n\
    crate.from_cargo(\n    host_tools = \"@x\",\n)\n";

#[test]
fn a_differing_pin_is_reported() {
    assert_eq!(check(MISE, TOOLCHAIN, None), Ok(()));
    let other = TOOLCHAIN.replace("1.99.1", "1.99.2");
    assert!(check(MISE, &other, None).is_err());
}

#[test]
fn a_differing_module_line_is_reported_with_the_fix() {
    assert_eq!(check(MISE, TOOLCHAIN, Some(MODULE)), Ok(()));
    for (from, to) in [
        ("versions = [\"1.99.1\"]", "versions = [\"1.99.2\"]"),
        ("    version = \"1.99.1\"", "    version = \"1.99.2\""),
    ] {
        let err = check(MISE, TOOLCHAIN, Some(&MODULE.replace(from, to))).unwrap_err();
        assert!(err.contains("bazel mod deps"), "{err}");
    }
    assert!(check(MISE, TOOLCHAIN, Some("")).is_err());
    // Only the entry inside rust.toolchain counts, and there is exactly one.
    let outside = format!("versions = [\"9.9.9\"]\n{MODULE}");
    assert_eq!(check(MISE, TOOLCHAIN, Some(&outside)), Ok(()));
    let two = MODULE.replace("[\"1.99.1\"]", "[\"1.99.1\", \"1.99.2\"]");
    assert!(
        check(MISE, TOOLCHAIN, Some(&two))
            .unwrap_err()
            .contains("exactly one")
    );
    // from_cargo must use the pinned host tools repo.
    let unpinned = MODULE.replace("@x", "@rust_host_tools");
    assert!(
        check(MISE, TOOLCHAIN, Some(&unpinned))
            .unwrap_err()
            .contains("host_tools")
    );
}
