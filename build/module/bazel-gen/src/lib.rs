#![doc = include_str!("../README.md")]

pub mod cfg;
pub mod config;
pub mod emit;
pub mod json;
pub mod model;
pub mod sha256;
pub mod starlark;
pub mod worlds;

#[cfg(test)]
mod fixture_tests;

#[cfg(test)]
mod rename_build_tests;

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use crate::model::Workspace;

/// Reads the configuration: the file `path` names, else the root's `bazel_gen.toml` when there is one, else the defaults.
/// Also gives the file's root-relative path when it is inside the root.
pub fn load_config(
    root: &Path,
    path: Option<&Path>,
) -> Result<(config::Config, Option<String>), String> {
    let path = match path {
        Some(path) => path.to_owned(),
        None if root.join(config::FILE).is_file() => root.join(config::FILE),
        None => return Ok((config::Config::default(), None)),
    };
    let text =
        std::fs::read_to_string(&path).map_err(|e| format!("reading {}: {e}", path.display()))?;
    let config = config::Config::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let inside = path
        .strip_prefix(root)
        .ok()
        .map(|p| p.to_string_lossy().replace('\\', "/"));
    Ok((config, inside))
}

/// Reads the workspace at `root` with `cargo metadata --no-deps` (run with `cargo`, and `CARGO_HOME` when given), the
/// members' manifests and the configuration ([`load_config`]).
pub fn load_workspace(
    root: &Path,
    cargo: &str,
    cargo_home: Option<&Path>,
    config_path: Option<&Path>,
) -> Result<Workspace, String> {
    let (config, config_file) = load_config(root, config_path)?;
    let mut command = Command::new(cargo);
    if let Some(home) = cargo_home {
        command.env("CARGO_HOME", home);
    }
    let output = command
        .args([
            "metadata",
            "--no-deps",
            "--format-version",
            "1",
            "--offline",
            "--manifest-path",
        ])
        .arg(root.join("Cargo.toml"))
        .output()
        .map_err(|e| format!("running `{cargo} metadata`: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "`cargo metadata` failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let text =
        String::from_utf8(output.stdout).map_err(|_| "cargo metadata printed invalid UTF-8")?;
    let metadata = json::parse(&text)?;
    let mut ws = model::load(
        &metadata,
        &|path| {
            std::fs::read_to_string(root.join(path)).map_err(|e| format!("reading {path}: {e}"))
        },
        config,
    )?;
    ws.config_file = config_file;
    find_path_modules(&mut ws, root)?;
    Ok(ws)
}

/// Fills each member's [`model::Package::path_modules`] (files of the package outside `src/` that `src/` names) and
/// [`model::Package::foreign_srcs`], and the workspace's [`model::Workspace::exports`], from the `#[path = "..."]`
/// attributes and literal includes (`include_str!`, `include_bytes!`, `include!`) of the member's `src/`, `tests/`,
/// `examples/`, `benches/` and `build.rs` files under `root` and of the foreign source files they name.
/// A file outside the workspace, under a `.bazelignore` entry, or in a Bazel package the generator does not write is
/// an error naming it.
pub fn find_path_modules(ws: &mut Workspace, root: &Path) -> Result<(), String> {
    let ignored = bazelignore(root)?;
    let member_dirs: Vec<String> = ws.packages.iter().map(|p| p.dir.clone()).collect();
    let mut exports: std::collections::BTreeMap<
        String,
        std::collections::BTreeMap<String, std::collections::BTreeSet<String>>,
    > = Default::default();
    for package in &mut ws.packages {
        let mut found = Vec::new();
        let mut labels = Vec::new();
        let mut data = Vec::new();
        let mut local: std::collections::BTreeMap<String, Vec<String>> = Default::default();
        let mut local_data: std::collections::BTreeMap<String, Vec<String>> = Default::default();
        let mut build_data = Vec::new();
        let mut files = Vec::new();
        for dir in ["src", "tests", "examples", "benches"] {
            rust_files(&root.join(&package.dir), dir, &mut files)?;
        }
        if root.join(&package.dir).join("build.rs").is_file() {
            files.push("build.rs".to_owned());
        }
        // Foreign files are read too, for the attributes they hold.
        let mut queue: Vec<String> = files
            .iter()
            .map(|f| format!("{}/{f}", package.dir))
            .collect();
        let mut seen: std::collections::BTreeSet<String> = queue.iter().cloned().collect();
        // The directory (`src`, `tests`, ...) whose targets a file belongs to, for foreign files that of their includer.
        let mut origin: std::collections::BTreeMap<String, String> = Default::default();
        let own = format!("{}/", package.dir);
        let build_rs = format!("{own}build.rs");
        let own_src = format!("{own}src/");
        while let Some(file) = queue.pop() {
            let text = std::fs::read_to_string(root.join(&file))
                .map_err(|e| format!("reading {file}: {e}"))?;
            let in_build = file == build_rs;
            let dir_key = file
                .strip_prefix(&own)
                .and_then(|rest| rest.split_once('/'))
                .map(|(d, _)| d.to_owned())
                .or_else(|| origin.get(&file).cloned())
                .unwrap_or_else(|| "src".to_owned());
            let mut named: Vec<(String, bool, &str)> = model::path_attributes(&file, &text)?
                .into_iter()
                .map(|t| (t, true, "`#[path]`"))
                .collect();
            for (target, source) in model::include_files(&file, &package.dir, &text)? {
                named.push((target, source, "an include"));
            }
            for (target, source, how) in named {
                if let Some(inside) = target.strip_prefix(&own) {
                    if source {
                        if file.starts_with(&own_src) && !inside.starts_with("src/") {
                            found.push(inside.to_owned());
                        } else if !file.starts_with(&own_src)
                            && !in_build
                            && !inside.starts_with(&format!("{dir_key}/"))
                        {
                            local
                                .entry(dir_key.clone())
                                .or_default()
                                .push(inside.to_owned());
                        }
                    } else {
                        if !root.join(&target).is_file() {
                            return Err(format!(
                                "{file}: {how} names {target}, which does not exist"
                            ));
                        }
                        if in_build {
                            build_data.push(inside.to_owned());
                        } else if !inside.starts_with(&format!("{dir_key}/")) {
                            local_data
                                .entry(dir_key.clone())
                                .or_default()
                                .push(inside.to_owned());
                        }
                    }
                    continue;
                }
                let (owner, rel) = owning_package(root, &member_dirs, &ignored, &target)
                    .map_err(|e| format!("{file}: {e}"))?;
                if !root.join(&target).is_file() {
                    return Err(format!(
                        "{file}: {how} names {target}, which does not exist"
                    ));
                }
                let label = format!("//{owner}:{rel}");
                if source {
                    labels.push(label);
                    if seen.insert(target.clone()) {
                        origin.insert(target.clone(), dir_key.clone());
                        queue.push(target.clone());
                    }
                } else if in_build {
                    build_data.push(label);
                } else {
                    data.push(label);
                }
                exports
                    .entry(owner)
                    .or_default()
                    .entry(rel)
                    .or_default()
                    .insert(package.dir.clone());
            }
        }
        for list in [&mut found, &mut labels, &mut data, &mut build_data] {
            list.sort();
            list.dedup();
        }
        for list in local_data.values_mut().chain(local.values_mut()) {
            list.sort();
            list.dedup();
        }
        package.path_modules = found;
        package.foreign_data = data;
        package.local_srcs = local;
        package.foreign_srcs = labels;
        package.local_data = local_data;
        package.build_data = build_data;
    }
    ws.exports = exports;
    Ok(())
}

/// The `.bazelignore` entries, without trailing slashes.
fn bazelignore(root: &Path) -> Result<Vec<String>, String> {
    match std::fs::read_to_string(root.join(".bazelignore")) {
        Ok(text) => Ok(text
            .lines()
            .map(|line| line.trim().trim_end_matches('/').to_owned())
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(format!("reading .bazelignore: {e}")),
    }
}

/// The Bazel package directory (`""` for the root) that owns the root-relative `file`, and the file's path inside it:
/// the nearest member or, when no member encloses it, the nearest directory above with a BUILD file; the root package
/// when none has one. A file under an ignored directory, or in a package the generator does not write, is an error.
fn owning_package(
    root: &Path,
    members: &[String],
    ignored: &[String],
    file: &str,
) -> Result<(String, String), String> {
    if let Some(entry) = ignored
        .iter()
        .find(|e| file == e.as_str() || file.starts_with(&format!("{e}/")))
    {
        return Err(format!(
            "`#[path]` names {file}, under the `.bazelignore` entry `{entry}`, so Bazel cannot see it: narrow `.bazelignore` (list the other subdirectories of `{entry}` instead of `{entry}`)"
        ));
    }
    let mut dir = file.rsplit_once('/').map_or("", |(d, _)| d).to_owned();
    loop {
        if members.contains(&dir) {
            break;
        }
        if !dir.is_empty()
            && (root.join(&dir).join("BUILD.bazel").is_file()
                || root.join(&dir).join("BUILD").is_file())
        {
            return Err(format!(
                "`#[path]` names {file}, in the Bazel package `{dir}`, which is not a workspace member: the generator writes no BUILD file there"
            ));
        }
        match dir.rsplit_once('/') {
            Some((parent, _)) => dir = parent.to_owned(),
            None if dir.is_empty() => break,
            None => dir = String::new(),
        }
    }
    let rel = if dir.is_empty() {
        file.to_owned()
    } else {
        file[dir.len() + 1..].to_owned()
    };
    Ok((dir, rel))
}

/// Every `.rs` file under `dir` (package-relative, `/`-separated), in `out`.
fn rust_files(package: &Path, dir: &str, out: &mut Vec<String>) -> Result<(), String> {
    let entries = match std::fs::read_dir(package.join(dir)) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(format!("listing {dir}: {e}")),
    };
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let path = format!("{dir}/{name}");
        // Runfiles trees hold symlinks: follow them.
        let meta = std::fs::metadata(entry.path()).map_err(|e| format!("{path}: {e}"))?;
        if meta.is_dir() {
            rust_files(package, &path, out)?;
        } else if name.ends_with(".rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// The generated files that differ from what `read` returns for their path (a missing file differs), sorted.
pub fn stale(
    expected: &BTreeMap<String, String>,
    read: &dyn Fn(&str) -> Option<String>,
) -> Vec<String> {
    expected
        .iter()
        .filter(|(path, content)| read(path).as_deref() != Some(content.as_str()))
        .map(|(path, _)| path.clone())
        .collect()
}

/// The BUILD files on disk that the generator does not write: one left by a removed member, or a hand-added one, which
/// would cut its parent package's globs. Sorted.
pub fn extra_build_files(expected: &BTreeMap<String, String>, on_disk: &[String]) -> Vec<String> {
    let mut extra: Vec<String> = on_disk
        .iter()
        .filter(|path| !expected.contains_key(*path))
        .cloned()
        .collect();
    extra.sort();
    extra
}

/// Every `BUILD.bazel` or `BUILD` file under `root` (root-relative, `/`-separated) that Bazel would read as a package:
/// the paths `.bazelignore` names, `.git`, every `target` directory next to a `Cargo.toml` (Cargo's output; a
/// `target` source directory is a package) and symlinked directories (the `bazel-*` output links) are skipped. A
/// directory that cannot be listed is an error naming it: skipping it could hide a BUILD file the generator should own.
pub fn build_files_on_disk(root: &Path) -> Result<Vec<String>, String> {
    let ignored: Vec<String> = match std::fs::read_to_string(root.join(".bazelignore")) {
        Ok(text) => text
            .lines()
            .map(|line| line.trim().trim_end_matches('/').to_owned())
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .collect(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(e) => return Err(format!("reading .bazelignore: {e}")),
    };
    let mut out = Vec::new();
    let mut dirs = vec![String::new()];
    while let Some(dir) = dirs.pop() {
        let entries =
            std::fs::read_dir(root.join(&dir)).map_err(|e| {
                let shown = if dir.is_empty() { "." } else { dir.as_str() };
                format!(
                    "cannot read directory `{shown}` ({e}): fix its permissions or list it in .bazelignore, since a BUILD file inside it could not be checked"
                )
            })?;
        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = if dir.is_empty() {
                name.clone()
            } else {
                format!("{dir}/{name}")
            };
            let kind = entry.file_type().map_err(|e| format!("{path}: {e}"))?;
            if kind.is_dir() {
                let cargo_output = name == "target" && root.join(&dir).join("Cargo.toml").is_file();
                if name != ".git" && !cargo_output && !ignored.contains(&path) {
                    dirs.push(path);
                }
            } else if (name == "BUILD.bazel" || name == "BUILD")
                && (kind.is_file() || std::fs::metadata(entry.path()).is_ok_and(|m| m.is_file()))
            {
                out.push(path);
            }
        }
    }
    out.sort();
    Ok(out)
}

/// The manifest files `MODULE.bazel.lock` records a digest of: the root's `Cargo.toml` and `Cargo.lock` and every
/// member's `Cargo.toml`.
pub fn locked_inputs(ws: &Workspace) -> Vec<String> {
    let mut paths = vec!["Cargo.lock".to_owned(), "Cargo.toml".to_owned()];
    paths.extend(ws.packages.iter().map(|p| format!("{}/Cargo.toml", p.dir)));
    paths
}

/// What is wrong with `MODULE.bazel.lock`'s record of the workspace manifests: a recorded digest that differs from the file
/// `read` returns, or a path in `required` with no record. Empty when the lock is current.
pub fn lock_problems(
    lock: &str,
    read: &dyn Fn(&str) -> Option<Vec<u8>>,
    required: &[String],
) -> Vec<String> {
    let mut recorded = BTreeMap::new();
    for line in lock.lines() {
        let Some(rest) = line
            .trim()
            .trim_start_matches('"')
            .strip_prefix("FILE:@@//")
        else {
            continue;
        };
        let rest = rest.trim_end_matches(',').trim_end_matches('"');
        if let Some((path, digest)) = rest.split_once(' ') {
            recorded.insert(path.to_owned(), digest.to_owned());
        }
    }
    let mut problems = Vec::new();
    for (path, digest) in &recorded {
        match read(path) {
            None => problems.push(format!("{path} is recorded but missing")),
            Some(bytes) if sha256::hex_digest(&bytes) != *digest => {
                problems.push(format!("{path} changed since the lock was written"))
            }
            Some(_) => {}
        }
    }
    for path in required {
        if !recorded.contains_key(path) {
            problems.push(format!("{path} is not recorded"));
        }
    }
    problems
}

/// The `name version` of every package in `Cargo.lock`, whatever its source (registry, git or path).
pub fn cargo_lock_packages(cargo_lock: &str) -> Vec<String> {
    let mut found = Vec::new();
    let (mut name, mut version) = (None, None);
    let mut close = |name: &mut Option<String>, version: &mut Option<String>| {
        if let (Some(n), Some(v)) = (name.take(), version.take()) {
            found.push(format!("{n} {v}"));
        }
    };
    for line in cargo_lock.lines() {
        if line.starts_with("[[package]]") {
            close(&mut name, &mut version);
        } else if let Some(v) = line.strip_prefix("name = ") {
            name = Some(v.trim_matches('"').to_owned());
        } else if let Some(v) = line.strip_prefix("version = ") {
            version = Some(v.trim_matches('"').to_owned());
        }
    }
    close(&mut name, &mut version);
    found.sort();
    found.dedup();
    found
}

/// What is wrong with crate_universe's own lock (`lockfile =` of `crate.from_cargo`, `cargo-bazel-lock.json`) next to
/// `Cargo.lock`: a package of `Cargo.lock` the lock has no crate for, or a crate of the lock that `Cargo.lock` lacks, the
/// workspace members aside on both sides. With such a lock `MODULE.bazel.lock` records none of the manifests and
/// crate_universe's digest does not cover `Cargo.lock`, so a `cargo update` without a repin would otherwise build the old
/// versions without a word. Empty when they agree.
pub fn crate_lock_problems(cargo_lock: &str, crate_lock: &str) -> Result<Vec<String>, String> {
    let lock = json::parse(crate_lock)?;
    let keys = |field: &str| -> Result<Vec<String>, String> {
        let object = lock
            .get(field)
            .and_then(json::Json::as_object)
            .ok_or_else(|| format!("the crate lock has no `{field}` object"))?;
        Ok(object.keys().cloned().collect())
    };
    let members = keys("workspace_members")?;
    let locked: Vec<String> = keys("crates")?
        .into_iter()
        .filter(|key| !members.contains(key))
        .collect();
    let wanted: Vec<String> = cargo_lock_packages(cargo_lock)
        .into_iter()
        .filter(|key| !members.contains(key))
        .collect();
    let mut problems = Vec::new();
    for package in wanted.iter().filter(|p| !locked.contains(p)) {
        problems.push(format!(
            "{package} is in Cargo.lock but not in the crate lock"
        ));
    }
    for package in locked.iter().filter(|p| !wanted.contains(p)) {
        problems.push(format!(
            "{package} is in the crate lock but not in Cargo.lock"
        ));
    }
    Ok(problems)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_changed_or_missing_file_is_stale() {
        let expected: BTreeMap<String, String> = [
            ("a/BUILD.bazel", "x\n"),
            ("BUILD.bazel", "y\n"),
            ("b/BUILD.bazel", "z\n"),
        ]
        .into_iter()
        .map(|(p, c)| (p.to_owned(), c.to_owned()))
        .collect();
        let on_disk = |path: &str| match path {
            "a/BUILD.bazel" => Some("x\n".to_owned()),
            "BUILD.bazel" => Some("y changed\n".to_owned()),
            _ => None,
        };
        assert_eq!(stale(&expected, &on_disk), ["BUILD.bazel", "b/BUILD.bazel"]);
        assert!(stale(&expected, &|p| expected.get(p).cloned()).is_empty());
    }

    #[test]
    fn a_build_file_the_generator_does_not_write_is_extra() {
        let expected: BTreeMap<String, String> = [("BUILD.bazel", ""), ("a/BUILD.bazel", "")]
            .into_iter()
            .map(|(p, c)| (p.to_owned(), c.to_owned()))
            .collect();
        let on_disk = [
            "BUILD.bazel".to_owned(),
            "removed/BUILD.bazel".to_owned(),
            "a/src/BUILD".to_owned(),
            "a/BUILD.bazel".to_owned(),
        ];
        assert_eq!(
            extra_build_files(&expected, &on_disk),
            ["a/src/BUILD", "removed/BUILD.bazel"]
        );
    }

    #[test]
    fn the_lock_check_names_changed_and_unrecorded_manifests() {
        let digest = sha256::hex_digest(b"[package]\n");
        let lock = format!(
            "  \"FILE:@@//a/Cargo.toml {digest}\",\n  \"FILE:@@//b/Cargo.toml {digest}\"\n  \"FILE:@@rules_rust+//MODULE.bazel 00\"\n"
        );
        let files = |path: &str| match path {
            "a/Cargo.toml" => Some(b"[package]\n".to_vec()),
            "b/Cargo.toml" => Some(b"[package]\nname = \"b\"\n".to_vec()),
            _ => None,
        };
        let required = ["a/Cargo.toml".to_owned(), "c/Cargo.toml".to_owned()];
        assert_eq!(
            lock_problems(&lock, &files, &required),
            [
                "b/Cargo.toml changed since the lock was written",
                "c/Cargo.toml is not recorded"
            ]
        );
        assert!(
            lock_problems(&lock, &|_| Some(b"[package]\n".to_vec()), &required[..1]).is_empty()
        );
    }

    #[test]
    fn the_crate_lock_check_names_a_package_only_one_lock_has() {
        // a from a registry, g from git (as a git dependency of another module is), m the workspace member.
        let cargo_lock = "version = 4\n\n[[package]]\nname = \"a\"\nversion = \"1.0.0\"\nsource = \"registry+x\"\n\n\
            [[package]]\nname = \"g\"\nversion = \"0.1.0\"\nsource = \"git+https://x#1\"\n\n\
            [[package]]\nname = \"m\"\nversion = \"0.1.0\"\n";
        let lock = |crates: &str| {
            format!(r#"{{"crates":{{{crates}}},"workspace_members":{{"m 0.1.0":"m"}}}}"#)
        };
        let agreeing = lock(r#""a 1.0.0":{},"g 0.1.0":{},"m 0.1.0":{}"#);
        assert!(
            crate_lock_problems(cargo_lock, &agreeing)
                .unwrap()
                .is_empty()
        );
        let behind = lock(r#""a 0.9.0":{},"g 0.1.0":{},"m 0.1.0":{}"#);
        assert_eq!(
            crate_lock_problems(cargo_lock, &behind).unwrap(),
            [
                "a 1.0.0 is in Cargo.lock but not in the crate lock",
                "a 0.9.0 is in the crate lock but not in Cargo.lock"
            ]
        );
        assert!(crate_lock_problems(cargo_lock, "{}").is_err());
    }
}
