//! The workspace as the generator sees it: every member's targets, dependencies, features, lints and Bazel settings, read
//! from `cargo metadata --no-deps` and from the members' `Cargo.toml` (`[lints]`, which the metadata does not carry).

use std::collections::BTreeMap;

use crate::cfg::{self, Applies};
use crate::json::Json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepKind {
    Normal,
    Dev,
}

#[derive(Debug, Clone)]
pub struct Dep {
    pub name: String,
    pub kind: DepKind,
    pub applies: Applies,
    pub optional: bool,
    pub default_features: bool,
    pub features: Vec<String>,
    /// The member this path dependency names, by package name; `None` for a third-party crate.
    pub member: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TargetKind {
    Lib,
    Bin,
    Test,
    Example,
}

#[derive(Debug, Clone)]
pub struct Target {
    pub kind: TargetKind,
    pub name: String,
    /// The root file, relative to the package directory, with `/`.
    pub src: String,
    pub edition: String,
    pub required_features: Vec<String>,
    pub test: bool,
    pub doctest: bool,
}

/// One `[lints.<tool>]` entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lint {
    pub tool: String,
    pub name: String,
    pub level: String,
    pub priority: i64,
}

/// `[package.metadata.bazel]` for one test or binary.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Extras {
    /// `data = ["//:workspace_sources"]` and a package-relative `CARGO_MANIFEST_DIR`.
    pub workspace_sources: bool,
    /// `tags = ["cpu:N"]`.
    pub cpus: Option<u32>,
    /// More `data` labels; the target also gets the package-relative `CARGO_MANIFEST_DIR`.
    pub data: Vec<String>,
    /// The toolchain's cargo as `data`, its path in `CARGO`.
    pub cargo: bool,
    /// A target of this name in the root package that is an alias of this binary.
    pub root_alias: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Package {
    pub name: String,
    pub version: String,
    /// The package directory relative to the workspace root, with `/`.
    pub dir: String,
    pub features: BTreeMap<String, Vec<String>>,
    pub deps: Vec<Dep>,
    pub targets: Vec<Target>,
    pub lints: Vec<Lint>,
    /// The names `[lints.rust] unexpected_cfgs`'s `check-cfg` declares (`cfg(loom)` gives `loom`).
    pub check_cfg: Vec<String>,
    /// Package-relative files outside `src/` that a `#[path = "..."]` in `src/` names (`tests/common/scratch.rs`), filled
    /// by [`crate::load_workspace`].
    pub path_modules: Vec<String>,
    /// By (`test` or `bin`, target name).
    pub extras: BTreeMap<(String, String), Extras>,
}

impl Package {
    pub fn lib(&self) -> Option<&Target> {
        self.targets.iter().find(|t| t.kind == TargetKind::Lib)
    }

    /// The crate name of the package's library: the package name with `-` as `_`.
    pub fn crate_name(&self) -> String {
        self.name.replace('-', "_")
    }

    pub fn extras(&self, kind: &str, name: &str) -> Extras {
        self.extras
            .get(&(kind.to_owned(), name.to_owned()))
            .cloned()
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone)]
pub struct Workspace {
    /// Sorted by directory.
    pub packages: Vec<Package>,
}

impl Workspace {
    pub fn package(&self, name: &str) -> Option<&Package> {
        self.packages.iter().find(|p| p.name == name)
    }
}

/// `path` with `\` as `/` and without a trailing `/`.
fn slashes(path: &str) -> String {
    path.replace('\\', "/").trim_end_matches('/').to_owned()
}

/// `path` relative to `root`, both as Cargo prints them (either separator). An absolute path outside the root is an error:
/// no absolute path may reach a generated file.
pub fn relative(root: &str, path: &str) -> Result<String, String> {
    let root = slashes(root);
    let path = slashes(path);
    if path == root {
        return Ok(String::new());
    }
    path.strip_prefix(&format!("{root}/"))
        .map(str::to_owned)
        .ok_or_else(|| format!("{path} is outside the workspace root {root}"))
}

fn str_field<'a>(value: &'a Json, key: &str, what: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Json::as_str)
        .ok_or_else(|| format!("cargo metadata: {what} has no string `{key}`"))
}

fn str_list(value: Option<&Json>) -> Vec<String> {
    value
        .and_then(Json::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Json::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// Reads the workspace from `cargo metadata --no-deps --format-version 1` and each member's manifest text, which `manifest`
/// returns for a path relative to the root (`euhedral-core/Cargo.toml`).
pub fn load(
    metadata: &Json,
    manifest: &dyn Fn(&str) -> Result<String, String>,
) -> Result<Workspace, String> {
    let root = str_field(metadata, "workspace_root", "the workspace")?;
    let packages_json = metadata
        .get("packages")
        .and_then(Json::as_array)
        .ok_or("cargo metadata: no packages")?;
    let mut dirs = BTreeMap::new();
    for package in packages_json {
        let name = str_field(package, "name", "a package")?;
        let manifest_path = str_field(package, "manifest_path", name)?;
        let manifest_path = slashes(manifest_path);
        let package_dir = manifest_path.strip_suffix("/Cargo.toml").ok_or_else(|| {
            format!("package {name}: manifest {manifest_path} is not a Cargo.toml")
        })?;
        let dir = relative(root, package_dir)?;
        if dir.is_empty() {
            return Err(format!(
                "package {name} is the workspace root: only virtual workspaces are modelled"
            ));
        }
        dirs.insert(dir, name.to_owned());
    }
    let mut packages = Vec::new();
    for package in packages_json {
        packages.push(load_package(root, package, &dirs, manifest)?);
    }
    packages.sort_by(|a, b| a.dir.cmp(&b.dir));
    Ok(Workspace { packages })
}

fn load_package(
    root: &str,
    package: &Json,
    dirs: &BTreeMap<String, String>,
    manifest: &dyn Fn(&str) -> Result<String, String>,
) -> Result<Package, String> {
    let name = str_field(package, "name", "a package")?.to_owned();
    let version = str_field(package, "version", &name)?.to_owned();
    let dir = dirs
        .iter()
        .find(|(_, n)| **n == name)
        .map(|(d, _)| d.clone())
        .ok_or_else(|| format!("package {name} has no directory"))?;
    let mut features = BTreeMap::new();
    if let Some(map) = package.get("features").and_then(Json::as_object) {
        for (feature, entries) in map {
            features.insert(feature.clone(), str_list(Some(entries)));
        }
    }

    let mut deps = Vec::new();
    for dep in package
        .get("dependencies")
        .and_then(Json::as_array)
        .unwrap_or_default()
    {
        let dep_name = str_field(dep, "name", &name)?.to_owned();
        let what = format!("{name}'s dependency {dep_name}");
        let kind = match dep.get("kind").and_then(Json::as_str) {
            None => DepKind::Normal,
            Some("dev") => DepKind::Dev,
            Some(other) => {
                return Err(format!(
                    "{what}: {other}-dependencies are not modelled (no build script exists)"
                ));
            }
        };
        if dep.get("rename").is_some_and(|r| !r.is_null()) {
            return Err(format!(
                "{what} is renamed: renamed dependencies are not modelled"
            ));
        }
        let target = dep.get("target").and_then(Json::as_str);
        let applies = cfg::applies(target).map_err(|e| format!("{what}: {e}"))?;
        let member = match dep.get("path").and_then(Json::as_str) {
            Some(path) => {
                let dep_dir = relative(root, path)?;
                Some(dirs.get(&dep_dir).cloned().ok_or_else(|| {
                    format!("{what} is a path dependency outside the workspace's members")
                })?)
            }
            None => None,
        };
        if member.is_some() && applies != Applies::Always {
            return Err(format!(
                "{what} is a first-party dependency under [target.*]: not modelled"
            ));
        }
        deps.push(Dep {
            name: dep_name,
            kind,
            applies,
            optional: dep.get("optional").and_then(Json::as_bool).unwrap_or(false),
            default_features: dep
                .get("uses_default_features")
                .and_then(Json::as_bool)
                .unwrap_or(true),
            features: str_list(dep.get("features")),
            member,
        });
    }

    let mut targets = Vec::new();
    for target in package
        .get("targets")
        .and_then(Json::as_array)
        .unwrap_or_default()
    {
        let target_name = str_field(target, "name", &name)?.to_owned();
        let kinds = str_list(target.get("kind"));
        let what = format!("{name}'s target {target_name}");
        let kind = match kinds
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .as_slice()
        {
            ["lib"] => TargetKind::Lib,
            ["bin"] => TargetKind::Bin,
            ["test"] => TargetKind::Test,
            ["example"] => TargetKind::Example,
            other => {
                return Err(format!(
                    "{what} has kind {other:?}: only lib, bin, test and example targets are modelled \
                     (a build script, bench, proc-macro or other crate type is a generator change)"
                ));
            }
        };
        let crate_types = str_list(target.get("crate_types"));
        let expected = if kind == TargetKind::Lib {
            "lib"
        } else {
            "bin"
        };
        if crate_types != [expected] {
            return Err(format!(
                "{what} has crate types {crate_types:?}: not modelled"
            ));
        }
        let src = relative(
            &format!("{root}/{dir}"),
            str_field(target, "src_path", &what)?,
        )?;
        targets.push(Target {
            kind,
            name: target_name,
            src,
            edition: str_field(target, "edition", &what)?.to_owned(),
            required_features: str_list(target.get("required-features")),
            test: target.get("test").and_then(Json::as_bool).unwrap_or(true),
            doctest: target
                .get("doctest")
                .and_then(Json::as_bool)
                .unwrap_or(false),
        });
    }
    targets.sort_by(|a, b| (a.kind, &a.name).cmp(&(b.kind, &b.name)));
    if let Some(lib) = targets.iter().find(|t| t.kind == TargetKind::Lib)
        && lib.name != name.replace('-', "_")
    {
        return Err(format!(
            "{dir}/Cargo.toml names its library `{}`: a `[lib] name` other than the package's is not modelled",
            lib.name
        ));
    }

    let text = manifest(&format!("{dir}/Cargo.toml"))?;
    let (lints, check_cfg) = parse_lints(&text).map_err(|e| format!("{dir}/Cargo.toml: {e}"))?;
    if sets_harness(&text) {
        return Err(format!(
            "{dir}/Cargo.toml sets `harness`: targets without the test harness are not modelled"
        ));
    }
    let extras = parse_extras(&name, package.get("metadata"), &targets)?;
    Ok(Package {
        name,
        version,
        dir,
        features,
        deps,
        targets,
        lints,
        check_cfg,
        path_modules: Vec::new(),
        extras,
    })
}

fn parse_extras(
    package: &str,
    metadata: Option<&Json>,
    targets: &[Target],
) -> Result<BTreeMap<(String, String), Extras>, String> {
    let mut out = BTreeMap::new();
    let Some(bazel) = metadata.and_then(|m| m.get("bazel")) else {
        return Ok(out);
    };
    let what = format!("{package}'s [package.metadata.bazel]");
    let table = bazel
        .as_object()
        .ok_or_else(|| format!("{what} is not a table"))?;
    for (section, entries) in table {
        let kind = match section.as_str() {
            "test" => TargetKind::Test,
            "bin" => TargetKind::Bin,
            other => return Err(format!("{what}: unknown key `{other}` (test, bin)")),
        };
        for (target, settings) in entries
            .as_object()
            .ok_or_else(|| format!("{what}.{section} is not a table"))?
        {
            let here = format!("{what}.{section}.{target}");
            if !targets.iter().any(|t| t.kind == kind && t.name == *target) {
                return Err(format!(
                    "{here}: {package} has no {section} target `{target}`"
                ));
            }
            let mut extras = Extras::default();
            for (key, value) in settings
                .as_object()
                .ok_or_else(|| format!("{here} is not a table"))?
            {
                let bad = || format!("{here}.{key} has the wrong type");
                match key.as_str() {
                    "workspace_sources" => {
                        extras.workspace_sources = value.as_bool().ok_or_else(bad)?
                    }
                    "cargo" => extras.cargo = value.as_bool().ok_or_else(bad)?,
                    "cpus" => {
                        let Json::Number(n) = value else {
                            return Err(bad());
                        };
                        extras.cpus = Some(n.parse().map_err(|_| bad())?);
                    }
                    "data" => {
                        let items = value.as_array().ok_or_else(bad)?;
                        for item in items {
                            extras.data.push(item.as_str().ok_or_else(bad)?.to_owned());
                        }
                    }
                    "root_alias" if kind == TargetKind::Bin => {
                        extras.root_alias = Some(value.as_str().ok_or_else(bad)?.to_owned());
                    }
                    other => return Err(format!("{here}: unknown key `{other}`")),
                }
            }
            out.insert((section.clone(), target.clone()), extras);
        }
    }
    Ok(out)
}

/// Whether a target table of the manifest (`[lib]`, `[[bin]]`, `[[test]]`, `[[bench]]`, `[[example]]`) has a `harness`
/// key, on its own line or in an inline table of a target array.
fn sets_harness(text: &str) -> bool {
    let mut in_target = false;
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.starts_with('[') {
            let header = line.trim_matches(|c| c == '[' || c == ']').trim();
            in_target = matches!(header, "lib" | "bin" | "test" | "bench" | "example");
            continue;
        }
        if let Some((key, value)) = line.split_once('=') {
            let key = key.trim().trim_matches('"');
            if in_target && key == "harness" {
                return true;
            }
            // `test = [{ name = "t", harness = false }]` and the like.
            if matches!(key, "bin" | "test" | "bench" | "example")
                && value.split(['{', ',', '}']).any(|part| {
                    part.split_once('=')
                        .is_some_and(|(k, _)| k.trim() == "harness")
                })
            {
                return true;
            }
        }
    }
    false
}

/// The quoted strings of a TOML inline array, `['cfg(loom)', "cfg(x)"]`.
fn quoted_strings(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(['"', '\'']) {
        let quote = rest.as_bytes()[start] as char;
        let after = &rest[start + 1..];
        let Some(end) = after.find(quote) else { break };
        out.push(after[..end].to_owned());
        rest = &after[end + 1..];
    }
    out
}

/// The `[lints.rust]`, `[lints.clippy]` and `[lints.rustdoc]` tables, one entry per line (`name = "level"` or
/// `name = { level = "...", priority = N, check-cfg = [...] }`). Anything else in a lints table is an error.
pub fn parse_lints(text: &str) -> Result<(Vec<Lint>, Vec<String>), String> {
    let mut lints = Vec::new();
    let mut check_cfg = Vec::new();
    let mut tool: Option<String> = None;
    for line in text.lines() {
        let line = line.split(" #").next().unwrap_or("").trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            let header = line.trim_matches(|c| c == '[' || c == ']').trim();
            tool = None;
            if header == "lints" || header == "workspace.lints" {
                return Err(format!(
                    "[{header}] is not modelled; use [lints.<tool>] tables"
                ));
            }
            if let Some(name) = header.strip_prefix("lints.") {
                match name {
                    "rust" | "clippy" | "rustdoc" => tool = Some(name.to_owned()),
                    other => return Err(format!("[lints.{other}] is not modelled")),
                }
            }
            continue;
        }
        let Some(tool) = &tool else { continue };
        let (name, value) = line
            .split_once('=')
            .ok_or_else(|| format!("[lints.{tool}]: cannot read `{line}`"))?;
        let name = name.trim().trim_matches('"').to_owned();
        if name.is_empty()
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            // A dotted key (`x.level = "warn"`) is a table, not a lint.
            return Err(format!("[lints.{tool}]: `{name}` is not a lint name"));
        }
        let value = value.trim();
        let (level, priority) = if let Some(table) =
            value.strip_prefix('{').and_then(|v| v.strip_suffix('}'))
        {
            let mut level = None;
            let mut priority = 0;
            // check-cfg holds commas inside its array: take it out first.
            let (plain, cfgs) = match table.split_once("check-cfg") {
                Some((before, after)) => {
                    let array = after
                        .split_once('[')
                        .and_then(|(_, a)| a.split_once(']'))
                        .ok_or_else(|| format!("[lints.{tool}] {name}: cannot read check-cfg"))?;
                    (format!("{before}{}", array.1), Some(array.0.to_owned()))
                }
                None => (table.to_owned(), None),
            };
            if let Some(cfgs) = cfgs {
                if name != "unexpected_cfgs" {
                    return Err(format!(
                        "[lints.{tool}] {name}: check-cfg outside unexpected_cfgs"
                    ));
                }
                for cfg in quoted_strings(&cfgs) {
                    let inner = cfg
                        .strip_prefix("cfg(")
                        .and_then(|c| c.strip_suffix(')'))
                        .filter(|c| c.chars().all(|ch| ch.is_alphanumeric() || ch == '_'))
                        .ok_or_else(|| {
                            format!("check-cfg `{cfg}`: only names without values are modelled")
                        })?;
                    check_cfg.push(inner.to_owned());
                }
            }
            for part in plain.split(',') {
                let part = part.trim();
                if part.is_empty() {
                    continue;
                }
                let (key, value) = part
                    .split_once('=')
                    .ok_or_else(|| format!("[lints.{tool}] {name}: cannot read `{part}`"))?;
                match key.trim() {
                    "level" => level = Some(value.trim().trim_matches('"').to_owned()),
                    "priority" => {
                        priority = value
                            .trim()
                            .parse()
                            .map_err(|_| format!("[lints.{tool}] {name}: bad priority"))?;
                    }
                    other => return Err(format!("[lints.{tool}] {name}: unknown key `{other}`")),
                }
            }
            (
                level.ok_or_else(|| format!("[lints.{tool}] {name}: no level"))?,
                priority,
            )
        } else {
            (value.trim_matches('"').to_owned(), 0)
        };
        if !matches!(
            level.as_str(),
            "allow" | "warn" | "deny" | "forbid" | "force-warn"
        ) {
            return Err(format!("[lints.{tool}] {name}: unknown level `{level}`"));
        }
        lints.push(Lint {
            tool: tool.clone(),
            name,
            level,
            priority,
        });
    }
    lints.sort_by(|a, b| (a.priority, &a.tool, &a.name).cmp(&(b.priority, &b.tool, &b.name)));
    check_cfg.sort();
    check_cfg.dedup();
    Ok((lints, check_cfg))
}

/// The module files a source file's `#[path = "..."]` attributes name, relative to the package: `file` is the package-relative
/// path of the source, and each path is relative to the file's directory. A path that leaves the package is an error.
pub fn path_attributes(file: &str, text: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.trim().strip_prefix("#[path") else {
            continue;
        };
        let target = rest
            .trim_start()
            .strip_prefix('=')
            .and_then(|r| r.split_once(']').map(|(inside, _)| inside))
            .and_then(|r| r.trim().strip_prefix('"'))
            .and_then(|r| r.strip_suffix('"'))
            .ok_or_else(|| format!("{file}: unreadable path attribute `{}`", line.trim()))?;
        let mut parts: Vec<&str> = file.split('/').collect();
        parts.pop();
        let target = target.replace('\\', "/");
        for part in target.split('/') {
            match part {
                "" | "." => {}
                ".." => {
                    parts
                        .pop()
                        .ok_or_else(|| format!("{file}: path `{target}` leaves the package"))?;
                }
                other => parts.push(other),
            }
        }
        out.push(parts.join("/"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lint_tables_are_read() {
        let text = "[package]\nname = \"x\"\n\n[lints.rust]\nunsafe_op_in_unsafe_fn = \"deny\"\n\
            unexpected_cfgs = { level = \"warn\", check-cfg = ['cfg(loom)', \"cfg(x_y)\"] }\n\n\
            [lints.clippy]\ntodo = { level = \"warn\", priority = -1 }\n\n[features]\na = []\n";
        let (lints, cfgs) = parse_lints(text).unwrap();
        let got: Vec<_> = lints
            .iter()
            .map(|l| {
                (
                    l.tool.as_str(),
                    l.name.as_str(),
                    l.level.as_str(),
                    l.priority,
                )
            })
            .collect();
        assert_eq!(
            got,
            [
                ("clippy", "todo", "warn", -1),
                ("rust", "unexpected_cfgs", "warn", 0),
                ("rust", "unsafe_op_in_unsafe_fn", "deny", 0),
            ]
        );
        assert_eq!(cfgs, ["loom", "x_y"]);
    }

    #[test]
    fn unmodelled_lint_forms_are_refused() {
        for text in [
            "[lints]\nworkspace = true\n",
            "[lints.rust]\nunexpected_cfgs = { level = \"warn\", check-cfg = ['cfg(x, values(\"a\"))'] }\n",
            "[lints.rust]\nx = \"loud\"\n",
            "[lints.rust.unexpected_cfgs]\nlevel = \"warn\"\n",
            "[lints.rust]\nunexpected_cfgs.level = \"warn\"\n",
        ] {
            assert!(parse_lints(text).is_err(), "{text} was accepted");
        }
    }

    #[test]
    fn only_a_target_tables_harness_key_is_refused() {
        for text in [
            "[[test]]\nname = \"t\"\nharness = false\n",
            "[lib]\nharness = false\n",
            "test = [{ name = \"t\", harness = false }]\n",
        ] {
            assert!(sets_harness(text), "{text} was accepted");
        }
        for text in [
            "[features]\nharness = []\n",
            "[package]\n# harness = false is not used here\nname = \"x\"\n",
            "[dependencies]\nharness = \"1\"\n",
        ] {
            assert!(!sets_harness(text), "{text} was refused");
        }
    }

    #[test]
    fn path_attributes_resolve_from_the_file() {
        let text = "#[cfg(test)]\n#[path = \"../tests/common/scratch.rs\"]\nmod scratch;\n  #[path=\"x/y.rs\"] mod y;\n";
        assert_eq!(
            path_attributes("src/lib.rs", text),
            Ok(vec![
                "tests/common/scratch.rs".to_owned(),
                "src/x/y.rs".to_owned()
            ])
        );
        assert!(path_attributes("src/lib.rs", "#[path = \"../../other/x.rs\"]\n").is_err());
    }

    #[test]
    fn paths_are_relative_with_forward_slashes() {
        assert_eq!(
            relative("/w/r", "/w/r/euhedral-core/src/lib.rs"),
            Ok("euhedral-core/src/lib.rs".into())
        );
        assert_eq!(
            relative(r"C:\w\r", r"C:\w\r\tools\bazel-gen\src\lib.rs"),
            Ok("tools/bazel-gen/src/lib.rs".into())
        );
        assert!(relative("/w/r", "/w/rx/src/lib.rs").is_err());
    }
}
