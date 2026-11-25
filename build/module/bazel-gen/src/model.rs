//! The workspace as the generator sees it: every member's targets, dependencies, features, lints and Bazel settings, read
//! from `cargo metadata --no-deps` and from the members' `Cargo.toml` (`[lints]`, which the metadata does not carry).

use std::collections::{BTreeMap, BTreeSet};

use crate::cfg::{self, Applies};
use crate::config::Config;
use crate::json::Json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepKind {
    Normal,
    Dev,
    /// `[build-dependencies]`: linked into the package's build script only.
    Build,
}

#[derive(Debug, Clone)]
pub struct Dep {
    /// The package's name, which names its target (`@crates//:name`, the member, the external crate).
    pub name: String,
    /// The key the manifest gives it when that differs from the package name (`k = { package = "name" }`): features
    /// (`dep:k`, `k/f`) and the crate's code (`k` with `-` as `_`) refer to it by this.
    pub rename: Option<String>,
    pub kind: DepKind,
    pub applies: Applies,
    pub optional: bool,
    pub default_features: bool,
    pub features: Vec<String>,
    /// The member this path dependency names, by package name; `None` for a third-party crate.
    pub member: Option<String>,
}

impl Dep {
    /// The dependency's key in the manifest: the rename, else the package name.
    pub fn key(&self) -> &str {
        self.rename.as_deref().unwrap_or(&self.name)
    }

    /// The name the crate's code uses for it (the key with `-` as `_`), when that differs from the package's crate name
    /// (`None`: no alias is needed).
    pub fn alias(&self) -> Option<String> {
        let rename = self.rename.as_ref()?;
        let alias = rename.replace('-', "_");
        (alias != self.name.replace('-', "_")).then_some(alias)
    }
}

/// A package's build script (`build.rs`).
#[derive(Debug, Clone)]
pub struct BuildScript {
    /// The root file relative to the package directory, always `build.rs`.
    pub src: String,
    pub edition: String,
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
    /// The build script, when the package has one.
    pub build_script: Option<BuildScript>,
    /// The manifest's `links` key.
    pub links: Option<String>,
    pub lints: Vec<Lint>,
    /// The names `[lints.rust] unexpected_cfgs`'s `check-cfg` declares (`cfg(loom)` gives `loom`).
    pub check_cfg: Vec<String>,
    /// Package-relative files outside `src/` that a `#[path = "..."]` in `src/` names (`tests/common/scratch.rs`), filled
    /// by [`crate::load_workspace`].
    pub path_modules: Vec<String>,
    /// Labels of the files outside the package that a `#[path = "..."]` names (`//other:src/x.rs`), filled by
    /// [`crate::load_workspace`]; every target of the package lists them in `srcs`.
    pub foreign_srcs: Vec<String>,
    /// Labels of the files outside the package that those files load with `include_str!` or `include_bytes!`, for
    /// every target's `compile_data`.
    pub foreign_data: Vec<String>,
    /// Package-relative files of another directory that the files of `tests/` or `examples/` (the key) name in a
    /// `#[path]` or `include!`: those targets' `srcs` glob only their own directory.
    pub local_srcs: BTreeMap<String, Vec<String>>,
    /// Package-relative files that the package's sources load with a literal `include_str!` or `include_bytes!` from
    /// outside the directory their includer's targets already take as `compile_data`, by that directory (`src`,
    /// `tests`, `examples` or `benches`).
    pub local_data: BTreeMap<String, Vec<String>>,
    /// What the build script loads that way: package-relative files, then labels of other packages' files.
    pub build_data: Vec<String>,
    /// By (`test` or `bin`, target name).
    pub extras: BTreeMap<(String, String), Extras>,
}

impl Package {
    /// The files `compile_data` of the targets built from `dir` (`src`, `tests`, ...) lists besides the directory itself.
    pub fn local_data(&self, dir: &str) -> Vec<String> {
        self.local_data.get(dir).cloned().unwrap_or_default()
    }

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
    /// The workspace's `bazel_gen.toml`, or the defaults.
    pub config: Config,
    /// The configuration's root-relative path when it is a file in the workspace (the stale test reads it).
    pub config_file: Option<String>,
    /// The files other packages name in a `#[path]`: by owning Bazel package directory (`""` is the root), by file
    /// relative to it, the directories of the packages that need it.
    pub exports: BTreeMap<String, BTreeMap<String, BTreeSet<String>>>,
    /// The crate hub's own lock (`[root] crate_lock`), read when `[device]` is set: what crate_universe resolved for each
    /// third-party crate (the features Cargo unified, the crates it links).
    pub locked: Option<Locked>,
}

/// The third-party crates of crate_universe's lock, by crate name (every version and target platform together: more
/// features and edges than one build links, which only tags more).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Locked {
    /// The resolved features of each crate.
    pub features: BTreeMap<String, BTreeSet<String>>,
    /// The crates each crate links (its normal dependencies).
    pub deps: BTreeMap<String, BTreeSet<String>>,
}

impl Locked {
    /// Reads `cargo-bazel-lock.json`'s `crates`.
    pub fn parse(text: &str) -> Result<Locked, String> {
        let lock = crate::json::parse(text)?;
        let crates = lock
            .get("crates")
            .and_then(Json::as_object)
            .ok_or("the crate lock has no `crates` object")?;
        // A list, or `{ common = [...], selects = { triple = [...] } }`.
        fn flat(value: Option<&Json>) -> Vec<&Json> {
            match value {
                Some(Json::Array(items)) => items.iter().collect(),
                Some(object) => {
                    let mut out: Vec<&Json> = object
                        .get("common")
                        .and_then(Json::as_array)
                        .map(|a| a.iter().collect())
                        .unwrap_or_default();
                    if let Some(selects) = object.get("selects").and_then(Json::as_object) {
                        for list in selects.values() {
                            out.extend(list.as_array().into_iter().flatten());
                        }
                    }
                    out
                }
                None => Vec::new(),
            }
        }
        let mut locked = Locked::default();
        for (key, krate) in crates {
            let name = krate
                .get("name")
                .and_then(Json::as_str)
                .unwrap_or_else(|| key.split(' ').next().unwrap_or(key))
                .to_owned();
            let attrs = krate.get("common_attrs");
            let features = locked.features.entry(name.clone()).or_default();
            for feature in flat(attrs.and_then(|a| a.get("crate_features"))) {
                features.extend(feature.as_str().map(str::to_owned));
            }
            let deps = locked.deps.entry(name).or_default();
            for dep in flat(attrs.and_then(|a| a.get("deps"))) {
                if let Some(id) = dep.get("id").and_then(Json::as_str) {
                    deps.insert(id.split(' ').next().unwrap_or(id).to_owned());
                }
            }
        }
        Ok(locked)
    }
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
/// returns for a path relative to the root (`example-core/Cargo.toml`).
pub fn load(
    metadata: &Json,
    manifest: &dyn Fn(&str) -> Result<String, String>,
    config: Config,
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
    // The tags need what crate_universe resolved for the third-party crates: Cargo.toml of a member only says what a
    // member asks for. Without the lock only those requests are seen.
    if config.device.is_some()
        && config.crate_lock.is_none()
        && let Some((package, dep)) = packages.iter().find_map(|p| {
            p.deps
                .iter()
                .find(|d| d.member.is_none() && config.external_key(d).is_none())
                .map(|d| (&p.name, &d.name))
        })
    {
        return Err(format!(
            "[device] is declared and {package} depends on the third-party crate {dep}, but no crate lock is set: without \
             crate_universe's own lock the generator cannot see which third-party crates resolve with a device feature or \
             link one, and would leave card tests untagged. Give the crate hub a lockfile (`lockfile = \
             \"//:cargo-bazel-lock.json\"` in `crate.from_cargo`), write it with `CARGO_BAZEL_REPIN=1 bazel mod deps \
             --lockfile_mode=update`, set `crate_lock = \"cargo-bazel-lock.json\"` under `[root]` in bazel_gen.toml and pass \
             the file as the stale test's `lock`"
        ));
    }
    let locked = match (&config.device, &config.crate_lock) {
        (Some(_), Some(path)) => Some(Locked::parse(&manifest(path)?).map_err(|e| {
            format!("{path}: {e} (the device tags read it: give the stale test the file as `lock`)")
        })?),
        _ => None,
    };
    Ok(Workspace {
        packages,
        config,
        config_file: None,
        exports: BTreeMap::new(),
        locked,
    })
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
            Some("build") => DepKind::Build,
            Some(other) => {
                return Err(format!("{what}: {other}-dependencies are not modelled"));
            }
        };
        let rename = dep.get("rename").and_then(Json::as_str).map(str::to_owned);
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
            rename,
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
    let mut build_script = None;
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
            ["custom-build"] => {
                let src = relative(
                    &format!("{root}/{dir}"),
                    str_field(target, "src_path", &what)?,
                )?;
                if src != "build.rs" {
                    return Err(format!(
                        "{dir}/Cargo.toml: the build script {src} is not `build.rs` (only that file is modelled)"
                    ));
                }
                build_script = Some(BuildScript {
                    src,
                    edition: str_field(target, "edition", &what)?.to_owned(),
                });
                continue;
            }
            ["lib"] => TargetKind::Lib,
            ["bin"] => TargetKind::Bin,
            ["test"] => TargetKind::Test,
            ["example"] => TargetKind::Example,
            other => {
                return Err(format!(
                    "{what} has kind {other:?}: only lib, bin, test and example targets are modelled \
                     (a bench, proc-macro or other crate type is a generator change)"
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
    let (mut lints, mut check_cfg, inherits) =
        lint_tables(&text, "lints").map_err(|e| format!("{dir}/Cargo.toml: {e}"))?;
    if inherits {
        // `[lints] workspace = true`: the root manifest's `[workspace.lints.<tool>]` tables.
        (lints, check_cfg, _) = lint_tables(&manifest("Cargo.toml")?, "workspace.lints")
            .map_err(|e| format!("Cargo.toml: {e}"))?;
    }
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
        build_script,
        links: package
            .get("links")
            .and_then(Json::as_str)
            .map(str::to_owned),
        lints,
        check_cfg,
        path_modules: Vec::new(),
        foreign_srcs: Vec::new(),
        foreign_data: Vec::new(),
        local_srcs: BTreeMap::new(),
        local_data: BTreeMap::new(),
        build_data: Vec::new(),
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

/// A piece of one manifest line: a quoted string (its content), a bare word, or one punctuation character.
#[derive(PartialEq)]
enum Tok {
    Text(String),
    Punct(char),
}

/// Splits one manifest line into tokens. Basic (`"..."`, with `\` escapes) and literal (`'...'`) strings are single
/// tokens, so brackets and `#` inside them mean nothing; a `#` outside a string starts a comment. `multiline` carries
/// the quote of a `"""` or `'''` string that is still open at the end of the line into the next call.
fn tokens(line: &str, multiline: &mut Option<char>) -> Vec<Tok> {
    let mut out = Vec::new();
    let mut chars = line.chars().peekable();
    // Skips to the end of a multi-line string; false when the line ends first.
    fn close(chars: &mut std::iter::Peekable<std::str::Chars<'_>>, q: char) -> bool {
        while let Some(d) = chars.next() {
            if d == '\\' && q == '"' {
                chars.next();
            } else if d == q {
                let mut rest = chars.clone();
                if rest.next() == Some(q) && rest.next() == Some(q) {
                    chars.next();
                    chars.next();
                    return true;
                }
            }
        }
        false
    }
    if let Some(q) = *multiline {
        if !close(&mut chars, q) {
            return out;
        }
        *multiline = None;
        out.push(Tok::Text(String::new()));
    }
    while let Some(c) = chars.next() {
        match c {
            '#' => break,
            '"' | '\'' => {
                let mut rest = chars.clone();
                if rest.next() == Some(c) && rest.next() == Some(c) {
                    chars.next();
                    chars.next();
                    if !close(&mut chars, c) {
                        *multiline = Some(c);
                    }
                    out.push(Tok::Text(String::new()));
                    continue;
                }
                let mut text = String::new();
                while let Some(d) = chars.next() {
                    if d == c {
                        break;
                    }
                    if d == '\\' && c == '"' {
                        text.extend(chars.next());
                    } else {
                        text.push(d);
                    }
                }
                out.push(Tok::Text(text));
            }
            '=' | '[' | ']' | '{' | '}' | ',' => out.push(Tok::Punct(c)),
            c if c.is_whitespace() => {}
            _ => {
                let mut text = c.to_string();
                while let Some(&d) = chars.peek() {
                    if d.is_whitespace() || "=[]{},\"'#".contains(d) {
                        break;
                    }
                    text.push(d);
                    chars.next();
                }
                out.push(Tok::Text(text));
            }
        }
    }
    out
}

/// Whether a target table of the manifest (`[lib]`, `[[bin]]`, `[[test]]`, `[[bench]]`, `[[example]]`) has a `harness`
/// key (bare, `"harness"` or `'harness'`), on its own line or in an inline table of a target array.
fn sets_harness(text: &str) -> bool {
    fn has_harness(toks: &[Tok]) -> bool {
        toks.windows(2)
            .any(|w| matches!((&w[0], &w[1]), (Tok::Text(k), Tok::Punct('=')) if k == "harness"))
    }
    fn is_target(name: &str) -> bool {
        matches!(name, "lib" | "bin" | "test" | "bench" | "example")
    }
    fn depth(toks: &[Tok]) -> i32 {
        let count = |c| toks.iter().filter(|t| **t == Tok::Punct(c)).count() as i32;
        count('[') - count(']')
    }
    let mut in_target = false;
    let mut multiline = None;
    // Open brackets of a target array that continues on the next lines.
    let mut open = 0;
    for line in text.lines() {
        let toks = tokens(line, &mut multiline);
        if toks.is_empty() {
            continue;
        }
        if open > 0 {
            if has_harness(&toks) {
                return true;
            }
            open += depth(&toks);
            continue;
        }
        if toks[0] == Tok::Punct('[') {
            let names: Vec<&Tok> = toks
                .iter()
                .filter(|t| !matches!(t, Tok::Punct('[' | ']')))
                .collect();
            in_target = matches!(names.as_slice(), [Tok::Text(h)]
                if is_target(h));
            continue;
        }
        if let [Tok::Text(key), Tok::Punct('='), value @ ..] = toks.as_slice() {
            // `harness = false` in a target table, or the dotted `lib.harness = false`.
            let dotted = key
                .split_once('.')
                .is_some_and(|(t, k)| is_target(t) && k == "harness");
            if (in_target && key == "harness") || dotted {
                return true;
            }
            // `test = [{ name = "t", harness = false }]` and the like, also over several lines.
            if is_target(key) {
                if has_harness(value) {
                    return true;
                }
                open = depth(value).max(0);
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
    let (lints, check_cfg, inherits) = lint_tables(text, "lints")?;
    if inherits {
        return Err("[lints] workspace = true names the root manifest's lints".to_owned());
    }
    Ok((lints, check_cfg))
}

/// The `[<prefix>.rust]`, `[<prefix>.clippy]` and `[<prefix>.rustdoc]` tables of a manifest (`prefix` is `lints` in a
/// member, `workspace.lints` in the root), and whether a member's `[lints]` says `workspace = true` (which Cargo allows
/// only alone).
fn lint_tables(text: &str, prefix: &str) -> Result<(Vec<Lint>, Vec<String>, bool), String> {
    let mut lints = Vec::new();
    let mut check_cfg = Vec::new();
    let mut inherits = false;
    let mut in_lints = false;
    let mut tool: Option<String> = None;
    for line in text.lines() {
        let line = line.split(" #").next().unwrap_or("").trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            let header = line.trim_matches(|c| c == '[' || c == ']').trim();
            tool = None;
            in_lints = false;
            if header == prefix {
                if prefix != "lints" {
                    return Err(format!(
                        "[{header}] is not modelled; use [{header}.<tool>] tables"
                    ));
                }
                in_lints = true;
            } else if let Some(name) = header
                .strip_prefix(prefix)
                .and_then(|h| h.strip_prefix('.'))
            {
                match name {
                    "rust" | "clippy" | "rustdoc" => tool = Some(name.to_owned()),
                    other => return Err(format!("[{prefix}.{other}] is not modelled")),
                }
            }
            continue;
        }
        if in_lints {
            let setting: Vec<&str> = line.splitn(2, '=').map(str::trim).collect();
            if setting != ["workspace", "true"] {
                return Err(format!(
                    "[lints]: `{line}` is not modelled (only `workspace = true`)"
                ));
            }
            inherits = true;
            continue;
        }
        let Some(tool) = &tool else { continue };
        let (name, value) = line
            .split_once('=')
            .ok_or_else(|| format!("[{prefix}.{tool}]: cannot read `{line}`"))?;
        let name = name.trim().trim_matches('"').to_owned();
        if name.is_empty()
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            // A dotted key (`x.level = "warn"`) is a table, not a lint.
            return Err(format!("[{prefix}.{tool}]: `{name}` is not a lint name"));
        }
        let value = value.trim();
        let (level, priority) =
            if let Some(table) = value.strip_prefix('{').and_then(|v| v.strip_suffix('}')) {
                let mut level = None;
                let mut priority = 0;
                // check-cfg holds commas inside its array: take it out first.
                let (plain, cfgs) = match table.split_once("check-cfg") {
                    Some((before, after)) => {
                        let array = after
                            .split_once('[')
                            .and_then(|(_, a)| a.split_once(']'))
                            .ok_or_else(|| {
                                format!("[{prefix}.{tool}] {name}: cannot read check-cfg")
                            })?;
                        (format!("{before}{}", array.1), Some(array.0.to_owned()))
                    }
                    None => (table.to_owned(), None),
                };
                if let Some(cfgs) = cfgs {
                    if name != "unexpected_cfgs" {
                        return Err(format!(
                            "[{prefix}.{tool}] {name}: check-cfg outside unexpected_cfgs"
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
                        .ok_or_else(|| format!("[{prefix}.{tool}] {name}: cannot read `{part}`"))?;
                    match key.trim() {
                        "level" => level = Some(value.trim().trim_matches('"').to_owned()),
                        "priority" => {
                            priority = value
                                .trim()
                                .parse()
                                .map_err(|_| format!("[{prefix}.{tool}] {name}: bad priority"))?;
                        }
                        other => {
                            return Err(format!("[{prefix}.{tool}] {name}: unknown key `{other}`"));
                        }
                    }
                }
                (
                    level.ok_or_else(|| format!("[{prefix}.{tool}] {name}: no level"))?,
                    priority,
                )
            } else {
                (value.trim_matches('"').to_owned(), 0)
            };
        if !matches!(
            level.as_str(),
            "allow" | "warn" | "deny" | "forbid" | "force-warn"
        ) {
            return Err(format!("[{prefix}.{tool}] {name}: unknown level `{level}`"));
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
    if inherits && !lints.is_empty() {
        return Err(
            "[lints] workspace = true with [lints.<tool>] tables: Cargo allows one or the other"
                .to_owned(),
        );
    }
    Ok((lints, check_cfg, inherits))
}

/// The module files a source file's `#[path = "..."]` attributes name, relative to the workspace root: `file` is the
/// root-relative path of the source. Each path resolves the way rustc does: against the file's directory, plus, inside
/// an inline `mod name { ... }`, the module's name (and, in a file that is not a crate root or `mod.rs`, the file's
/// stem first). A path that leaves the workspace is an error naming the file.
pub fn path_attributes(file: &str, text: &str) -> Result<Vec<String>, String> {
    let mut dir: Vec<&str> = file.split('/').collect();
    let name = dir.pop().unwrap_or("");
    let mod_rs = matches!(name, "mod.rs" | "main.rs" | "lib.rs")
        || matches!(
            dir.last().copied(),
            Some("tests" | "examples" | "benches" | "bin")
        );
    let stem = name.strip_suffix(".rs").unwrap_or(name);
    let mut out = Vec::new();
    // The inline modules the scan is inside: (name, brace depth outside it).
    let mut inline: Vec<(String, usize)> = Vec::new();
    let mut depth = 0usize;
    for line in text.lines() {
        let code = line.split("//").next().unwrap_or("");
        if let Some(rest) = code.trim().strip_prefix("#[path") {
            let target = rest
                .trim_start()
                .strip_prefix('=')
                .and_then(|r| r.split_once(']').map(|(inside, _)| inside))
                .and_then(|r| r.trim().strip_prefix('"'))
                .and_then(|r| r.strip_suffix('"'))
                .ok_or_else(|| format!("{file}: unreadable path attribute `{}`", code.trim()))?;
            let target = target.replace('\\', "/");
            if target.starts_with('/') {
                return Err(format!("{file}: path `{target}` is absolute"));
            }
            let mut parts = dir.clone();
            if !mod_rs && !inline.is_empty() {
                parts.push(stem);
            }
            parts.extend(inline.iter().map(|(n, _)| n.as_str()));
            for part in target.split('/') {
                match part {
                    "" | "." => {}
                    ".." => {
                        parts.pop().ok_or_else(|| {
                            format!("{file}: path `{target}` leaves the workspace")
                        })?;
                    }
                    other => parts.push(other),
                }
            }
            out.push(parts.join("/"));
        }
        // Track the inline modules: `mod name {` opens one, its closing brace ends it.
        let mut words = code
            .split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .filter(|w| !w.is_empty());
        let mut opening = None;
        while let Some(word) = words.next() {
            if word == "mod"
                && let Some(name) = words.next()
            {
                let after = code
                    .split_once(&format!("mod {name}"))
                    .map_or("", |(_, r)| r);
                if after.trim_start().starts_with('{') {
                    opening = Some(name.to_owned());
                }
            }
        }
        let mut in_string = false;
        let mut prev = ' ';
        for c in code.chars() {
            if c == '"' && prev != '\\' {
                in_string = !in_string;
            } else if !in_string && c == '{' {
                if let Some(name) = opening.take() {
                    inline.push((name, depth));
                }
                depth += 1;
            } else if !in_string && c == '}' {
                depth = depth.saturating_sub(1);
                if inline.last().is_some_and(|(_, d)| *d == depth) {
                    inline.pop();
                }
            }
            prev = c;
        }
    }
    Ok(out)
}

/// The files a source file loads with `include_str!`, `include_bytes!` or `include!` whose argument is a string literal,
/// or `concat!(env!("CARGO_MANIFEST_DIR"), "<literal>")`, relative to the workspace root, each with whether it is
/// source (`include!`). `file` is the root-relative path of the source and `manifest_dir` that of the crate being
/// built; a literal is relative to the file's directory, a `CARGO_MANIFEST_DIR` one to the crate's. Any other argument
/// is not read.
pub fn include_files(
    file: &str,
    manifest_dir: &str,
    text: &str,
) -> Result<Vec<(String, bool)>, String> {
    const MANIFEST: &str = "concat!(env!(\"CARGO_MANIFEST_DIR\"),";
    let mut out = Vec::new();
    for line in text.lines() {
        let code = line.split("//").next().unwrap_or("");
        for (macro_name, source) in [
            ("include_str!(", false),
            ("include_bytes!(", false),
            ("include!(", true),
        ] {
            let mut rest = code;
            while let Some((before, after)) = rest.split_once(macro_name) {
                rest = after;
                // `my_include!(` is another macro.
                if before.ends_with(|c: char| c.is_alphanumeric() || c == '_') {
                    continue;
                }
                let arg = after.trim_start();
                let (base, arg) = match arg.strip_prefix(MANIFEST) {
                    Some(r) => (manifest_dir, r.trim_start()),
                    None => (file.rsplit_once('/').map_or("", |(d, _)| d), arg),
                };
                let Some(literal) = arg.strip_prefix('"') else {
                    continue;
                };
                let Some((target, _)) = literal.split_once('"') else {
                    continue;
                };
                let mut parts: Vec<&str> = base.split('/').filter(|p| !p.is_empty()).collect();
                for part in target.split('/') {
                    match part {
                        "" | "." => {}
                        ".." => {
                            parts.pop().ok_or_else(|| {
                                format!("{file}: path `{target}` leaves the workspace")
                            })?;
                        }
                        other => parts.push(other),
                    }
                }
                out.push((parts.join("/"), source));
            }
        }
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
            "test = [\n  { name = \"a\" },\n  {\n    name = \"t\",\n    harness = false,\n  },\n]\n",
            "test = [\n  { name = \"a]b\" },\n  { name = \"t\", harness = false },\n]\n",
            "[[test]]\nname = \"t\"\n'harness' = false\n",
            "[[test]]\nname = \"t\"\n\"harness\" = false\n",
            "lib = { harness = false }\n",
            "lib.harness = false\n",
            "test.harness = false\n",
            "bench.harness = false\n",
        ] {
            assert!(sets_harness(text), "{text} was accepted");
        }
        for text in [
            "[features]\nharness = []\n",
            "[package]\n# harness = false is not used here\nname = \"x\"\n",
            "[dependencies]\nharness = \"1\"\n",
            "test = [\n  { name = \"a#b\" },\n]\n[features]\nharness = []\n",
            "test = [\"a#]b\"]\n[dependencies]\nharness = \"1\"\n",
            "[package]\ndescription = \"\"\"\n[lib]\nharness = false\n\"\"\"\n",
            "[package]\ndescription = \'\'\'\n[lib]\nharness = false\n\'\'\'\n",
        ] {
            assert!(!sets_harness(text), "{text} was refused");
        }
    }

    #[test]
    fn path_attributes_resolve_from_the_file() {
        let text = "#[cfg(test)]\n#[path = \"../tests/common/scratch.rs\"]\nmod scratch;\n  #[path=\"x/y.rs\"] mod y;\n";
        assert_eq!(
            path_attributes("p/src/lib.rs", text),
            Ok(vec![
                "p/tests/common/scratch.rs".to_owned(),
                "p/src/x/y.rs".to_owned()
            ])
        );
        assert_eq!(
            path_attributes("p/src/lib.rs", "#[path = \"../../other/x.rs\"]\n"),
            Ok(vec!["other/x.rs".to_owned()])
        );
        let err = path_attributes("p/src/lib.rs", "#[path = \"../../../x.rs\"]\n").unwrap_err();
        assert!(err.contains("p/src/lib.rs") && err.contains("leaves the workspace"));
    }

    #[test]
    fn path_attributes_follow_inline_modules() {
        let text = "mod a {\n    mod b {\n        #[path = \"x.rs\"]\n        mod x;\n    }\n    #[path = \"y.rs\"]\n    mod y;\n}\n#[path = \"z.rs\"]\nmod z;\n";
        // A crate root or mod.rs resolves against its directory plus the module names.
        assert_eq!(
            path_attributes("p/src/lib.rs", text),
            Ok(vec![
                "p/src/a/b/x.rs".to_owned(),
                "p/src/a/y.rs".to_owned(),
                "p/src/z.rs".to_owned()
            ])
        );
        // Any other file adds its stem.
        assert_eq!(
            path_attributes("p/src/net.rs", text),
            Ok(vec![
                "p/src/net/a/b/x.rs".to_owned(),
                "p/src/net/a/y.rs".to_owned(),
                "p/src/z.rs".to_owned()
            ])
        );
    }

    #[test]
    fn included_files_resolve_from_the_file() {
        let text = "static A: &[u8] = include_bytes!(\"../data/a.bin\");\nconst B: &str = include_str!( \"b.txt\" ); // include_str!(\"c\")\nconst C: &str = include_str!(concat!(env!(\"OUT_DIR\"), \"/d\"));\ninclude!(\"gen.rs\");\nconst E: &str = include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/data/e.txt\"));\nmy_include!(\"f\");\n";
        let mut got = include_files("p/src/x.rs", "p", text).unwrap();
        got.sort();
        assert_eq!(
            got,
            vec![
                ("p/data/a.bin".to_owned(), false),
                ("p/data/e.txt".to_owned(), false),
                ("p/src/b.txt".to_owned(), false),
                ("p/src/gen.rs".to_owned(), true),
            ]
        );
    }

    #[test]
    fn paths_are_relative_with_forward_slashes() {
        assert_eq!(
            relative("/w/r", "/w/r/example-core/src/lib.rs"),
            Ok("example-core/src/lib.rs".into())
        );
        assert_eq!(
            relative(r"C:\w\r", r"C:\w\r\tools\bazel-gen\src\lib.rs"),
            Ok("tools/bazel-gen/src/lib.rs".into())
        );
        assert!(relative("/w/r", "/w/rx/src/lib.rs").is_err());
    }
}
