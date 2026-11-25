//! The per-repo configuration, `bazel_gen.toml` at the workspace root (README, "Configuration"): what differs between
//! workspaces (the header, the root package, the device world, dependencies taken from other Bazel modules, per-package
//! extras). A workspace without the file gets [`Config::default`].

use std::collections::BTreeMap;

use crate::json::Json;
use crate::model::Dep;

/// The device world: the feature that loads the device's driver, and where it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Device {
    /// The member whose `feature` opens the card: a test whose graph links it with the feature is tagged `gpu` and
    /// `external`, and the testing world may not turn it on. `None` when only crates of other modules open it.
    pub member: Option<String>,
    /// The feature the device world adds to every member that declares it (`cuda`); it also names the world's targets
    /// (`<crate>_cuda_testing`, `unit_cuda_test`).
    pub feature: String,
    /// Crates from other modules ([`Config::external`]) whose `feature` opens the card: the testing world may not turn it
    /// on for them. (A device-world test that links any crate of [`Config::external`] is tagged like one that links
    /// `member`.)
    pub external: Vec<String>,
    /// More device features (`vulkan`): a crate resolved with any of them, like one with `feature`, opens the card.
    pub also: Vec<String>,
}

impl Device {
    /// `feature` and `also`: every feature that opens the card.
    pub fn all_features(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.feature.as_str()).chain(self.also.iter().map(String::as_str))
    }
}

/// A dependency that is a library of another Bazel module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct External {
    /// The label of its package (`@euhedral_example//example-gpu`); the target is the crate name with the world's suffix.
    pub package: String,
    /// This crate's own suffix for a world (production, testing, device), over [`Config::external_worlds`].
    pub worlds: [Option<String>; 3],
}

/// Bazel-only settings of one member, beside its own `[package.metadata.bazel]`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PackageConfig {
    /// More package-relative globs in the `compile_data` of every target of the member (`kernels/**`).
    pub compile_data: Vec<String>,
    /// Labels in the `compile_data` and `data` of the member's tests (`//:testdata`): files they include or read.
    pub test_data: Vec<String>,
    /// Globs of package files the build script reads (`kernels/**`): its `data`.
    pub build_script_data: Vec<String>,
    /// Environment variables of the build script run (`build_script_env`): what the script reads from the machine.
    pub build_script_env: Vec<(String, String)>,
    /// Tags of the build script target only (`gpu`; `external` follows from it): its library and tests do not inherit them.
    pub build_script_tags: Vec<String>,
    /// `Some(true)`: the package's tests (and those of every package that links it) open the card, whatever its features;
    /// `Some(false)`: its own tests never do, though they link a crate with a device feature on. `None`: derived.
    pub card_tests: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// The first line of every generated file (with its `#`).
    pub header: String,
    /// The root files exported for other packages (crate_universe's inputs, a test's `data`).
    pub exports: Vec<String>,
    /// Filegroups of the root package, by name: globs of root files that a member's tests name (`testdata`).
    pub filegroups: BTreeMap<String, Vec<String>>,
    /// Whether the root package gets `//:gazelle` (the shared generator, `bazel_gen`) and the stale test
    /// (`bazel_gen_stale_test`) from `@euhedral_build//bazel-gen:defs.bzl`.
    pub gazelle: bool,
    /// Whether the stale test also checks the `MODULE.bazel.lock` records of the manifests (crate_universe reads them).
    pub lock_check: bool,
    /// crate_universe's own lock (`lockfile =` of the hub's `crate.from_cargo`), root-relative. With one the lock check
    /// compares it with `Cargo.lock` instead of reading `MODULE.bazel.lock`, which then records no manifest.
    pub crate_lock: Option<String>,
    /// The crate_universe hub repository of the third-party crates (`crates` gives `@crates//:serde`).
    pub crates: String,
    /// The device world; `None` builds only the production and testing worlds.
    pub device: Option<Device>,
    /// Dependencies that are libraries of another Bazel module, by crate name.
    pub external: BTreeMap<String, External>,
    /// The suffix of an external library's target in each of this workspace's worlds (production, testing, device): the
    /// other module's world that this world links. A crate taken with `default-features = false` and features chosen by
    /// this workspace's own features gets them from the world named here.
    pub external_worlds: [String; 3],
    /// By member name.
    pub packages: BTreeMap<String, PackageConfig>,
}

/// The configuration's file name, at the workspace root. Not `bazel-…`: Bazel leaves the root's `bazel-*` entries out of the
/// execroot (they are its output symlinks), so a test could not read such a file from its runfiles.
pub const FILE: &str = "bazel_gen.toml";

impl Default for Config {
    fn default() -> Config {
        Config {
            header: "# @generated by @euhedral_build//bazel-gen from the Cargo manifests; do not edit. Regenerate with \
                     `bazel run //:gazelle`."
                .to_owned(),
            exports: ["Cargo.lock", "Cargo.toml", "MODULE.bazel", "MODULE.bazel.lock"]
                .map(str::to_owned)
                .to_vec(),
            filegroups: BTreeMap::new(),
            gazelle: true,
            lock_check: true,
            crate_lock: None,
            crates: "crates".to_owned(),
            device: None,
            external: BTreeMap::new(),
            external_worlds: ["", "_testing", "_cuda_testing"].map(str::to_owned),
            packages: BTreeMap::new(),
        }
    }
}

impl Config {
    pub fn package(&self, name: &str) -> PackageConfig {
        self.packages.get(name).cloned().unwrap_or_default()
    }

    /// The label of the external crate `name`'s library in world `slot` (0 production, 1 testing, 2 device), or `None`
    /// when `name` is not an external crate.
    pub fn external_label(&self, name: &str, slot: usize) -> Option<String> {
        self.label_of(name, name, slot)
    }

    /// The `[external.crates]` entry that maps `dep`: the one named by its key, else the one named by its package. A
    /// renamed dependency (`k = { package = "p" }`) so picks its own world through an entry `k` (the same package under a
    /// second feature set), and shares the package's entry otherwise.
    pub fn external_key(&self, dep: &Dep) -> Option<&str> {
        [dep.key(), dep.name.as_str()]
            .into_iter()
            .find_map(|key| self.external.get_key_value(key).map(|(k, _)| k.as_str()))
    }

    /// The label of `dep`'s library in world `slot`, when another module builds it.
    pub fn external_dep_label(&self, dep: &Dep, slot: usize) -> Option<String> {
        self.label_of(self.external_key(dep)?, &dep.name, slot)
    }

    /// The label of the entry `key`'s package's library `crate_name`, in world `slot`.
    fn label_of(&self, key: &str, crate_name: &str, slot: usize) -> Option<String> {
        let external = self.external.get(key)?;
        let suffix = external.worlds[slot]
            .as_deref()
            .unwrap_or(&self.external_worlds[slot]);
        Some(format!(
            "{}:{}{suffix}",
            external.package,
            crate_name.replace('-', "_")
        ))
    }

    /// Reads `bazel_gen.toml`'s text. An unknown key or a value of the wrong type is an error: a misspelt setting must not
    /// fall back to the default silently.
    pub fn parse(text: &str) -> Result<Config, String> {
        let doc = toml(text)?;
        let mut config = Config::default();
        let top = doc.as_object().expect("a document is a table");
        for (key, value) in top {
            match key.as_str() {
                "header" => {
                    let header = string(value, "header")?;
                    if !header.starts_with('#') || header.contains('\n') {
                        return Err("header: one line that starts with `#`".to_owned());
                    }
                    config.header = header;
                }
                "root" => {
                    for (key, value) in table(value, "root")? {
                        let here = format!("root.{key}");
                        match key.as_str() {
                            "exports" => config.exports = strings(value, &here)?,
                            "gazelle" => config.gazelle = boolean(value, &here)?,
                            "lock_check" => config.lock_check = boolean(value, &here)?,
                            "crate_lock" => config.crate_lock = Some(string(value, &here)?),
                            "filegroups" => {
                                for (name, globs) in table(value, &here)? {
                                    config.filegroups.insert(
                                        name.clone(),
                                        strings(globs, &format!("{here}.{name}"))?,
                                    );
                                }
                            }
                            other => return Err(format!("root: unknown key `{other}`")),
                        }
                    }
                }
                "crates" => config.crates = string(value, "crates")?,
                "device" => {
                    let (mut member, mut feature, mut external) = (None, None, Vec::new());
                    let mut also = Vec::new();
                    for (key, value) in table(value, "device")? {
                        let here = format!("device.{key}");
                        match key.as_str() {
                            "member" => member = Some(string(value, &here)?),
                            "feature" => feature = Some(string(value, &here)?),
                            "external" => external = strings(value, &here)?,
                            "also" => also = strings(value, &here)?,
                            other => return Err(format!("device: unknown key `{other}`")),
                        }
                    }
                    config.device = Some(Device {
                        member,
                        feature: feature.ok_or("device: no `feature`")?,
                        external,
                        also,
                    });
                }
                "external" => {
                    for (key, value) in table(value, "external")? {
                        match key.as_str() {
                            "crates" => {
                                for (name, entry) in table(value, "external.crates")? {
                                    let here = format!("external.crates.{name}");
                                    let external = match entry {
                                        Json::Object(fields) => {
                                            let mut external = External {
                                                package: String::new(),
                                                worlds: [None, None, None],
                                            };
                                            for (key, value) in fields {
                                                match key.as_str() {
                                                    "package" => {
                                                        external.package = string(value, &here)?
                                                    }
                                                    "worlds" => {
                                                        for (world, suffix) in table(value, &here)?
                                                        {
                                                            external.worlds
                                                                [world_slot(world, &here)?] =
                                                                Some(string(suffix, &here)?);
                                                        }
                                                    }
                                                    other => {
                                                        return Err(format!(
                                                            "{here}: unknown key `{other}`"
                                                        ));
                                                    }
                                                }
                                            }
                                            external
                                        }
                                        _ => External {
                                            package: string(entry, &here)?,
                                            worlds: [None, None, None],
                                        },
                                    };
                                    let label = &external.package;
                                    if !label.starts_with('@') || label.contains(':') {
                                        return Err(format!(
                                            "{here}: `{label}` is not a package label of another module \
                                             (`@module//dir`)"
                                        ));
                                    }
                                    config.external.insert(name.clone(), external);
                                }
                            }
                            "worlds" => {
                                for (world, suffix) in table(value, "external.worlds")? {
                                    config.external_worlds[world_slot(world, "external.worlds")?] =
                                        string(suffix, &format!("external.worlds.{world}"))?;
                                }
                            }
                            other => return Err(format!("external: unknown key `{other}`")),
                        }
                    }
                }
                "packages" => {
                    for (name, settings) in table(value, "packages")? {
                        let mut package = PackageConfig::default();
                        for (key, value) in table(settings, &format!("packages.{name}"))? {
                            let here = format!("packages.{name}.{key}");
                            match key.as_str() {
                                "compile_data" => package.compile_data = strings(value, &here)?,
                                "test_data" => package.test_data = strings(value, &here)?,
                                "build_script_data" => {
                                    package.build_script_data = strings(value, &here)?
                                }
                                "card_tests" => package.card_tests = Some(boolean(value, &here)?),
                                "build_script_tags" => {
                                    package.build_script_tags = strings(value, &here)?
                                }
                                "build_script_env" => {
                                    for (var, text) in table(value, &here)? {
                                        package.build_script_env.push((
                                            var.clone(),
                                            string(text, &format!("{here}.{var}"))?,
                                        ));
                                    }
                                }
                                other => {
                                    return Err(format!("packages.{name}: unknown key `{other}`"));
                                }
                            }
                        }
                        config.packages.insert(name.clone(), package);
                    }
                }
                other => return Err(format!("unknown key `{other}`")),
            }
        }
        if let Some(device) = &config.device
            && let Some(name) = device
                .external
                .iter()
                .find(|n| !config.external.contains_key(*n))
        {
            return Err(format!(
                "device.external: `{name}` is not in [external.crates]"
            ));
        }
        Ok(config)
    }
}

fn world_slot(world: &str, what: &str) -> Result<usize, String> {
    match world {
        "production" => Ok(0),
        "testing" => Ok(1),
        "device" => Ok(2),
        other => Err(format!(
            "{what}: unknown world `{other}` (production, testing, device)"
        )),
    }
}

fn table<'a>(value: &'a Json, what: &str) -> Result<&'a BTreeMap<String, Json>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{what} is not a table"))
}

fn string(value: &Json, what: &str) -> Result<String, String> {
    value
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| format!("{what} is not a string"))
}

fn boolean(value: &Json, what: &str) -> Result<bool, String> {
    value
        .as_bool()
        .ok_or_else(|| format!("{what} is not true or false"))
}

fn strings(value: &Json, what: &str) -> Result<Vec<String>, String> {
    value
        .as_array()
        .ok_or_else(|| format!("{what} is not an array"))?
        .iter()
        .map(|item| string(item, &format!("an entry of {what}")))
        .collect()
}

/// The subset of TOML a configuration needs: `[a.b]` headers, `key = value` with bare, quoted or dotted keys, and values that
/// are basic or literal strings, booleans, integers, arrays (over several lines too) and inline tables. Anything else (array
/// tables, multi-line strings, dates, floats) is an error. The result is a table of [`Json`] values.
pub fn toml(text: &str) -> Result<Json, String> {
    let mut root = BTreeMap::new();
    let mut current: Vec<String> = Vec::new();
    let mut headers = std::collections::BTreeSet::new();
    let mut chars = Chars {
        text: text.chars().collect(),
        at: 0,
        line: 1,
    };
    loop {
        chars.skip_blank_lines();
        let Some(c) = chars.peek() else { break };
        let line = chars.line;
        let fail = |e: String| format!("{}:{line}: {e}", FILE);
        if c == '[' {
            chars.next();
            if chars.peek() == Some('[') {
                return Err(fail("arrays of tables are not read".to_owned()));
            }
            let path = chars.key_path().map_err(fail)?;
            chars.skip_spaces();
            if chars.next() != Some(']') {
                return Err(fail("a table header ends with `]`".to_owned()));
            }
            chars.end_of_line().map_err(fail)?;
            let table = descend(&mut root, &path).map_err(fail)?;
            if !table.is_empty() || !headers.insert(path.clone()) {
                return Err(fail(format!("table [{}] is defined twice", path.join("."))));
            }
            current = path;
            continue;
        }
        let path = chars.key_path().map_err(fail)?;
        chars.skip_spaces();
        if chars.next() != Some('=') {
            return Err(fail("expected `=` after the key".to_owned()));
        }
        let value = chars.value().map_err(fail)?;
        chars.end_of_line().map_err(fail)?;
        let mut full = current.clone();
        full.extend(path);
        insert(&mut root, &full, value).map_err(fail)?;
    }
    Ok(Json::Object(root))
}

/// The table at `path`, made where missing.
fn descend<'a>(
    mut table: &'a mut BTreeMap<String, Json>,
    path: &[String],
) -> Result<&'a mut BTreeMap<String, Json>, String> {
    for key in path {
        let entry = table
            .entry(key.clone())
            .or_insert_with(|| Json::Object(BTreeMap::new()));
        table = match entry {
            Json::Object(inner) => inner,
            _ => return Err(format!("`{key}` is a value, not a table")),
        };
    }
    Ok(table)
}

fn insert(root: &mut BTreeMap<String, Json>, path: &[String], value: Json) -> Result<(), String> {
    let (last, parents) = path.split_last().expect("a key has a name");
    let table = descend(root, parents)?;
    if table.insert(last.clone(), value).is_some() {
        return Err(format!("`{}` is set twice", path.join(".")));
    }
    Ok(())
}

struct Chars {
    text: Vec<char>,
    at: usize,
    line: usize,
}

impl Chars {
    fn peek(&self) -> Option<char> {
        self.text.get(self.at).copied()
    }

    fn next(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.at += 1;
        if c == '\n' {
            self.line += 1;
        }
        Some(c)
    }

    fn skip_spaces(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t')) {
            self.next();
        }
    }

    fn skip_comment(&mut self) {
        if self.peek() == Some('#') {
            while !matches!(self.peek(), None | Some('\n')) {
                self.next();
            }
        }
    }

    /// Spaces, comments and line breaks, as inside an array.
    fn skip_blank_lines(&mut self) {
        loop {
            self.skip_spaces();
            self.skip_comment();
            match self.peek() {
                Some('\n' | '\r') => {
                    self.next();
                }
                _ => return,
            }
        }
    }

    fn end_of_line(&mut self) -> Result<(), String> {
        self.skip_spaces();
        self.skip_comment();
        match self.peek() {
            None | Some('\n' | '\r') => Ok(()),
            Some(c) => Err(format!("unexpected `{c}` after the value")),
        }
    }

    /// `a`, `"a b"`, `a.b."c"`.
    fn key_path(&mut self) -> Result<Vec<String>, String> {
        let mut path = Vec::new();
        loop {
            self.skip_spaces();
            let key = match self.peek() {
                Some(q @ ('"' | '\'')) => {
                    self.next();
                    self.string(q)?
                }
                _ => {
                    let mut key = String::new();
                    while let Some(c) = self
                        .peek()
                        .filter(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
                    {
                        key.push(c);
                        self.next();
                    }
                    if key.is_empty() {
                        return Err("expected a key".to_owned());
                    }
                    key
                }
            };
            path.push(key);
            self.skip_spaces();
            if self.peek() == Some('.') {
                self.next();
            } else {
                return Ok(path);
            }
        }
    }

    /// The rest of a string whose opening quote `q` was read.
    fn string(&mut self, q: char) -> Result<String, String> {
        if self.peek() == Some(q) && self.text.get(self.at + 1) == Some(&q) {
            return Err("multi-line strings are not read".to_owned());
        }
        let mut out = String::new();
        loop {
            match self.next() {
                None | Some('\n') => return Err("a string is not closed on its line".to_owned()),
                Some(c) if c == q => return Ok(out),
                Some('\\') if q == '"' => match self.next() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('n') => out.push('\n'),
                    Some('t') => out.push('\t'),
                    other => {
                        return Err(format!(
                            "the escape `\\{}` is not read",
                            other.unwrap_or(' ')
                        ));
                    }
                },
                Some(c) => out.push(c),
            }
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.skip_spaces();
        match self.peek() {
            Some(q @ ('"' | '\'')) => {
                self.next();
                Ok(Json::String(self.string(q)?))
            }
            Some('[') => {
                self.next();
                let mut items = Vec::new();
                loop {
                    self.skip_blank_lines();
                    if self.peek() == Some(']') {
                        self.next();
                        return Ok(Json::Array(items));
                    }
                    items.push(self.value()?);
                    self.skip_blank_lines();
                    match self.next() {
                        Some(',') => {}
                        Some(']') => return Ok(Json::Array(items)),
                        _ => return Err("expected `,` or `]` in an array".to_owned()),
                    }
                }
            }
            Some('{') => {
                self.next();
                let mut table = BTreeMap::new();
                self.skip_spaces();
                if self.peek() == Some('}') {
                    self.next();
                    return Ok(Json::Object(table));
                }
                loop {
                    let path = self.key_path()?;
                    self.skip_spaces();
                    if self.next() != Some('=') {
                        return Err("expected `=` in an inline table".to_owned());
                    }
                    let value = self.value()?;
                    insert(&mut table, &path, value)?;
                    self.skip_spaces();
                    match self.next() {
                        Some(',') => {}
                        Some('}') => return Ok(Json::Object(table)),
                        _ => return Err("expected `,` or `}` in an inline table".to_owned()),
                    }
                }
            }
            _ => {
                let mut word = String::new();
                while let Some(c) = self
                    .peek()
                    .filter(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_' || *c == '+')
                {
                    word.push(c);
                    self.next();
                }
                match word.as_str() {
                    "true" => Ok(Json::Bool(true)),
                    "false" => Ok(Json::Bool(false)),
                    w if !w.is_empty()
                        && w.trim_start_matches(['-', '+'])
                            .chars()
                            .all(|c| c.is_ascii_digit()) =>
                    {
                        Ok(Json::Number(w.to_owned()))
                    }
                    _ => Err(format!("cannot read the value `{word}`")),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_toml_subset_reads_tables_arrays_and_inline_tables() {
        let doc = toml(
            "# c\nheader = \"# x\" # trailing\n[root]\nexports = [\n  \"a\", # one\n  'b',\n]\n\
             gazelle = false\n[external.crates]\n\"example-gpu\" = '@m//example-gpu'\n\
             [packages.p]\ncompile_data = []\nx = { a = 1, b.c = true }\n",
        )
        .unwrap();
        assert_eq!(doc.get("header"), Some(&Json::String("# x".into())));
        let root = doc.get("root").unwrap();
        assert_eq!(
            root.get("exports"),
            Some(&Json::Array(vec![
                Json::String("a".into()),
                Json::String("b".into())
            ]))
        );
        assert_eq!(root.get("gazelle"), Some(&Json::Bool(false)));
        assert_eq!(
            doc.get("external")
                .unwrap()
                .get("crates")
                .unwrap()
                .get("example-gpu"),
            Some(&Json::String("@m//example-gpu".into()))
        );
        let x = doc
            .get("packages")
            .unwrap()
            .get("p")
            .unwrap()
            .get("x")
            .unwrap();
        assert_eq!(x.get("a"), Some(&Json::Number("1".into())));
        assert_eq!(x.get("b").unwrap().get("c"), Some(&Json::Bool(true)));
    }

    #[test]
    fn unread_toml_forms_are_refused() {
        for text in [
            "[[a]]\n",
            "a = \"\"\"x\"\"\"\n",
            "a = 1.5\n",
            "a = 1\na = 2\n",
            "[a]\n[a]\n",
            "a = \"x\" b\n",
            "a = \"open\n",
            "a = [1 2]\n",
        ] {
            assert!(toml(text).is_err(), "{text:?} was read");
        }
    }

    #[test]
    fn the_configuration_is_read_and_unknown_keys_are_refused() {
        let config = Config::parse(
            "header = \"# gen\"\n[root]\nexports = [\"Cargo.toml\"]\ncrate_lock = \"l.json\"\nfilegroups = { testdata = [\"testdata/**\"] }\n\
             [device]\nmember = \"engine\"\nfeature = \"cuda\"\nexternal = [\"example-memory\"]\n\
             [external.crates]\nexample-gpu = \"@euhedral_example//example-gpu\"\n\
             example-memory = { package = \"@euhedral_example//example-memory\", worlds = { testing = \"\" } }\n\
             [external.worlds]\ndevice = \"_cuda_testing\"\n\
             [packages.engine]\ncompile_data = [\"kernels/**\"]\ntest_data = [\"//:testdata\"]\n",
        )
        .unwrap();
        assert_eq!(config.header, "# gen");
        assert_eq!(config.exports, ["Cargo.toml"]);
        assert_eq!(config.filegroups["testdata"], ["testdata/**"]);
        assert_eq!(config.crate_lock.as_deref(), Some("l.json"));
        assert_eq!(
            config.device,
            Some(Device {
                member: Some("engine".into()),
                feature: "cuda".into(),
                external: vec!["example-memory".into()],
                also: Vec::new(),
            })
        );
        assert_eq!(
            config.external_label("example-gpu", 1).as_deref(),
            Some("@euhedral_example//example-gpu:example_gpu_testing")
        );
        assert_eq!(
            config.external_label("example-memory", 1).as_deref(),
            Some("@euhedral_example//example-memory:example_memory")
        );
        assert_eq!(config.external_label("serde", 1), None);
        assert_eq!(config.external_worlds, ["", "_testing", "_cuda_testing"]);
        assert_eq!(config.package("engine").compile_data, ["kernels/**"]);
        assert_eq!(config.package("engine").test_data, ["//:testdata"]);
        for text in [
            "headr = \"# x\"\n",
            "header = \"x\"\n",
            "[root]\nexport = []\n",
            "[device]\nmember = \"m\"\n",
            "[packages.p]\ncard_tests = \"yes\"\n",
            "[device]\nfeature = \"cuda\"\nexternal = [\"x\"]\n",
            "[external.crates]\nx = \"@m//x:x\"\n",
            "[external.crates]\nx = \"//x\"\n",
            "[external.worlds]\ncuda = \"\"\n",
            "[packages.p]\ndata = []\n",
            "[root]\ngazelle = \"yes\"\n",
        ] {
            assert!(Config::parse(text).is_err(), "{text:?} was accepted");
        }
    }
}
