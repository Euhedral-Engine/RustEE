//! Turns the workspace and its three worlds into BUILD files (docs/design/bazel.md, "First-party BUILD files").

use std::collections::{BTreeMap, BTreeSet};

use crate::cfg::{Applies, Os};
use crate::model::{Dep, DepKind, Package, Target, TargetKind, Workspace};
use crate::starlark::{Rule, Value, list, quote};
use crate::worlds::{Resolved, World, resolve_world};

/// The rules of the shared generator, for the root package of a workspace with `[root] gazelle = true`.
const DEFS: &str = "@euhedral_build//bazel-gen:defs.bzl";

const CARGO_TOOLCHAIN: &str = "@rules_rust//cargo:current_cargo_toolchain";

/// Integration tests read `CARGO_TARGET_TMPDIR` at compile time; under Bazel they use `TEST_TMPDIR`, so this path is
/// never used.
const UNUSED_TMPDIR: &str = "unused-under-bazel";

struct Gen<'a> {
    ws: &'a Workspace,
    worlds: BTreeMap<World, Resolved>,
    owners: BTreeMap<(World, String), World>,
}

/// Every BUILD file of the workspace, by path relative to the root (`BUILD.bazel`, `example-core/BUILD.bazel`).
pub fn generate(ws: &Workspace) -> Result<BTreeMap<String, String>, String> {
    let mut worlds = BTreeMap::new();
    for world in World::built(&ws.config) {
        worlds.insert(world, resolve_world(ws, world)?);
    }
    for (name, settings) in &ws.config.packages {
        let sets = !settings.build_script_data.is_empty()
            || !settings.build_script_env.is_empty()
            || !settings.build_script_tags.is_empty();
        if sets && ws.package(name).is_none_or(|p| p.build_script.is_none()) {
            return Err(format!(
                "bazel_gen.toml: [packages.{name}] sets a build_script_* key, but {name} has no build script"
            ));
        }
    }
    let mut generator = Gen {
        ws,
        worlds,
        owners: BTreeMap::new(),
    };
    generator.check_card_tests(&World::built(&ws.config))?;
    let mut files = BTreeMap::new();
    let mut aliases = Vec::new();
    for package in &ws.packages {
        files.insert(
            format!("{}/BUILD.bazel", package.dir),
            generator.package(package, &mut aliases)?,
        );
    }
    files.insert("BUILD.bazel".to_owned(), root(ws, &aliases));
    for (owner, wanted) in &ws.exports {
        let path = if owner.is_empty() {
            "BUILD.bazel".to_owned()
        } else {
            format!("{owner}/BUILD.bazel")
        };
        let text = files.get_mut(&path).ok_or_else(|| {
            format!(
                "{path}: the generator writes no BUILD file for the package that owns {}",
                wanted.keys().next().map_or("", String::as_str)
            )
        })?;
        for (file, consumers) in wanted {
            // The root's own exports already cover the file.
            if owner.is_empty() && ws.config.exports.contains(file) {
                continue;
            }
            // The root's `workspace_sources` lists the file for the stale test.
            let mut visibility: Vec<String> = consumers
                .iter()
                .chain(std::iter::once(&String::new()))
                .filter(|c| *c != owner)
                .map(|c| format!("//{c}:__pkg__"))
                .collect();
            visibility.sort();
            visibility.dedup();
            text.push_str(&format!(
                "\n# Named by a `#[path]` in another package.\nexports_files({}, visibility = {})\n",
                list(std::slice::from_ref(file), 0),
                list(&visibility, 0)
            ));
        }
    }
    Ok(files)
}

/// The first line of every generated file, with its line break.
fn header(ws: &Workspace) -> String {
    format!("{}\n", ws.config.header)
}

fn root(ws: &Workspace, aliases: &[(String, String)]) -> String {
    let config = &ws.config;
    let mut out = header(ws);
    if config.gazelle {
        out.push_str(&format!(
            "\nload({}, \"bazel_gen\", \"bazel_gen_stale_test\")\n",
            quote(DEFS)
        ));
    }
    out.push_str("\n# The root files crate_universe reads, and the root files a test or binary names in its `data`.\n");
    let mut exports: Vec<String> = config.exports.clone();
    for package in &ws.packages {
        for extras in package.extras.values() {
            exports.extend(
                extras
                    .data
                    .iter()
                    .filter_map(|label| label.strip_prefix("//:"))
                    .map(str::to_owned),
            );
        }
    }
    exports.sort();
    exports.dedup();
    out.push_str(&format!("exports_files({})\n", list(&exports, 0)));
    for (name, actual) in aliases {
        out.push('\n');
        let mut rule = Rule::new("alias", name);
        rule.attr("actual", Value::Str(actual.clone()));
        out.push_str(&rule.render());
    }
    if config.gazelle {
        let config_file: Vec<(&'static str, Value)> = ws
            .config_file
            .iter()
            .map(|file| ("config", Value::Str(file.clone())))
            .collect();
        out.push_str("\n# `bazel run //:gazelle` writes the BUILD files from the Cargo manifests (`-- -mode=diff` only checks).\n");
        out.push_str(&Rule::new("bazel_gen", "gazelle").render());
        out.push_str(if config.lock_check {
            if config.crate_lock.is_some() {
                "\n# Fails when a BUILD file, or the crate lock, is out of date with the Cargo manifests and Cargo.lock.\n"
            } else {
                "\n# Fails when a BUILD file, or MODULE.bazel.lock's record of the manifests, is out of date with the manifests.\n"
            }
        } else {
            "\n# Fails when a BUILD file is out of date with the Cargo manifests.\n"
        });
        let mut rule = Rule::new("bazel_gen_stale_test", "bazel_gen_stale_test");
        for (key, value) in config_file {
            rule.attr(key, value);
        }
        if config.lock_check {
            let lock = config.crate_lock.as_deref().unwrap_or("MODULE.bazel.lock");
            rule.attr("lock", Value::Str(lock.to_owned()));
        }
        out.push_str(&rule.render());
    }
    for (name, globs) in &config.filegroups {
        out.push_str(
            "\n# A filegroup of root files from bazel_gen.toml, for the tests that name it.\n",
        );
        let mut rule = Rule::new("filegroup", name);
        rule.attr(
            "srcs",
            Value::Glob {
                include: globs.clone(),
                exclude: Vec::new(),
                allow_empty: false,
            },
        )
        .attr("visibility", public());
        out.push_str(&rule.render());
    }
    out.push_str(
        "\n# Every member's manifest, BUILD file and target files, for the tests that read the workspace's sources.\n",
    );
    let mut sources = vec!["BUILD.bazel".to_owned(), "Cargo.toml".to_owned()];
    if let Some(file) = &ws.config_file {
        sources.push(file.clone());
    }
    sources.extend(ws.packages.iter().map(|p| format!("//{}:sources", p.dir)));
    // The files a `#[path]` names outside every member's `sources`, which the stale test reads too.
    for (owner, files) in &ws.exports {
        let is_member = ws.packages.iter().any(|p| &p.dir == owner);
        for file in files
            .keys()
            .filter(|f| !(is_member && f.starts_with("src/")))
        {
            let label = if owner.is_empty() {
                file.clone()
            } else {
                format!("//{owner}:{file}")
            };
            if !sources.contains(&label) {
                sources.push(label);
            }
        }
    }
    let mut rule = Rule::new("filegroup", "workspace_sources");
    rule.attr("srcs", Value::List(sources))
        .attr("visibility", public());
    out.push_str(&rule.render());
    out
}

fn public() -> Value {
    Value::List(vec!["//visibility:public".to_owned()])
}

/// The tags of a build script target: the configured ones, `external` with `gpu` (a target that opens the card is not
/// sandboxed from it), and `no-cache` and `no-remote-cache` when an environment value is an absolute path (the script
/// reads a tree outside the workspace, which the action's inputs do not name, so a cached result would be stale).
fn script_tags(extra: &crate::config::PackageConfig) -> Vec<String> {
    let mut tags = extra.build_script_tags.clone();
    if tags.iter().any(|t| t == "gpu") {
        tags.push("external".to_owned());
    }
    if extra
        .build_script_env
        .iter()
        .any(|(_, v)| v.starts_with('/'))
    {
        tags.push("no-cache".to_owned());
        tags.push("no-remote-cache".to_owned());
    }
    tags.sort();
    tags.dedup();
    tags
}

/// The package files the `sources` filegroup gathers: the manifest and the target directories, and the package's own
/// files outside them that its sources name (`include_str!("../schema.sql")`, a build script's `kernels/x`), which the
/// stale test's generator run reads and so needs in its runfiles.
fn sources_include(package: &Package) -> Vec<String> {
    let mut include = strings(&[
        "BUILD.bazel",
        "Cargo.toml",
        "README.md",
        "benches/**",
        "build.rs",
        "examples/**",
        "src/**",
        "tests/**",
    ]);
    let named = package
        .path_modules
        .iter()
        .chain(package.local_data.values().flatten())
        .chain(package.local_srcs.values().flatten())
        .chain(package.build_data.iter().filter(|l| !l.starts_with("//")));
    for file in named {
        let covered = ["benches", "examples", "src", "tests"]
            .iter()
            .any(|dir| file.starts_with(&format!("{dir}/")));
        if !covered && !include.contains(file) {
            include.push(file.clone());
        }
    }
    include.sort();
    include
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

/// The dependency labels of one target, merged by label (an entry on Linux from one edge and on every OS from another
/// is on every OS).
#[derive(Default)]
struct DepSet {
    labels: BTreeMap<String, BTreeSet<Os>>,
    /// The names the crate's code uses for each label when they differ from the target's own (the `aliases` attribute),
    /// with `-` as `_` as Cargo does.
    aliases: BTreeMap<String, String>,
    /// The manifest keys that reach each label: two keys on one label would give the code two names for one crate.
    keys: BTreeMap<String, BTreeSet<String>>,
}

impl DepSet {
    fn add(&mut self, label: String, applies: &Applies) {
        let oses: Vec<Os> = match applies {
            Applies::Always => Os::ALL.to_vec(),
            Applies::On(oses) => oses.clone(),
        };
        if oses.is_empty() {
            return;
        }
        self.labels.entry(label).or_default().extend(oses);
    }

    /// Adds the label of `dep`, with the name its manifest key gives it.
    fn add_dep(&mut self, label: String, dep: &Dep) {
        self.keys
            .entry(label.clone())
            .or_default()
            .insert(dep.key().to_owned());
        if let Some(alias) = dep.alias() {
            self.aliases.insert(label.clone(), alias);
        }
        self.add(label, &dep.applies);
    }

    /// Refuses a label two manifest keys of `package` reach: `aliases` maps a label to one name.
    fn check(&self, package: &Package, target: &str) -> Result<(), String> {
        for (label, keys) in &self.keys {
            if keys.len() > 1 {
                let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
                return Err(format!(
                    "{}: the {target} depends on {label} under the keys `{}` and `{}`: one label has one name in the \
                     crate; map the renamed key to its own world in [external.crates] of bazel_gen.toml or drop one",
                    package.dir, keys[0], keys[1]
                ));
            }
        }
        Ok(())
    }

    /// The `aliases` attribute: the renamed labels of `self`.
    fn aliases(&self) -> Value {
        Value::Dict(
            self.aliases
                .iter()
                .filter(|(label, _)| self.labels.contains_key(*label))
                .map(|(label, alias)| (label.clone(), alias.clone()))
                .collect(),
        )
    }

    /// Sets `deps` and `aliases` of `rule`.
    fn set(&self, rule: &mut Rule) {
        rule.attr("deps", self.value())
            .attr("aliases", self.aliases());
    }

    /// The entries of `self` on the OSes `other` does not have them on.
    fn without(&self, other: &DepSet) -> DepSet {
        DepSet {
            aliases: self.aliases.clone(),
            keys: self.keys.clone(),
            labels: self
                .labels
                .iter()
                .filter_map(|(label, oses)| {
                    let left: BTreeSet<Os> = match other.labels.get(label) {
                        Some(theirs) => oses.difference(theirs).copied().collect(),
                        None => oses.clone(),
                    };
                    (!left.is_empty()).then(|| (label.clone(), left))
                })
                .collect(),
        }
    }

    fn value(&self) -> Value {
        let mut always = Vec::new();
        let mut by_os: Vec<(Os, Vec<String>)> =
            Os::ALL.iter().map(|os| (*os, Vec::new())).collect();
        for (label, oses) in &self.labels {
            if oses.len() == Os::ALL.len() {
                always.push(label.clone());
            } else {
                for (os, deps) in &mut by_os {
                    if oses.contains(os) {
                        deps.push(label.clone());
                    }
                }
            }
        }
        Value::Deps { always, by_os }
    }
}

impl Gen<'_> {
    fn resolved(&self, world: World) -> &Resolved {
        &self.worlds[&world]
    }

    /// The world whose library target `package` uses in `world`: the earliest world with the same features and the same
    /// first-party dependency targets.
    fn owner(&mut self, world: World, package: &str) -> Result<World, String> {
        if let Some(owner) = self.owners.get(&(world, package.to_owned())) {
            return Ok(*owner);
        }
        let owner = match world {
            World::Production => World::Production,
            _ => {
                let previous = World::ALL[world as usize - 1];
                let same_features = self.resolved(world).features.get(package)
                    == self.resolved(previous).features.get(package);
                let here = self.lib_deps(world, package)?;
                let there = self.lib_deps(previous, package)?;
                let same_external =
                    self.external_deps(world, package) == self.external_deps(previous, package);
                let same_build =
                    self.build_labels(world, package)? == self.build_labels(previous, package)?;
                if same_features && here == there && same_external && same_build {
                    self.owner(previous, package)?
                } else {
                    world
                }
            }
        };
        self.owners.insert((world, package.to_owned()), owner);
        Ok(owner)
    }

    /// The members `package`'s library links in `world`, with the world each one's target comes from.
    fn lib_deps(
        &mut self,
        world: World,
        package: &str,
    ) -> Result<BTreeSet<(String, World)>, String> {
        let pkg = self.ws.package(package).ok_or("unknown member")?;
        let members: Vec<String> = pkg
            .deps
            .iter()
            .filter(|d| d.kind == DepKind::Normal && self.resolved(world).dep_on(package, d))
            .filter_map(|d| d.member.clone())
            .collect();
        let mut out = BTreeSet::new();
        for member in members {
            let owner = self.owner(world, &member)?;
            out.insert((member, owner));
        }
        Ok(out)
    }

    /// The labels of the crates of other modules `package`'s library links in `world`.
    fn external_deps(&self, world: World, package: &str) -> BTreeSet<String> {
        let Some(pkg) = self.ws.package(package) else {
            return BTreeSet::new();
        };
        pkg.deps
            .iter()
            .filter(|d| {
                d.kind == DepKind::Normal
                    && d.member.is_none()
                    && self.resolved(world).dep_on(package, d)
            })
            .filter_map(|d| self.ws.config.external_dep_label(d, world as usize))
            .collect()
    }

    /// The labels (and names) of `package`'s build-dependencies in `world`.
    fn build_labels(
        &mut self,
        world: World,
        package: &str,
    ) -> Result<BTreeSet<(String, String)>, String> {
        let ws = self.ws;
        let Some(pkg) = ws.package(package).filter(|p| p.build_script.is_some()) else {
            return Ok(BTreeSet::new());
        };
        let set = self.dep_set(world, pkg, &|kind| kind == DepKind::Build)?;
        Ok(set
            .labels
            .keys()
            .map(|label| {
                (
                    label.clone(),
                    set.aliases.get(label).cloned().unwrap_or_default(),
                )
            })
            .collect())
    }

    fn lib_name(&mut self, world: World, package: &Package) -> Result<String, String> {
        let owner = self.owner(world, &package.name)?;
        Ok(format!(
            "{}{}",
            package.crate_name(),
            owner.lib_suffix(&self.ws.config)
        ))
    }

    /// The label of `member`'s library in `world`, seen from the package in `from_dir`.
    fn lib_label(&mut self, world: World, member: &str, from_dir: &str) -> Result<String, String> {
        let pkg = self.ws.package(member).ok_or("unknown member")?;
        if pkg.lib().is_none() {
            return Err(format!("{member} is depended on but has no library"));
        }
        let name = self.lib_name(world, pkg)?;
        Ok(if pkg.dir == from_dir {
            format!(":{name}")
        } else {
            format!("//{}:{name}", pkg.dir)
        })
    }

    /// The dependencies of `package`'s targets in `world`: the normal ones, and with `dev` also the dev-dependencies, and
    /// the package's build script when it has one. The package's own library (a self dev-dependency) is left out: the
    /// world's library already has its features.
    fn deps(&mut self, world: World, package: &Package, dev: bool) -> Result<DepSet, String> {
        let mut set = self.dep_set(world, package, &|kind| {
            kind == DepKind::Normal || (dev && kind == DepKind::Dev)
        })?;
        if package.build_script.is_some() {
            let name = self.build_script_name(world, package)?;
            set.add(format!(":{name}"), &Applies::Always);
        }
        Ok(set)
    }

    /// The target of `package`'s build script in `world`: `build_script` and a suffix per world that differs, as the
    /// library's.
    fn build_script_name(&mut self, world: World, package: &Package) -> Result<String, String> {
        let owner = self.owner(world, &package.name)?;
        Ok(format!("build_script{}", owner.lib_suffix(&self.ws.config)))
    }

    /// The dependencies of the kinds `kinds` accepts that are part of `package`'s build in `world`.
    fn dep_set(
        &mut self,
        world: World,
        package: &Package,
        kinds: &dyn Fn(DepKind) -> bool,
    ) -> Result<DepSet, String> {
        let mut set = DepSet::default();
        let deps: Vec<Dep> = package
            .deps
            .iter()
            .filter(|d| kinds(d.kind) && self.resolved(world).dep_on(&package.name, d))
            .cloned()
            .collect();
        for dep in deps {
            match &dep.member {
                Some(member) if *member == package.name => {}
                Some(member) => {
                    let label = self.lib_label(world, member, &package.dir)?;
                    set.add_dep(label, &dep);
                }
                None => {
                    let label = match self.ws.config.external_dep_label(&dep, world as usize) {
                        Some(label) => label,
                        None => format!("@{}//:{}", self.ws.config.crates, dep.name),
                    };
                    set.add_dep(label, &dep);
                }
            }
        }
        set.check(
            package,
            if kinds(DepKind::Build) {
                "build script"
            } else {
                "package"
            },
        )?;
        Ok(set)
    }

    /// What a test of `package` in `world` links that opens the card: `(firm, soft)`. Firm: the device member with a device
    /// feature, or (in the device world) a crate of another module, which `card_tests = false` does not undo. Soft: any
    /// other crate resolved with a device feature (a member, or a third-party crate directly or through the crates it
    /// links) or a package marked `card_tests = true`.
    fn card_links(&self, world: World, package: &Package) -> (bool, bool) {
        let config = &self.ws.config;
        let marked = |name: &str| config.package(name).card_tests == Some(true);
        let resolved = self.resolved(world);
        let device = config.device.as_ref();
        let opens_with = |features: &BTreeSet<String>| {
            device.is_some_and(|d| d.all_features().any(|f| features.contains(f)))
        };
        // In the device world every crate of another module is its device build, which may open the card: a test that
        // links one is tagged, whether or not `device.external` names the crate.
        let external_on = world == World::CudaTesting
            && device.is_some_and(|d| !d.external.is_empty() || !config.external.is_empty());
        // A third-party crate opens the card when it, or a crate it links (the crate hub's lock), is resolved with a
        // device feature: by the lock, or by a member's dependency asking for it.
        let third_on = |dep: &Dep| {
            if dep.member.is_some() || device.is_none() {
                return false;
            }
            let mut seen = BTreeSet::new();
            let mut stack = vec![dep.name.clone()];
            while let Some(name) = stack.pop() {
                if !seen.insert(name.clone()) {
                    continue;
                }
                let asked = resolved.third_party.get(&name).is_some_and(&opens_with);
                let locked = self.ws.locked.as_ref();
                if asked
                    || locked
                        .and_then(|l| l.features.get(&name))
                        .is_some_and(&opens_with)
                {
                    return true;
                }
                stack.extend(
                    locked
                        .and_then(|l| l.deps.get(&name))
                        .into_iter()
                        .flatten()
                        .cloned(),
                );
            }
            false
        };
        let (mut firm, mut soft) = (false, false);
        let mut seen = BTreeSet::new();
        let mut stack = vec![package.name.clone()];
        let mut visit = |dep: &Dep, stack: &mut Vec<String>| {
            stack.extend(dep.member.clone());
            if external_on
                && dep.member.is_none()
                && (device.is_some_and(|d| d.external.contains(&dep.name))
                    || config.external_key(dep).is_some())
            {
                firm = true;
            }
            if third_on(dep) {
                soft = true;
            }
        };
        // The test's own dev-dependencies, then every library's normal dependencies.
        for dep in &package.deps {
            if dep.kind == DepKind::Dev && resolved.dep_on(&package.name, dep) {
                visit(dep, &mut stack);
            }
        }
        while let Some(name) = stack.pop() {
            if !seen.insert(name.clone()) {
                continue;
            }
            if let Some(pkg) = self.ws.package(&name) {
                for dep in &pkg.deps {
                    if dep.kind == DepKind::Normal && resolved.dep_on(&name, dep) {
                        visit(dep, &mut stack);
                    }
                }
            }
        }
        for name in &seen {
            let on = resolved.features.get(name).is_some_and(&opens_with);
            if on && device.is_some_and(|d| d.member.as_deref() == Some(name.as_str())) {
                firm = true;
            } else if on || marked(name) {
                soft = true;
            }
        }
        (firm, soft)
    }

    /// Whether a test of `package` in `world` opens the card (`gpu` and `external`); `card_tests = false` on the package
    /// undoes only what `card_links` calls soft.
    fn opens_the_card(&self, world: World, package: &Package) -> bool {
        let (firm, soft) = self.card_links(world, package);
        firm || (soft && self.ws.config.package(&package.name).card_tests != Some(false))
    }

    /// Refuses a `card_tests` that cannot do what it says: on a name that is no member, or a `false` on a package whose
    /// tests nothing would tag in any world (it would hide nothing).
    fn check_card_tests(&self, worlds: &[World]) -> Result<(), String> {
        for (name, settings) in &self.ws.config.packages {
            let Some(answer) = settings.card_tests else {
                continue;
            };
            let Some(package) = self.ws.package(name) else {
                return Err(format!(
                    "packages.{name}.card_tests: {name} is not a workspace member"
                ));
            };
            if !answer
                && !worlds
                    .iter()
                    .filter(|w| w.builds_tests())
                    .any(|w| matches!(self.card_links(*w, package), (false, true)))
            {
                return Err(format!(
                    "packages.{name}.card_tests = false: no test of {name} links a crate resolved with a device feature, \
                     or the device world tags it regardless; the key changes nothing (remove it)"
                ));
            }
        }
        Ok(())
    }

    /// What decides whether a world needs its own test targets: the package's features and every member target its tests
    /// link.
    /// What a binary of `package` is built from in `world`: its features and dependency labels.
    fn bin_key(&mut self, world: World, package: &Package) -> Result<String, String> {
        let mut deps = self.deps(world, package, false)?;
        if package.lib().is_some() {
            let lib = self.lib_name(world, package)?;
            deps.add(format!(":{lib}"), &Applies::Always);
        }
        Ok(format!(
            "{:?} {:?}",
            self.resolved(world).features_of(&package.name),
            deps.labels
        ))
    }

    /// The `rust_binary` rule named `name` for `bin` built in `world`, and its dependencies (not yet set on it).
    fn bin_rule(
        &mut self,
        world: World,
        package: &Package,
        bin: &Target,
        name: &str,
    ) -> Result<(Rule, DepSet), String> {
        let extra = self.ws.config.package(&package.name);
        let mut rule = Rule::new("rust_binary", name);
        let mut deps = self.deps(world, package, false)?;
        if package.lib().is_some() {
            let lib = self.lib_name(world, package)?;
            deps.add(format!(":{lib}"), &Applies::Always);
        }
        rule.attr("crate_name", Value::Str(bin.name.replace('-', "_")))
            .attr("srcs", bin_srcs(package, bin)?)
            .attr("crate_root", Value::Str(bin.src.clone()))
            .attr(
                "compile_data",
                compile_data(
                    "src",
                    &[package.path_modules.clone(), package.local_data("src")].concat(),
                    &extra.compile_data,
                    &package.foreign_data,
                ),
            )
            .attr(
                "crate_features",
                Value::List(self.resolved(world).features_of(&package.name)),
            );
        Ok((rule, deps))
    }

    fn test_key(&mut self, world: World, package: &Package) -> Result<String, String> {
        let deps = self.deps(world, package, true)?;
        let lib = match package.lib() {
            Some(_) => self.lib_name(world, package)?,
            None => String::new(),
        };
        Ok(format!(
            "{:?} {lib} {:?}",
            self.resolved(world).features.get(&package.name),
            deps.labels
        ))
    }

    fn package(
        &mut self,
        package: &Package,
        aliases: &mut Vec<(String, String)>,
    ) -> Result<String, String> {
        let config = &self.ws.config;
        let extra = config.package(&package.name);
        let worlds = World::built(config);
        let mut rules: Vec<(Option<String>, String)> = Vec::new();
        let mut names = BTreeSet::new();
        let mut bin_keys: BTreeMap<String, String> = BTreeMap::new();
        let mut used = BTreeSet::new();
        let mut push = |rule: Rule,
                        comment: Option<String>,
                        names: &mut BTreeSet<String>|
         -> Result<(), String> {
            let Some((_, Value::Str(name))) = rule.attrs.first() else {
                unreachable!("a rule starts with its name")
            };
            if !names.insert(name.clone()) {
                return Err(format!(
                    "{}: two targets are named `{name}`; rename the Cargo target",
                    package.dir
                ));
            }
            if rule.kind.starts_with("rust_") || rule.kind == "cargo_build_script" {
                used.insert(rule.kind);
            }
            rules.push((comment, rule.render()));
            Ok(())
        };

        // Lints.
        let mut lints = Rule::new("rust_lint_config", "lints");
        for tool in ["rust", "clippy", "rustdoc"] {
            let entries: Vec<(String, String)> = package
                .lints
                .iter()
                .filter(|l| l.tool == tool)
                .map(|l| (l.name.clone(), l.level.clone()))
                .collect();
            let key = match tool {
                "rust" => "rustc",
                other => other,
            };
            lints.attr(key, Value::Dict(entries));
        }
        let mut check_cfg: Vec<(String, Vec<String>)> = vec![
            ("docsrs".to_owned(), Vec::new()),
            (
                "feature".to_owned(),
                package.features.keys().cloned().collect(),
            ),
            ("test".to_owned(), Vec::new()),
        ];
        for name in &package.check_cfg {
            if !check_cfg.iter().any(|(n, _)| n == name) {
                check_cfg.push((name.clone(), Vec::new()));
            }
        }
        check_cfg.sort();
        lints.attr("rustc_check_cfg", Value::ListDict(check_cfg));
        push(
            lints,
            Some(
                "The [lints] table, and the cfg names Cargo declares (features, test, docsrs) with the crate's own.".to_owned(),
            ),
            &mut names,
        )?;

        // Sources.
        let mut sources = Rule::new("filegroup", "sources");
        sources
            .attr(
                "srcs",
                Value::Glob {
                    include: sources_include(package),
                    exclude: Vec::new(),
                    allow_empty: true,
                },
            )
            .attr("visibility", public());
        push(
            sources,
            Some("The manifest and target files, gathered at //:workspace_sources.".to_owned()),
            &mut names,
        )?;

        let bins: Vec<&Target> = package
            .targets
            .iter()
            .filter(|t| t.kind == TargetKind::Bin)
            .collect();

        // The build script, one per world that differs; aliases for the others.
        if let Some(script) = &package.build_script {
            for world in worlds.iter().copied() {
                let name = format!("build_script{}", world.lib_suffix(config));
                let owner = self.owner(world, &package.name)?;
                if owner != world {
                    let mut alias = Rule::new("alias", &name);
                    alias
                        .attr(
                            "actual",
                            Value::Str(format!(":build_script{}", owner.lib_suffix(config))),
                        )
                        .attr("visibility", public());
                    push(alias, Some(world_comment(config, world, true)), &mut names)?;
                    continue;
                }
                let mut rule = Rule::new("cargo_build_script", &name);
                rule.attr("srcs", Value::List(vec![script.src.clone()]))
                    .attr("crate_root", Value::Str(script.src.clone()))
                    .attr("edition", Value::Str(script.edition.clone()))
                    .attr(
                        "crate_features",
                        Value::List(self.resolved(world).features_of(&package.name)),
                    )
                    .attr(
                        "build_script_env",
                        Value::Dict(extra.build_script_env.clone()),
                    )
                    .attr("tags", Value::List(script_tags(&extra)))
                    .attr("version", Value::Str(package.version.clone()));
                if !package.build_data.is_empty() {
                    rule.attr("compile_data", Value::List(package.build_data.clone()));
                }
                if let Some(links) = &package.links {
                    rule.attr("links", Value::Str(links.clone()));
                }
                if !extra.build_script_data.is_empty() {
                    rule.attr(
                        "data",
                        Value::Glob {
                            include: extra.build_script_data.clone(),
                            exclude: Vec::new(),
                            allow_empty: true,
                        },
                    );
                }
                self.dep_set(world, package, &|kind| kind == DepKind::Build)?
                    .set(&mut rule);
                let comment = match world {
                    World::Production => {
                        "The build script, run by Cargo's rules: its OUT_DIR and cfgs reach the package's targets."
                            .to_owned()
                    }
                    _ => world_comment(config, world, false),
                };
                push(rule, Some(comment), &mut names)?;
            }
        }

        // Libraries, one per world that differs; aliases for the others.
        if let Some(lib) = package.lib() {
            let lib_src = lib
                .src
                .strip_prefix("src/")
                .map(|_| lib.src.clone())
                .ok_or_else(|| {
                    format!(
                        "{}: library root {} is outside src/: not modelled",
                        package.dir, lib.src
                    )
                })?;
            let mut exclude = Vec::new();
            for bin in &bins {
                exclude.push(if bin.src == "src/main.rs" {
                    "src/main.rs".to_owned()
                } else {
                    "src/bin/**".to_owned()
                });
            }
            exclude.sort();
            exclude.dedup();
            for world in worlds.iter().copied() {
                let name = format!("{}{}", package.crate_name(), world.lib_suffix(config));
                let owner = self.owner(world, &package.name)?;
                if owner != world {
                    let mut alias = Rule::new("alias", &name);
                    alias
                        .attr(
                            "actual",
                            Value::Str(format!(
                                ":{}{}",
                                package.crate_name(),
                                owner.lib_suffix(config)
                            )),
                        )
                        .attr("visibility", public());
                    push(alias, Some(world_comment(config, world, true)), &mut names)?;
                    continue;
                }
                let mut rule = Rule::new("rust_library", &name);
                rule.attr("crate_name", Value::Str(package.crate_name()))
                    .attr(
                        "srcs",
                        foreign(
                            package,
                            Value::Glob {
                                include: strings(&["src/**/*.rs"]),
                                exclude: exclude.clone(),
                                allow_empty: false,
                            },
                        ),
                    )
                    .attr("crate_root", Value::Str(lib_src.clone()))
                    .attr(
                        "compile_data",
                        compile_data(
                            "src",
                            &[package.path_modules.clone(), package.local_data("src")].concat(),
                            &extra.compile_data,
                            &package.foreign_data,
                        ),
                    )
                    .attr(
                        "crate_features",
                        Value::List(self.resolved(world).features_of(&package.name)),
                    );
                self.deps(world, package, false)?.set(&mut rule);
                common(&mut rule, package, lib);
                rule.attr("visibility", public());
                push(rule, Some(world_comment(config, world, false)), &mut names)?;
            }
        }

        // Binaries: the first world with their required features.
        for bin in &bins {
            let Some(world) = worlds
                .iter()
                .copied()
                .find(|w| self.satisfied(*w, package, bin))
            else {
                return Err(required_error(package, bin));
            };
            let (mut rule, deps) = self.bin_rule(world, package, bin, &bin.name)?;
            bin_keys.insert(bin.name.clone(), self.bin_key(world, package)?);
            deps.set(&mut rule);
            common(&mut rule, package, bin);
            let extras = package.extras("bin", &bin.name);
            run_extras(&mut rule, package, &extras, None);
            rule.attr("visibility", public());
            if let Some(alias) = &extras.root_alias {
                aliases.push((alias.clone(), format!("//{}:{}", package.dir, bin.name)));
            }
            push(rule, None, &mut names)?;
        }

        // Tests, in each test world whose features or linked targets differ from the last world that has them.
        let mut last_key: BTreeMap<String, String> = BTreeMap::new();
        let mut examples_done = BTreeSet::new();
        for world in worlds.iter().copied().filter(|w| w.builds_tests()) {
            let key = self.test_key(world, package)?;
            let infix = world.test_infix(config);
            let tags = if self.opens_the_card(world, package) {
                strings(&["external", "gpu"])
            } else {
                Vec::new()
            };
            let features = Value::List(self.resolved(world).features_of(&package.name));
            let normal = self.deps(world, package, false)?;
            let all = self.deps(world, package, true)?;
            let mut changed = |id: String| -> bool {
                if last_key.get(&id) == Some(&key) {
                    false
                } else {
                    last_key.insert(id, key.clone());
                    true
                }
            };
            if let Some(lib) = package.lib() {
                let lib_name = self.lib_name(world, package)?;
                let dev_only = all.without(&normal);
                if lib.test && changed("lib".to_owned()) {
                    let mut rule = Rule::new("rust_test", &format!("unit{infix}_test"));
                    rule.attr("crate", Value::Str(format!(":{lib_name}")))
                        .attr("crate_features", features.clone())
                        .attr("tags", Value::List(tags.clone()));
                    dev_only.set(&mut rule);
                    common(&mut rule, package, lib);
                    push(
                        rule,
                        Some("The library's unit tests (#[cfg(test)]).".to_owned()),
                        &mut names,
                    )?;
                }
                if lib.doctest && changed("doc".to_owned()) {
                    let mut rule = Rule::new("rust_doc_test", &format!("doc{infix}_test"));
                    rule.attr("crate", Value::Str(format!(":{lib_name}")))
                        .attr("crate_features", features.clone())
                        .attr("tags", Value::List(tags.clone()));
                    // rust_doc_test has no `aliases`: a renamed dev-only dependency would be under its own name in
                    // the doctests, not the manifest key's.
                    if let Value::Dict(renamed) = dev_only.aliases()
                        && let Some((label, alias)) = renamed.first()
                    {
                        return Err(format!(
                            "{}: the doctests depend on {label} as `{alias}` (a renamed dev-dependency), and \
                             rust_doc_test has no `aliases`: make it a normal dependency, or set doctest = false \
                             on the library",
                            package.dir
                        ));
                    }
                    rule.attr("deps", dev_only.value());
                    push(
                        rule,
                        Some("The doctests, the README's included.".to_owned()),
                        &mut names,
                    )?;
                }
            }
            let mut with_lib = self.deps(world, package, true)?;
            if package.lib().is_some() {
                let lib = self.lib_name(world, package)?;
                with_lib.add(format!(":{lib}"), &Applies::Always);
            }
            // The binaries an integration test runs are the ones built in its own world: the production binary when it
            // is built the same way, else a copy of its own.
            let mut test_bins: Vec<(String, String)> = Vec::new();
            for bin in &bins {
                if !self.satisfied(world, package, bin) {
                    continue;
                }
                if bin_keys.get(&bin.name) == Some(&self.bin_key(world, package)?) {
                    test_bins.push((bin.name.clone(), bin.name.clone()));
                    continue;
                }
                let name = format!("{}{}", bin.name, world.lib_suffix(config));
                let (mut rule, deps) = self.bin_rule(world, package, bin, &name)?;
                deps.set(&mut rule);
                common(&mut rule, package, bin);
                push(
                    rule,
                    Some(format!("`{}` as the tests of this world run it.", bin.name)),
                    &mut names,
                )?;
                test_bins.push((bin.name.clone(), name));
            }
            for target in &package.targets {
                let satisfied = self.satisfied(world, package, target);
                match target.kind {
                    TargetKind::Lib => {}
                    TargetKind::Test | TargetKind::Bin if !target.test || !satisfied => {}
                    TargetKind::Test | TargetKind::Bin => {
                        if !changed(format!("{:?} {}", target.kind, target.name)) {
                            continue;
                        }
                        let crate_name = target.name.replace('-', "_");
                        let (name, srcs, extras) = if target.kind == TargetKind::Test {
                            (
                                format!("{crate_name}{infix}_test"),
                                test_srcs(package, target)?,
                                package.extras("test", &target.name),
                            )
                        } else {
                            (
                                format!("{crate_name}_bin{infix}_test"),
                                bin_srcs(package, target)?,
                                Default::default(),
                            )
                        };
                        let mut rule = Rule::new("rust_test", &name);
                        let mut tags = tags.clone();
                        if let Some(cpus) = extras.cpus {
                            tags.push(format!("cpu:{cpus}"));
                        }
                        tags.sort();
                        rule.attr("crate_name", Value::Str(crate_name))
                            .attr("srcs", srcs)
                            .attr("crate_root", Value::Str(target.src.clone()))
                            .attr(
                                "compile_data",
                                compile_data(
                                    if target.kind == TargetKind::Test {
                                        "tests"
                                    } else {
                                        "src"
                                    },
                                    &if target.kind == TargetKind::Test {
                                        package.local_data("tests")
                                    } else {
                                        [package.path_modules.clone(), package.local_data("src")]
                                            .concat()
                                    },
                                    &extra.compile_data,
                                    &[extra.test_data.clone(), package.foreign_data.clone()]
                                        .concat(),
                                ),
                            )
                            .attr("crate_features", features.clone())
                            .attr("tags", Value::List(tags));
                        with_lib.set(&mut rule);
                        common(&mut rule, package, target);
                        let mut extras = extras;
                        extras.data.extend(extra.test_data.iter().cloned());
                        let integration =
                            (target.kind == TargetKind::Test).then_some(&test_bins[..]);
                        run_extras(&mut rule, package, &extras, integration);
                        push(rule, None, &mut names)?;
                    }
                    TargetKind::Example if !satisfied || examples_done.contains(&target.name) => {}
                    TargetKind::Example => {
                        examples_done.insert(target.name.clone());
                        let mut rule = Rule::new("rust_binary", &target.name);
                        rule.attr("crate_name", Value::Str(target.name.replace('-', "_")))
                            .attr(
                                "srcs",
                                foreign_and_local(
                                    package,
                                    "examples",
                                    example_srcs(package, target)?,
                                ),
                            )
                            .attr("crate_root", Value::Str(target.src.clone()))
                            .attr(
                                "compile_data",
                                compile_data(
                                    "examples",
                                    &package.local_data("examples"),
                                    &extra.compile_data,
                                    &package.foreign_data,
                                ),
                            )
                            .attr("crate_features", features.clone());
                        with_lib.set(&mut rule);
                        common(&mut rule, package, target);
                        push(
                            rule,
                            Some(
                                "An example, built in the first world with its required features."
                                    .to_owned(),
                            ),
                            &mut names,
                        )?;
                    }
                }
            }
        }
        for target in &package.targets {
            let built = match target.kind {
                TargetKind::Lib | TargetKind::Bin => true,
                TargetKind::Example => examples_done.contains(&target.name),
                TargetKind::Test => {
                    !target.test
                        || last_key.contains_key(&format!("{:?} {}", target.kind, target.name))
                }
            };
            if !built {
                return Err(required_error(package, target));
            }
        }

        let mut out = header(self.ws);
        let loads: Vec<String> = used
            .iter()
            .filter(|k| **k != "cargo_build_script")
            .map(|k| quote(k))
            .collect();
        out.push_str(&format!(
            "\nload(\"@rules_rust//rust:defs.bzl\", {})\n",
            loads.join(", ")
        ));
        if used.contains("cargo_build_script") {
            out.push_str("load(\"@rules_rust//cargo:defs.bzl\", \"cargo_build_script\")\n");
        }
        for (comment, rule) in rules {
            out.push('\n');
            if let Some(comment) = comment {
                out.push_str(&format!("# {comment}\n"));
            }
            out.push_str(&rule);
        }
        Ok(out)
    }

    fn satisfied(&self, world: World, package: &Package, target: &Target) -> bool {
        target
            .required_features
            .iter()
            .all(|f| self.resolved(world).has(&package.name, f))
    }
}

fn required_error(package: &Package, target: &Target) -> String {
    format!(
        "{}: target {} needs features {:?}, which no world has: not modelled",
        package.dir, target.name, target.required_features
    )
}

fn world_comment(config: &crate::config::Config, world: World, alias: bool) -> String {
    match (world, alias) {
        (World::Production, _) => {
            "The production world: the crate's defaults, as a consumer gets it.".to_owned()
        }
        (World::Testing, false) => {
            "The testing world: `cargo test --workspace`'s features.".to_owned()
        }
        (World::Testing, true) => {
            "The testing world has the same features and dependencies.".to_owned()
        }
        (World::CudaTesting, alias) => {
            let feature = &config.device.as_ref().expect("a device world").feature;
            if alias {
                format!("The {feature} testing world has the same features and dependencies.")
            } else {
                format!("The {feature} testing world: the testing world plus `{feature}`.")
            }
        }
    }
}

/// The attributes every Rust target of the package has.
fn common(rule: &mut Rule, package: &Package, target: &Target) {
    rule.attr("edition", Value::Str(target.edition.clone()))
        .attr("lint_config", Value::Str(":lints".to_owned()))
        .attr("version", Value::Str(package.version.clone()));
}

/// The data and environment `[package.metadata.bazel]` asks for; `integration` adds what Cargo gives an integration test:
/// `CARGO_TARGET_TMPDIR` and `CARGO_BIN_EXE_<bin>` for each binary of the package, as (binary, target) pairs.
fn run_extras(
    rule: &mut Rule,
    package: &Package,
    extras: &crate::model::Extras,
    integration: Option<&[(String, String)]>,
) {
    let mut data = extras.data.clone();
    let mut rustc_env = Vec::new();
    let mut env = Vec::new();
    if extras.workspace_sources {
        data.push("//:workspace_sources".to_owned());
    }
    if extras.cargo {
        data.push(CARGO_TOOLCHAIN.to_owned());
        env.push(("CARGO".to_owned(), format!("$(rootpath {CARGO_TOOLCHAIN})")));
    }
    if extras.workspace_sources || !extras.data.is_empty() {
        // Relative to the runfiles root, where a test starts: the sandbox's execution root is gone by then.
        rustc_env.push(("CARGO_MANIFEST_DIR".to_owned(), package.dir.clone()));
    }
    if let Some(bins) = integration {
        rustc_env.push(("CARGO_TARGET_TMPDIR".to_owned(), UNUSED_TMPDIR.to_owned()));
        // Cargo gives an integration test the path of each of the package's binaries; here it is relative to the
        // runfiles root, where the test starts.
        for (bin, target) in bins {
            data.push(format!(":{target}"));
            rustc_env.push((
                format!("CARGO_BIN_EXE_{bin}"),
                format!("$(rootpath :{target})"),
            ));
        }
    }
    data.sort();
    data.dedup();
    rule.attr("data", Value::List(data))
        .attr("env", Value::Dict(env))
        .attr("rustc_env", Value::Dict(rustc_env));
}

/// The non-Rust files of `dir` and the README, with `extra` (module files outside `src/` the sources name by path), the
/// package's `compile_data` globs from bazel_gen.toml, and `labels` (its tests' `test_data`).
fn compile_data(dir: &str, extra: &[String], globs: &[String], labels: &[String]) -> Value {
    let mut include = vec!["README.md".to_owned(), format!("{dir}/**")];
    for item in extra.iter().chain(globs) {
        if !include.contains(item) {
            include.push(item.clone());
        }
    }
    Value::Plus(
        labels.to_vec(),
        Box::new(Value::Glob {
            include,
            exclude: vec![format!("{dir}/**/*.rs")],
            allow_empty: true,
        }),
    )
}

/// `dir/<name>.rs` with `dir/<name>/**/*.rs`, or `dir/<name>/main.rs` with its directory.
fn named_srcs(package: &Package, target: &Target, dir: &str) -> Result<Vec<String>, String> {
    let rest = target.src.strip_prefix(&format!("{dir}/")).ok_or_else(|| {
        format!(
            "{}: target {} has its root {} outside {dir}/: not modelled",
            package.dir, target.name, target.src
        )
    })?;
    if let Some(stem) = rest.strip_suffix(".rs").filter(|s| !s.contains('/')) {
        Ok(vec![
            format!("{dir}/{stem}.rs"),
            format!("{dir}/{stem}/**/*.rs"),
        ])
    } else if let Some(stem) = rest.strip_suffix("/main.rs").filter(|s| !s.contains('/')) {
        Ok(vec![format!("{dir}/{stem}/**/*.rs")])
    } else {
        Err(format!(
            "{}: target {} root {} is not modelled",
            package.dir, target.name, target.src
        ))
    }
}

/// `srcs` with the files other packages export for this package's `#[path]` attributes.
fn foreign(package: &Package, srcs: Value) -> Value {
    if package.foreign_srcs.is_empty() {
        srcs
    } else {
        Value::Plus(package.foreign_srcs.clone(), Box::new(srcs))
    }
}

/// [`foreign`], with the files of other directories that the package's `tests/` or `examples/` (`dir`) name.
fn foreign_and_local(package: &Package, dir: &str, srcs: Value) -> Value {
    let mut labels = package.foreign_srcs.clone();
    labels.extend(
        package
            .local_srcs
            .get(dir)
            .into_iter()
            .flatten()
            .map(|file| format!("//{}:{file}", package.dir)),
    );
    if labels.is_empty() {
        srcs
    } else {
        Value::Plus(labels, Box::new(srcs))
    }
}

fn bin_srcs(package: &Package, bin: &Target) -> Result<Value, String> {
    bin_srcs_own(package, bin).map(|srcs| foreign(package, srcs))
}

fn bin_srcs_own(package: &Package, bin: &Target) -> Result<Value, String> {
    if bin.src == "src/main.rs" {
        return Ok(Value::Glob {
            include: vec!["src/**/*.rs".to_owned()],
            exclude: vec!["src/bin/**".to_owned()],
            allow_empty: false,
        });
    }
    let mut include = named_srcs(package, bin, "src/bin")?;
    include.retain(|p| p.ends_with(".rs"));
    Ok(Value::Glob {
        include,
        exclude: Vec::new(),
        allow_empty: true,
    })
}

fn test_srcs(package: &Package, test: &Target) -> Result<Value, String> {
    test_srcs_own(package, test).map(|srcs| foreign_and_local(package, "tests", srcs))
}

fn test_srcs_own(package: &Package, test: &Target) -> Result<Value, String> {
    let mut include = named_srcs(package, test, "tests")?;
    // The shared modules (`tests/common`) of every integration test.
    include.retain(|p| !p.ends_with("/**/*.rs"));
    include.push("tests/*/**/*.rs".to_owned());
    include.dedup();
    // Bazel checks each pattern: most crates have no shared test modules.
    Ok(Value::Glob {
        include,
        exclude: Vec::new(),
        allow_empty: true,
    })
}

fn example_srcs(package: &Package, example: &Target) -> Result<Value, String> {
    Ok(Value::Glob {
        include: named_srcs(package, example, "examples")?,
        exclude: Vec::new(),
        allow_empty: true,
    })
}
