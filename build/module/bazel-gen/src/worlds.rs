//! The feature resolver for workspace crates, and the three worlds (docs/design/bazel.md, "Features").
//!
//! A world is one consistent choice of features for every member. Resolution follows Cargo's resolver for first-party
//! crates: a selected member gets its `default` feature (and the world's extras), every active dependency edge requests
//! its `features` (and `default` unless `default-features = false`), `dep:x` turns on an optional dependency, `x/f` turns
//! on `x` and its feature `f`, and dev-dependencies count only for the selected members and only when the world builds
//! tests. Third-party crates are `crate_universe`'s: the resolver only records which optional ones are on.

use std::collections::{BTreeMap, BTreeSet};

use crate::cfg::Applies;
use crate::config::Config;
use crate::model::{Dep, DepKind, Workspace};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum World {
    /// Each member's defaults, no dev-dependencies: what a consumer of the crate gets.
    Production = 0,
    /// `cargo test --workspace`: defaults plus every member's dev-dependency features.
    Testing = 1,
    /// Testing plus the device feature (`[device] feature`, `cuda`) on every member that has it.
    CudaTesting = 2,
}

impl World {
    pub const ALL: [World; 3] = [World::Production, World::Testing, World::CudaTesting];

    pub fn builds_tests(self) -> bool {
        self != World::Production
    }

    /// The worlds the configuration builds: the device world only with a `[device]`.
    pub fn built(config: &Config) -> Vec<World> {
        // The device world turns a feature on for the members that declare it: it exists when a member or a crate of
        // another module takes part. A `[device]` alone (a third-party crate resolved with the feature) builds none.
        let device_world = config
            .device
            .as_ref()
            .is_some_and(|d| d.member.is_some() || !config.external.is_empty());
        let count = if device_world { 3 } else { 2 };
        World::ALL[..count].to_vec()
    }

    /// The suffix of the world's library targets (`_testing`, `_cuda_testing`).
    pub fn lib_suffix(self, config: &Config) -> String {
        match self {
            World::Production => String::new(),
            World::Testing => "_testing".to_owned(),
            World::CudaTesting => format!("_{}_testing", device_feature(config)),
        }
    }

    /// The infix of the world's test targets (`unit_test`, `unit_cuda_test`).
    pub fn test_infix(self, config: &Config) -> String {
        match self {
            World::Production | World::Testing => String::new(),
            World::CudaTesting => format!("_{}", device_feature(config)),
        }
    }
}

fn device_feature(config: &Config) -> &str {
    config
        .device
        .as_ref()
        .map(|d| d.feature.as_str())
        .expect("the device world exists only with a [device]")
}

/// The features and optional dependencies one resolution turned on.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Resolved {
    /// Every package the selected members reach, with its features (possibly none).
    pub features: BTreeMap<String, BTreeSet<String>>,
    /// The optional dependencies turned on, by package.
    pub optional: BTreeMap<String, BTreeSet<String>>,
    /// The features requested of crates of other modules (`[external.crates]`), by crate: the world those crates are
    /// taken from has to have them. A renamed dependency is under its key and under its package name.
    pub external: BTreeMap<String, BTreeSet<String>>,
    /// The features requested of every third-party crate (not a member), by package name, on any active edge: Cargo
    /// unifies them, so the crate links with all of them.
    pub third_party: BTreeMap<String, BTreeSet<String>>,
    roots: BTreeSet<String>,
    dev: bool,
}

impl Resolved {
    pub fn features_of(&self, package: &str) -> Vec<String> {
        self.features
            .get(package)
            .map(|f| f.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Whether any member resolves `feature`.
    pub fn has_any(&self, feature: &str) -> bool {
        self.features.values().any(|f| f.contains(feature))
    }

    pub fn has(&self, package: &str, feature: &str) -> bool {
        self.features
            .get(package)
            .is_some_and(|f| f.contains(feature))
    }

    /// Whether `dep` of `package` is part of the build in this resolution (optional ones only when turned on). A
    /// dev-dependency is on only for a selected member in a world that builds tests.
    pub fn dep_on(&self, package: &str, dep: &Dep) -> bool {
        let kind_on = match dep.kind {
            DepKind::Normal | DepKind::Build => true,
            DepKind::Dev => self.dev && self.roots.contains(package),
        };
        kind_on
            && (!dep.optional
                || self
                    .optional
                    .get(package)
                    .is_some_and(|o| o.contains(dep.key())))
    }
}

struct Resolver<'a> {
    ws: &'a Workspace,
    out: Resolved,
}

impl Resolver<'_> {
    fn package(&self, name: &str) -> Result<&crate::model::Package, String> {
        self.ws
            .package(name)
            .ok_or_else(|| format!("{name} is not a workspace member"))
    }

    fn deps_in_build<'b>(
        &self,
        package: &'b crate::model::Package,
    ) -> impl Iterator<Item = &'b Dep> + use<'b> {
        let dev = self.out.dev && self.out.roots.contains(&package.name);
        package.deps.iter().filter(move |d| {
            (d.kind != DepKind::Dev || dev) && d.applies != Applies::On(Vec::new())
        })
    }

    fn activate(&mut self, name: &str) -> Result<(), String> {
        if self.out.features.contains_key(name) {
            return Ok(());
        }
        self.out.features.insert(name.to_owned(), BTreeSet::new());
        let package = self.package(name)?;
        let deps: Vec<Dep> = self
            .deps_in_build(package)
            .filter(|d| !d.optional)
            .cloned()
            .collect();
        for dep in deps {
            self.edge(&dep)?;
        }
        Ok(())
    }

    /// An active dependency edge: a member gets the edge's features.
    fn edge(&mut self, dep: &Dep) -> Result<(), String> {
        let Some(member) = &dep.member else {
            self.out
                .third_party
                .entry(dep.name.clone())
                .or_default()
                .extend(dep.features.iter().cloned());
            if let Some(key) = self.ws.config.external_key(dep) {
                // Under the entry's key and under the package: `[device] external` names either.
                for name in [key, dep.name.as_str()] {
                    self.out
                        .external
                        .entry(name.to_owned())
                        .or_default()
                        .extend(dep.features.iter().cloned());
                }
            }
            return Ok(());
        };
        self.activate(member)?;
        if dep.default_features {
            self.feature(member, "default", true)?;
        }
        for feature in &dep.features {
            self.feature(member, feature, false)?;
        }
        Ok(())
    }

    fn optional(&mut self, package: &str, dep_name: &str) -> Result<(), String> {
        let first = self
            .out
            .optional
            .entry(package.to_owned())
            .or_default()
            .insert(dep_name.to_owned());
        if !first {
            return Ok(());
        }
        let pkg = self.package(package)?;
        let deps: Vec<Dep> = self
            .deps_in_build(pkg)
            .filter(|d| d.optional && d.key() == dep_name)
            .cloned()
            .collect();
        if deps.is_empty() && !pkg.deps.iter().any(|d| d.optional && d.key() == dep_name) {
            return Err(format!("{package} has no optional dependency {dep_name}"));
        }
        for dep in deps {
            self.edge(&dep)?;
        }
        Ok(())
    }

    /// Turns on `feature` of the active member `package`. `default` may be absent (`if_exists`).
    fn feature(&mut self, package: &str, feature: &str, if_exists: bool) -> Result<(), String> {
        let pkg = self.package(package)?;
        let Some(entries) = pkg.features.get(feature).cloned() else {
            if if_exists && feature == "default" {
                return Ok(());
            }
            return Err(format!("{package} has no feature `{feature}`"));
        };
        let set = self.out.features.entry(package.to_owned()).or_default();
        if !set.insert(feature.to_owned()) {
            return Ok(());
        }
        for entry in entries {
            if let Some(dep) = entry.strip_prefix("dep:") {
                self.optional(package, dep)?;
            } else if entry.contains("?/") {
                return Err(format!(
                    "{package} feature `{feature}` has the weak entry `{entry}`: weak dependency features are not modelled"
                ));
            } else if let Some((dep_name, dep_feature)) = entry.split_once('/') {
                let pkg = self.package(package)?;
                let deps: Vec<Dep> = self
                    .deps_in_build(pkg)
                    .filter(|d| d.key() == dep_name)
                    .cloned()
                    .collect();
                if !pkg.deps.iter().any(|d| d.key() == dep_name) {
                    return Err(format!(
                        "{package} feature `{feature}` names `{entry}`, but {package} has no dependency {dep_name}"
                    ));
                }
                for dep in deps {
                    if dep.member.is_none()
                        && let Some(key) = self.ws.config.external_key(&dep)
                    {
                        if dep.optional {
                            self.optional(package, dep_name)?;
                        }
                        for name in [key, dep.name.as_str()] {
                            self.out
                                .external
                                .entry(name.to_owned())
                                .or_default()
                                .insert(dep_feature.to_owned());
                        }
                        self.out
                            .third_party
                            .entry(dep.name.clone())
                            .or_default()
                            .insert(dep_feature.to_owned());
                        continue;
                    }
                    let Some(member) = &dep.member else {
                        return Err(format!(
                            "{package} feature `{feature}` sets the third-party feature `{entry}`: not modelled \
                             (it needs a crate.annotation in MODULE.bazel and a generator change; a crate of another \
                             Bazel module goes in [external.crates] of bazel_gen.toml)"
                        ));
                    };
                    if dep.optional {
                        self.optional(package, dep_name)?;
                        // As in Cargo, `x/f` also turns on a feature named `x`, the implicit one or a declared one.
                        if self.package(package)?.features.contains_key(dep_name) {
                            self.feature(package, dep_name, false)?;
                        }
                    }
                    self.feature(member, dep_feature, false)?;
                }
            } else {
                self.feature(package, &entry, false)?;
            }
        }
        Ok(())
    }
}

/// Resolves the features of everything `roots` reach, each root with `default` plus its `extras`; `dev` counts the roots'
/// dev-dependencies.
pub fn resolve(
    ws: &Workspace,
    roots: &[&str],
    extras: &dyn Fn(&str) -> Vec<String>,
    dev: bool,
) -> Result<Resolved, String> {
    let mut resolver = Resolver {
        ws,
        out: Resolved {
            roots: roots.iter().map(|r| (*r).to_owned()).collect(),
            dev,
            ..Resolved::default()
        },
    };
    for root in roots {
        resolver.activate(root)?;
    }
    for root in roots {
        resolver.feature(root, "default", true)?;
        for extra in extras(root) {
            resolver.feature(root, &extra, false)?;
        }
    }
    Ok(resolver.out)
}

/// The features of every member in `world`.
pub fn resolve_world(ws: &Workspace, world: World) -> Result<Resolved, String> {
    let device = ws.config.device.as_ref();
    if let Some(member) = device.and_then(|d| d.member.as_deref()) {
        let feature = &device.expect("a member is a device's").feature;
        if !ws
            .package(member)
            .is_some_and(|p| p.features.contains_key(feature.as_str()))
        {
            // The tags and the testing-world refusal look for this feature by name: without it they would pass silently.
            return Err(format!(
                "no member {member} with the feature `{feature}`: the generator finds the device backend by that name \
                 ([device] in bazel_gen.toml); update it with the rename"
            ));
        }
    }
    let members: Vec<&str> = ws.packages.iter().map(|p| p.name.as_str()).collect();
    let extras = |name: &str| -> Vec<String> {
        match device {
            Some(device) if world == World::CudaTesting => device
                .all_features()
                .filter(|f| {
                    ws.package(name)
                        .is_some_and(|p| p.features.contains_key(*f))
                })
                .map(str::to_owned)
                .collect(),
            _ => Vec::new(),
        }
    };
    let resolved = resolve(ws, &members, &extras, world.builds_tests())?;
    if world == World::Testing
        && let Some(device) = device
    {
        // `cargo test --workspace` would link the device build of the device member, or of a crate of another module
        // (taken here as its `_testing` label, which lacks the feature): the device world builds those tests.
        let opened = device
            .member
            .iter()
            .filter(|m| device.all_features().any(|f| resolved.has(m, f)))
            .chain(ws.config.external.keys().filter(|c| {
                device.all_features().any(|f| {
                    resolved
                        .external
                        .get(*c)
                        .is_some_and(|features| features.contains(f))
                })
            }))
            .next();
        if let Some(opened) = opened {
            return Err(format!(
                "the testing world resolves {opened} with a device feature: a dependency would build it with the device \
                 backend, but the testing world links its build without. Remove that feature request; the {} testing \
                 world builds those tests",
                device.feature
            ));
        }
    }
    // A declared device feature that nothing resolves is a stale declaration: the tags would never fire. The last world
    // built is the one that has it (the device world, else the testing world).
    if world == *World::built(&ws.config).last().expect("a world")
        && let Some(device) = device
        && let Some(dead) = device.all_features().find(|f| {
            !(resolved.has_any(f)
                || resolved.third_party.values().any(|t| t.contains(*f))
                || ws
                    .locked
                    .as_ref()
                    .is_some_and(|l| l.features.values().any(|t| t.contains(*f))))
        })
    {
        return Err(format!(
            "[device] declares the feature `{dead}`, but no crate of the workspace or of its dependencies is resolved with \
             it in the {} world: the declaration is stale (remove it, or the dependency's feature was dropped)",
            if world == World::CudaTesting {
                "device"
            } else {
                "testing"
            },
        ));
    }
    if world == World::Production {
        // One consistent world: every member a member reaches when built alone must have the features it has with all of
        // them built, since both link the same target. An empty `default` turns on nothing, so it may differ.
        let effective = |package: &str, resolved: &Resolved| -> Vec<String> {
            let empty_default = ws
                .package(package)
                .and_then(|p| p.features.get("default"))
                .is_some_and(Vec::is_empty);
            resolved
                .features_of(package)
                .into_iter()
                .filter(|f| !(empty_default && f == "default"))
                .collect()
        };
        for member in &members {
            let alone = resolve(ws, &[member], &|_| Vec::new(), false)?;
            for reached in alone.features.keys() {
                if effective(reached, &alone) != effective(reached, &resolved) {
                    return Err(format!(
                        "{reached}'s production features differ with every member built ({:?}) and with {member} \
                         built alone ({:?}): a member's normal dependency turns on a feature of another; not modelled",
                        resolved.features_of(reached),
                        alone.features_of(reached),
                    ));
                }
            }
        }
    }
    Ok(resolved)
}
