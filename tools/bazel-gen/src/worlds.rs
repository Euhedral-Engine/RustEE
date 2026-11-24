//! The feature resolver for workspace crates, and the three worlds (docs/design/bazel.md, "Features").
//!
//! A world is one consistent choice of features for every member. Resolution follows Cargo's resolver for first-party
//! crates: a selected member gets its `default` feature (and the world's extras), every active dependency edge requests
//! its `features` (and `default` unless `default-features = false`), `dep:x` turns on an optional dependency, `x/f` turns
//! on `x` and its feature `f`, and dev-dependencies count only for the selected members and only when the world builds
//! tests. Third-party crates are `crate_universe`'s: the resolver only records which optional ones are on.

use std::collections::{BTreeMap, BTreeSet};

use crate::cfg::Applies;
use crate::model::{Dep, DepKind, Workspace};

/// The feature that loads the CUDA driver: a test whose graph has it opens the card.
pub const DEVICE_FEATURE: (&str, &str) = ("euhedral-memory", "cuda");

/// The feature the cuda testing world adds to every member that declares it.
pub const CUDA: &str = "cuda";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum World {
    /// Each member's defaults, no dev-dependencies: what a consumer of the crate gets.
    Production = 0,
    /// `cargo test --workspace`: defaults plus every member's dev-dependency features.
    Testing = 1,
    /// Testing plus `cuda` on every member that has it.
    CudaTesting = 2,
}

impl World {
    pub const ALL: [World; 3] = [World::Production, World::Testing, World::CudaTesting];

    pub fn builds_tests(self) -> bool {
        self != World::Production
    }

    /// The suffix of the world's library targets.
    pub fn lib_suffix(self) -> &'static str {
        match self {
            World::Production => "",
            World::Testing => "_testing",
            World::CudaTesting => "_cuda_testing",
        }
    }

    /// The infix of the world's test targets (`unit_test`, `unit_cuda_test`).
    pub fn test_infix(self) -> &'static str {
        match self {
            World::Production | World::Testing => "",
            World::CudaTesting => "_cuda",
        }
    }
}

/// The features and optional dependencies one resolution turned on.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Resolved {
    /// Every package the selected members reach, with its features (possibly none).
    pub features: BTreeMap<String, BTreeSet<String>>,
    /// The optional dependencies turned on, by package.
    pub optional: BTreeMap<String, BTreeSet<String>>,
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

    pub fn has(&self, package: &str, feature: &str) -> bool {
        self.features
            .get(package)
            .is_some_and(|f| f.contains(feature))
    }

    /// Whether `dep` of `package` is part of the build in this resolution (optional ones only when turned on). A
    /// dev-dependency is on only for a selected member in a world that builds tests.
    pub fn dep_on(&self, package: &str, dep: &Dep) -> bool {
        let kind_on = match dep.kind {
            DepKind::Normal => true,
            DepKind::Dev => self.dev && self.roots.contains(package),
        };
        kind_on
            && (!dep.optional
                || self
                    .optional
                    .get(package)
                    .is_some_and(|o| o.contains(&dep.name)))
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
            (d.kind == DepKind::Normal || dev) && d.applies != Applies::On(Vec::new())
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
            .filter(|d| d.optional && d.name == dep_name)
            .cloned()
            .collect();
        if deps.is_empty() && !pkg.deps.iter().any(|d| d.optional && d.name == dep_name) {
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
                    .filter(|d| d.name == dep_name)
                    .cloned()
                    .collect();
                if !pkg.deps.iter().any(|d| d.name == dep_name) {
                    return Err(format!(
                        "{package} feature `{feature}` names `{entry}`, but {package} has no dependency {dep_name}"
                    ));
                }
                for dep in deps {
                    let Some(member) = &dep.member else {
                        return Err(format!(
                            "{package} feature `{feature}` sets the third-party feature `{entry}`: not modelled \
                             (it needs a crate.annotation in MODULE.bazel and a generator change)"
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
    let (device_crate, device_feature) = DEVICE_FEATURE;
    if !ws
        .package(device_crate)
        .is_some_and(|p| p.features.contains_key(device_feature))
    {
        // The tags and the testing-world refusal look for this feature by name: without it they would pass silently.
        return Err(format!(
            "no member {device_crate} with the feature `{device_feature}`: the generator finds the device backend by \
             that name (DEVICE_FEATURE in tools/bazel-gen/src/worlds.rs); update it with the rename"
        ));
    }
    let members: Vec<&str> = ws.packages.iter().map(|p| p.name.as_str()).collect();
    let cuda = |name: &str| -> Vec<String> {
        let has = ws
            .package(name)
            .is_some_and(|p| p.features.contains_key(CUDA));
        if world == World::CudaTesting && has {
            vec![CUDA.to_owned()]
        } else {
            Vec::new()
        }
    };
    let resolved = resolve(ws, &members, &cuda, world.builds_tests())?;
    if world == World::Testing && resolved.has(device_crate, device_feature) {
        return Err(format!(
            "the testing world resolves {device_crate}/{device_feature}: a member's dependency turns on the device \
             backend, so its tests would open the card untagged. Remove that feature request; the cuda testing world \
             builds those tests"
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
