//! The generator end to end on a small workspace: feature-world resolution, the `gpu`/`external` tags, the refusals and
//! Windows paths.

use std::collections::BTreeMap;

use crate::config::{Config, Device};
use crate::emit::generate;
use crate::json;
use crate::model::{Workspace, load};
use crate::worlds::{World, resolve_world};

/// One dependency of the fixture: (name, dev, path member, features); the feature `!default` is `default-features = false`
/// and `?optional` is `optional = true`.
type FixtureDep<'a> = (&'a str, bool, Option<&'a str>, &'a [&'a str]);

struct Fixture<'a> {
    name: &'a str,
    features: &'a [(&'a str, &'a [&'a str])],
    deps: Vec<FixtureDep<'a>>,
    tests: &'a [&'a str],
}

/// A workspace of:
/// - `example-memory`, with `cuda`;
/// - `user`, whose `cuda` turns on `example-memory/cuda`, with the test `t`;
/// - `plain`, which has a `testing` feature its own dev-dependency on itself turns on, with the test `p`, linking neither.
fn members<'a>() -> Vec<Fixture<'a>> {
    vec![
        Fixture {
            name: "example-memory",
            features: &[("cuda", &[]), ("default", &[])],
            deps: vec![],
            tests: &[],
        },
        Fixture {
            name: "user",
            features: &[("cuda", &["example-memory/cuda"])],
            deps: vec![("example-memory", false, Some("example-memory"), &[])],
            tests: &["t"],
        },
        Fixture {
            name: "plain",
            features: &[("default", &["x"]), ("testing", &[]), ("x", &[])],
            deps: vec![("plain", true, Some("plain"), &["testing"])],
            tests: &["p"],
        },
    ]
}

/// `cargo metadata --no-deps` for `members` under `root`, with `sep` as the path separator.
fn metadata(members: &[Fixture], root: &str, sep: &str) -> String {
    let path = |parts: &[&str]| json_str(&parts.join(sep));
    let list = |items: &[&str]| {
        format!(
            "[{}]",
            items
                .iter()
                .map(|i| json_str(i))
                .collect::<Vec<_>>()
                .join(",")
        )
    };
    let mut packages = Vec::new();
    for m in members {
        let features: Vec<String> = m
            .features
            .iter()
            .map(|(f, e)| format!("{}:{}", json_str(f), list(e)))
            .collect();
        let deps: Vec<String> = m
            .deps
            .iter()
            .map(|(name, dev, member, features)| {
                let defaults = !features.contains(&"!default");
                let optional = features.contains(&"?optional");
                let features: Vec<&str> = features
                    .iter()
                    .copied()
                    .filter(|f| *f != "!default" && *f != "?optional")
                    .collect();
                format!(
                    r#"{{"name":{},"kind":{},"rename":null,"target":null,"optional":{optional},"uses_default_features":{defaults},"features":{}{}}}"#,
                    json_str(name),
                    if *dev { "\"dev\"" } else { "null" },
                    list(&features),
                    member.map(|d| format!(",\"path\":{}", path(&[root, d]))).unwrap_or_default()
                )
            })
            .collect();
        let target = |kind: &str, name: &str, src: &[&str]| {
            let crate_type = if kind == "lib" { "lib" } else { "bin" };
            let mut src_path = vec![root, m.name];
            src_path.extend(src);
            format!(
                r#"{{"name":{},"kind":["{kind}"],"crate_types":["{crate_type}"],"src_path":{},"edition":"2024","test":true,"doctest":{}}}"#,
                json_str(name),
                path(&src_path),
                kind == "lib"
            )
        };
        let mut targets = vec![target("lib", &m.name.replace('-', "_"), &["src", "lib.rs"])];
        for t in m.tests {
            targets.push(target("test", t, &["tests", &format!("{t}.rs")]));
        }
        packages.push(format!(
            r#"{{"name":{},"version":"0.1.0","manifest_path":{},"features":{{{}}},"dependencies":[{}],"targets":[{}],"metadata":null}}"#,
            json_str(m.name),
            path(&[root, m.name, "Cargo.toml"]),
            features.join(","),
            deps.join(","),
            targets.join(",")
        ));
    }
    format!(
        r#"{{"workspace_root":{},"packages":[{}]}}"#,
        json_str(root),
        packages.join(",")
    )
}

/// The producer's device: `example-memory/cuda` opens the card.
fn config() -> Config {
    Config {
        device: Some(Device {
            member: Some("example-memory".to_owned()),
            feature: "cuda".to_owned(),
            external: Vec::new(),
            also: Vec::new(),
        }),
        ..Config::default()
    }
}

fn json_str(text: &str) -> String {
    format!("\"{}\"", text.replace('\\', "\\\\"))
}

fn workspace(members: &[Fixture], root: &str, sep: &str) -> Result<Workspace, String> {
    let metadata = json::parse(&metadata(members, root, sep))?;
    load(&metadata, &|_| Ok("[package]\n".to_owned()), config())
}

fn files(members: &[Fixture]) -> BTreeMap<String, String> {
    generate(&workspace(members, "/w", "/").expect("the fixture loads"))
        .expect("the fixture generates")
}

/// The rule named `name` in a BUILD file.
fn rule<'a>(file: &'a str, name: &str) -> Option<&'a str> {
    let at = file.find(&format!("    name = \"{name}\",\n"))?;
    let start = file[..at - 1].rfind('\n').map_or(0, |i| i + 1);
    let end = at + file[at..].find("\n)\n")? + 3;
    Some(&file[start..end])
}

#[test]
fn each_world_resolves_its_features() {
    let ws = workspace(&members(), "/w", "/").expect("the fixture loads");
    let features = |world, package: &str| {
        resolve_world(&ws, world)
            .expect("resolves")
            .features_of(package)
    };
    assert_eq!(features(World::Production, "plain"), ["default", "x"]);
    assert_eq!(
        features(World::Testing, "plain"),
        ["default", "testing", "x"]
    );
    assert_eq!(features(World::Testing, "example-memory"), ["default"]);
    assert_eq!(features(World::CudaTesting, "user"), ["cuda"]);
    assert_eq!(
        features(World::CudaTesting, "example-memory"),
        ["cuda", "default"]
    );
}

#[test]
fn only_tests_that_link_the_device_feature_are_tagged() {
    let files = files(&members());
    let user = &files["user/BUILD.bazel"];
    let cuda = rule(user, "t_cuda_test").expect("the cuda world builds user's test");
    assert!(cuda.contains("\"external\",\n        \"gpu\",\n"), "{cuda}");
    let plain_test = rule(user, "t_test").expect("the testing world builds user's test");
    assert!(!plain_test.contains("gpu"), "{plain_test}");
    let unit = rule(user, "unit_cuda_test").expect("and its unit tests");
    assert!(unit.contains("\"gpu\""), "{unit}");
    // plain reaches no device feature: the cuda world has nothing new to build for it.
    let plain = &files["plain/BUILD.bazel"];
    assert!(rule(plain, "p_test").is_some());
    assert!(rule(plain, "p_cuda_test").is_none(), "{plain}");
    assert!(rule(plain, "plain_cuda_testing").is_some_and(|r| r.starts_with("alias(")));
    assert!(!plain.contains("\"gpu\""), "{plain}");
}

#[test]
fn a_testing_world_that_builds_the_device_member_with_its_feature_is_refused() {
    let mut members = members();
    // A dev-dependency asks for example-memory/cuda: `cargo test --workspace` would open the card.
    members[2]
        .deps
        .push(("example-memory", true, Some("example-memory"), &["cuda"]));
    let ws = workspace(&members, "/w", "/").expect("the fixture loads");
    let error = generate(&ws).expect_err("the generator refuses");
    assert!(
        error.contains("testing world resolves example-memory"),
        "{error}"
    );
}

#[test]
fn a_production_feature_one_member_turns_on_for_another_is_refused() {
    // user's normal dependency turns on plain's `testing`: plain's one production target cannot serve both.
    let mut on = members();
    on[1]
        .deps
        .push(("plain", false, Some("plain"), &["testing"]));
    // user takes plain without its defaults, which turn on `x`: user alone would link a plain without `x`.
    let mut off = members();
    off[1]
        .deps
        .push(("plain", false, Some("plain"), &["!default"]));
    for members in [on, off] {
        let ws = workspace(&members, "/w", "/").expect("the fixture loads");
        let error = generate(&ws).expect_err("the generator refuses");
        assert!(
            error.contains("plain's production features differ"),
            "{error}"
        );
    }
    // An empty `default` turns on nothing: example-memory without it is the same library.
    let mut empty = members();
    empty[2].deps.push((
        "example-memory",
        false,
        Some("example-memory"),
        &["!default"],
    ));
    generate(&workspace(&empty, "/w", "/").expect("the fixture loads"))
        .expect("an empty default may differ");
}

#[test]
fn unmodelled_manifests_are_refused() {
    let mut weak = members();
    weak[1].features = &[("cuda", &["example-memory?/cuda"])];
    let ws = workspace(&weak, "/w", "/").expect("the fixture loads");
    assert!(generate(&ws).expect_err("weak features").contains("weak"));

    let mut stray = members();
    stray[1]
        .deps
        .push(("elsewhere", false, Some("../elsewhere"), &[]));
    assert!(
        workspace(&stray, "/w", "/")
            .expect_err("outside path")
            .contains("outside")
    );
}

#[test]
fn windows_paths_generate_the_same_files() {
    let windows = workspace(&members(), r"C:\w", r"\").expect("the Windows fixture loads");
    assert_eq!(generate(&windows).expect("generates"), files(&members()));
}

#[test]
fn a_dependency_feature_turns_on_the_feature_named_after_an_optional_dependency() {
    // user's `extra` asks for plain/testing through an optional plain, and user declares a feature `plain` for it: Cargo
    // turns that feature on too, so the code under cfg(feature = "plain") is built.
    let mut members = members();
    members[1].features = &[
        ("cuda", &["example-memory/cuda"]),
        ("default", &["extra"]),
        ("extra", &["plain/testing"]),
        ("plain", &["dep:plain"]),
    ];
    members[1]
        .deps
        .push(("plain", false, Some("plain"), &["?optional"]));
    let ws = workspace(&members, "/w", "/").expect("the fixture loads");
    let resolved = resolve_world(&ws, World::Testing).expect("resolves");
    assert_eq!(resolved.features_of("user"), ["default", "extra", "plain"]);
    assert!(resolved.has("plain", "testing"));
}

#[test]
fn a_workspace_without_the_device_feature_is_refused() {
    // Renamed away, the device feature would leave every cuda test untagged and the testing world unchecked.
    let mut members = members();
    members[0].features = &[("gpu", &[]), ("default", &[])];
    members[1].features = &[("cuda", &["example-memory/gpu"])];
    let ws = workspace(&members, "/w", "/").expect("the fixture loads");
    let error = generate(&ws).expect_err("the generator refuses");
    assert!(
        error.contains("example-memory with the feature `cuda`"),
        "{error}"
    );
}

#[test]
fn a_library_named_other_than_its_package_is_refused() {
    let metadata = metadata(&members(), "/w", "/").replace(
        r#"{"name":"user","kind":["lib"]"#,
        r#"{"name":"other","kind":["lib"]"#,
    );
    let error = load(
        &json::parse(&metadata).expect("json"),
        &|_| Ok("[package]\n".to_owned()),
        config(),
    )
    .expect_err("a renamed library");
    assert!(error.contains("names its library `other`"), "{error}");
}

#[test]
fn a_dev_dependency_wider_than_a_target_specific_normal_one_reaches_the_unit_tests() {
    // plain needs libc on Unix only, and its tests on every OS: the unit tests take it on Windows.
    let mut members = members();
    members[2].deps.push(("libc", false, None, &[]));
    members[2].deps.push(("libc", true, None, &[]));
    let metadata = metadata(&members, "/w", "/").replacen(
        r#""name":"libc","kind":null,"rename":null,"target":null"#,
        r#""name":"libc","kind":null,"rename":null,"target":"cfg(unix)""#,
        1,
    );
    let ws = load(
        &json::parse(&metadata).expect("json"),
        &|path| {
            Ok(if path == "l.json" {
                "{\"crates\":{}}"
            } else {
                "[package]\n"
            }
            .to_owned())
        },
        Config {
            crate_lock: Some("l.json".to_owned()),
            ..config()
        },
    )
    .expect("the fixture loads");
    let files = generate(&ws).expect("generates");
    let unit = rule(&files["plain/BUILD.bazel"], "unit_test").expect("plain's unit tests");
    assert!(unit.contains("windows"), "{unit}");
    assert!(unit.contains("@crates//:libc"), "{unit}");
}

/// `app`, whose `cuda` turns on the git dependency `ext`'s `cuda`, with the test `t`.
fn git_dependency_user<'a>() -> Vec<Fixture<'a>> {
    vec![Fixture {
        name: "app",
        features: &[("cuda", &["ext/cuda"])],
        deps: vec![("ext", false, None, &["!default"])],
        tests: &["t"],
    }]
}

const GIT_DEPENDENCY_CONFIG: &str = "[device]\nfeature = \"cuda\"\nexternal = [\"ext\"]\n\n\
    [external.crates]\next = \"@m//ext\"\n\n[packages.app]\ncompile_data = [\"kernels/**\"]\n\
    test_data = [\"//:testdata\"]\n\n[root]\nfilegroups = { testdata = [\"testdata/**\"] }\n";

fn generate_with(
    members: &[Fixture],
    config: &str,
    manifest: &dyn Fn(&str) -> Result<String, String>,
) -> BTreeMap<String, String> {
    let metadata = json::parse(&metadata(members, "/w", "/")).expect("json");
    let config = Config::parse(config).expect("the configuration parses");
    generate(&load(&metadata, manifest, config).expect("the fixture loads")).expect("generates")
}

#[test]
fn a_git_dependency_takes_the_label_of_each_world_and_its_device_feature_opens_the_card() {
    let files = generate_with(&git_dependency_user(), GIT_DEPENDENCY_CONFIG, &|_| {
        Ok("[package]\n".to_owned())
    });
    let app = &files["app/BUILD.bazel"];
    for (lib, label) in [
        ("app", "\"@m//ext:ext\""),
        ("app_testing", "\"@m//ext:ext_testing\""),
        ("app_cuda_testing", "\"@m//ext:ext_cuda_testing\""),
    ] {
        let rule = rule(app, lib).expect("a library per world");
        assert!(rule.contains(&format!("deps = [{label}]")), "{rule}");
    }
    assert!(!app.contains("@crates//:ext"), "{app}");
    let test = rule(app, "t_test").expect("the testing world's test");
    assert!(!test.contains("gpu"), "{test}");
    let cuda = rule(app, "t_cuda_test").expect("the device world's test");
    assert!(cuda.contains("\"@m//ext:ext_cuda_testing\""), "{cuda}");
    assert!(cuda.contains("\"external\",\n        \"gpu\",\n"), "{cuda}");
}

#[test]
fn package_data_and_root_filegroups_come_from_the_configuration() {
    let files = generate_with(&git_dependency_user(), GIT_DEPENDENCY_CONFIG, &|_| {
        Ok("[package]\n".to_owned())
    });
    let root = &files["BUILD.bazel"];
    let testdata = rule(root, "testdata").expect("the configured filegroup");
    assert!(testdata.contains("glob([\"testdata/**\"])"), "{testdata}");
    let app = &files["app/BUILD.bazel"];
    assert!(
        rule(app, "app").is_some_and(|r| r.contains("\"kernels/**\"")),
        "{app}"
    );
    let test = rule(app, "t_test").expect("the test");
    // A test that reads the data gets it at run time and at compile time (include_bytes!), from CARGO_MANIFEST_DIR.
    assert!(
        test.contains("compile_data = [\"//:testdata\"] + glob("),
        "{test}"
    );
    assert!(test.contains("data = [\"//:testdata\"]"), "{test}");
    assert!(test.contains("\"CARGO_MANIFEST_DIR\": \"app\""), "{test}");
}

#[test]
fn a_member_inherits_the_workspace_lints() {
    let manifest = |path: &str| -> Result<String, String> {
        Ok(match path {
            "Cargo.toml" => {
                "[workspace]\n\n[workspace.lints.rust]\nunsafe_code = \"forbid\"\n".to_owned()
            }
            _ => "[package]\nname = \"app\"\n\n[lints]\nworkspace = true\n".to_owned(),
        })
    };
    let files = generate_with(&git_dependency_user(), GIT_DEPENDENCY_CONFIG, &manifest);
    let lints = rule(&files["app/BUILD.bazel"], "lints").expect("the lint config");
    assert!(lints.contains("\"unsafe_code\": \"forbid\""), "{lints}");
    let mixed = |path: &str| -> Result<String, String> {
        Ok(match path {
            "Cargo.toml" => "[workspace]\n".to_owned(),
            _ => {
                "[package]\n\n[lints]\nworkspace = true\n\n[lints.rust]\nunsafe_code = \"forbid\"\n"
                    .to_owned()
            }
        })
    };
    let metadata = json::parse(&metadata(&git_dependency_user(), "/w", "/")).expect("json");
    let config = Config::parse(GIT_DEPENDENCY_CONFIG).expect("parses");
    let error = load(&metadata, &mixed, config).expect_err("inherited and own lints at once");
    assert!(error.contains("one or the other"), "{error}");
}

#[test]
fn each_package_metadata_key_reaches_its_target() {
    let bin = r#"{"name":"tool","kind":["bin"],"crate_types":["bin"],"src_path":"/w/app/src/main.rs","edition":"2024","test":false,"doctest":false}"#;
    let bazel = r#"{"bazel":{"test":{"t":{"cpus":4,"workspace_sources":true,"cargo":true}},"bin":{"tool":{"root_alias":"tool"}}}}"#;
    let metadata = metadata(&git_dependency_user(), "/w", "/")
        .replacen("\"targets\":[", &format!("\"targets\":[{bin},"), 1)
        .replacen("\"metadata\":null", &format!("\"metadata\":{bazel}"), 1);
    let config = Config::parse(GIT_DEPENDENCY_CONFIG).expect("the configuration parses");
    let ws = load(
        &json::parse(&metadata).expect("json"),
        &|_| Ok("[package]\n".to_owned()),
        config,
    )
    .expect("the fixture loads");
    let files = generate(&ws).expect("generates");
    let app = &files["app/BUILD.bazel"];
    let test = rule(app, "t_test").expect("the test");
    assert!(test.contains("\"cpu:4\""), "{test}");
    assert!(test.contains("\"//:workspace_sources\""), "{test}");
    assert!(test.contains("\"CARGO_MANIFEST_DIR\": \"app\""), "{test}");
    assert!(
        test.contains("\"@rules_rust//cargo:current_cargo_toolchain\""),
        "{test}"
    );
    assert!(
        test.contains("\"CARGO\": \"$(rootpath @rules_rust//cargo:current_cargo_toolchain)\""),
        "{test}"
    );
    // The keys belong to the test they name: the device world's copy of it gets them, other targets do not.
    assert!(
        rule(app, "t_cuda_test").is_some_and(|r| r.contains("\"cpu:4\"")),
        "{app}"
    );
    assert!(
        rule(app, "unit_test").is_some_and(|r| !r.contains("cpu:") && !r.contains("CARGO\"")),
        "{app}"
    );
    let alias = rule(&files["BUILD.bazel"], "tool").expect("the root alias");
    assert!(alias.contains("actual = \"//app:tool\""), "{alias}");
}

#[test]
fn in_the_device_world_any_mapped_external_crate_opens_the_card() {
    // `ext` is mapped but not in `device.external`: its device build may still open the card.
    let config = GIT_DEPENDENCY_CONFIG.replace("external = [\"ext\"]\n", "");
    let files = generate_with(&git_dependency_user(), &config, &|_| {
        Ok("[package]\n".to_owned())
    });
    let app = &files["app/BUILD.bazel"];
    let test = rule(app, "t_test").expect("the testing world's test");
    assert!(!test.contains("gpu"), "{test}");
    let cuda = rule(app, "t_cuda_test").expect("the device world's test");
    assert!(cuda.contains("\"external\",\n        \"gpu\",\n"), "{cuda}");
}

/// A consumer whose workspace dependencies turn `cuda` on for the third-party crate `gpu` (no member declares it):
/// - `lib` depends on `gpu` with `cuda`, with the test `l`;
/// - `app` depends only on `lib`, with the test `a`;
/// - `quiet` depends on `lib` and never opens the card, with the test `q`;
/// - `plain` reaches no `gpu`, with the test `p`.
fn third_party_workspace<'a>() -> Vec<Fixture<'a>> {
    vec![
        Fixture {
            name: "lib",
            features: &[],
            deps: vec![("gpu", false, None, &["cuda"])],
            tests: &["l"],
        },
        Fixture {
            name: "app",
            features: &[],
            deps: vec![("lib", false, Some("lib"), &[])],
            tests: &["a"],
        },
        Fixture {
            name: "quiet",
            features: &[],
            deps: vec![("lib", false, Some("lib"), &[])],
            tests: &["q"],
        },
        Fixture {
            name: "plain",
            features: &[],
            deps: vec![],
            tests: &["p"],
        },
    ]
}

const DEVICE_ONLY: &str = "[device]\nfeature = \"cuda\"\n[root]\ncrate_lock = \"l.json\"\n";

fn try_generate(members: &[Fixture], config: &str) -> Result<BTreeMap<String, String>, String> {
    let metadata = json::parse(&metadata(members, "/w", "/")).expect("json");
    let config = Config::parse(config).expect("the configuration parses");
    let read = |path: &str| {
        Ok(if path == "l.json" {
            "{\"crates\":{}}"
        } else {
            "[package]\n"
        }
        .to_owned())
    };
    generate(&load(&metadata, &read, config)?)
}

fn tagged(files: &BTreeMap<String, String>, package: &str, test: &str) -> bool {
    let text = &files[&format!("{package}/BUILD.bazel")];
    rule(text, test)
        .expect("a test")
        .contains("\"external\",\n        \"gpu\",\n")
}

#[test]
fn tests_linking_a_third_party_crate_with_the_device_feature_are_tagged_without_naming_it() {
    let files = try_generate(&third_party_workspace(), DEVICE_ONLY).expect("generates");
    for (package, test) in [
        ("lib", "l_test"),
        ("lib", "unit_test"),
        ("lib", "doc_test"),
        ("app", "a_test"),
        ("app", "unit_test"),
        ("quiet", "q_test"),
    ] {
        assert!(tagged(&files, package, test), "{package}/{test}");
    }
    for test in ["p_test", "unit_test", "doc_test"] {
        assert!(!tagged(&files, "plain", test), "plain/{test}");
    }
    // The feature is on in every world already: there is no device world.
    assert!(!files["lib/BUILD.bazel"].contains("cuda"));
}

#[test]
fn card_tests_overrides_the_derived_tag_in_both_directions() {
    let config = format!(
        "{DEVICE_ONLY}[packages.quiet]\ncard_tests = false\n[packages.plain]\ncard_tests = true\n"
    );
    let files = try_generate(&third_party_workspace(), &config).expect("generates");
    assert!(!tagged(&files, "quiet", "q_test"));
    assert!(!tagged(&files, "quiet", "unit_test"));
    assert!(tagged(&files, "plain", "p_test"));
    assert!(tagged(&files, "app", "a_test"));
}

#[test]
fn a_package_marked_card_tests_without_any_device_feature_tags_its_dependents_too() {
    let files = try_generate(
        &third_party_workspace(),
        "[packages.lib]\ncard_tests = true\n",
    )
    .expect("generates");
    assert!(tagged(&files, "lib", "l_test"));
    assert!(tagged(&files, "app", "a_test"));
    assert!(!tagged(&files, "plain", "p_test"));
}

#[test]
fn a_second_device_feature_opens_the_card_too_and_each_declared_feature_must_be_resolved() {
    let mut members = third_party_workspace();
    members[0].deps = vec![("gpu", false, None, &["vulkan"])];
    members[3].deps = vec![("other", false, None, &["cuda"])];
    let config =
        "[device]\nfeature = \"cuda\"\nalso = [\"vulkan\"]\n[root]\ncrate_lock = \"l.json\"\n";
    let files = try_generate(&members, config).expect("generates");
    assert!(tagged(&files, "app", "a_test"));
    let error = try_generate(
        &members,
        &config.replace("[\"vulkan\"]", "[\"vulkan\", \"dead\"]"),
    )
    .expect_err("a dead `also` entry");
    assert!(error.contains("`dead`"), "{error}");
}

#[test]
fn a_device_feature_no_crate_resolves_is_refused() {
    let mut members = third_party_workspace();
    members[0].deps = vec![("gpu", false, None, &[])];
    let error = try_generate(&members, DEVICE_ONLY).expect_err("a stale declaration");
    assert!(error.contains("[device] declares"), "{error}");
}

#[test]
fn a_member_other_than_the_device_member_that_resolves_a_device_feature_opens_the_card() {
    // `engine` has `cuda`; `app` asks for it in a dev-dependency, and `eng2` turns it on by default.
    let members = vec![
        Fixture {
            name: "engine",
            features: &[("cuda", &[])],
            deps: vec![],
            tests: &["e"],
        },
        Fixture {
            name: "app",
            features: &[],
            deps: vec![("engine", true, Some("engine"), &["cuda"])],
            tests: &["a"],
        },
        Fixture {
            name: "eng2",
            features: &[("default", &["cuda"]), ("cuda", &[])],
            deps: vec![],
            tests: &["d"],
        },
        Fixture {
            name: "plain",
            features: &[],
            deps: vec![],
            tests: &["p"],
        },
    ];
    let files = try_generate(&members, DEVICE_ONLY).expect("generates");
    assert!(tagged(&files, "engine", "e_test"));
    assert!(tagged(&files, "app", "a_test"));
    assert!(tagged(&files, "eng2", "d_test"));
    assert!(!tagged(&files, "plain", "p_test"));
}

fn locked_workspace<'a>() -> Vec<Fixture<'a>> {
    // via_chain -> a -> b (cuda); via_default -> c (cuda: the lock's resolved features hold what a default or another
    // feature turned on); clean -> d.
    let on = |name: &'static str, krate: &'static str| Fixture {
        name,
        features: &[],
        deps: vec![(krate, false, None, &[])],
        tests: &["t"],
    };
    vec![
        on("via_chain", "a"),
        on("via_default", "c"),
        on("clean", "d"),
    ]
}

const LOCK: &str = r#"{"crates":{
  "a 1.0.0":{"name":"a","common_attrs":{"crate_features":{"common":["std"],"selects":{}},
     "deps":{"common":[],"selects":{"x86_64-unknown-linux-gnu":[{"id":"b 1.0.0","target":"b"}]}}}},
  "b 1.0.0":{"name":"b","common_attrs":{"crate_features":{"common":[],"selects":{"x86_64-unknown-linux-gnu":["cuda"]}}}},
  "c 2.0.0":{"name":"c","common_attrs":{"crate_features":{"common":["default","gpu-accel","cuda"],"selects":{}}}},
  "d 1.0.0":{"name":"d","common_attrs":{"crate_features":{"common":["default"],"selects":{}},"deps":{"common":[{"id":"a 1.0.0"}]}}}
}}"#;

#[test]
fn the_crate_lock_gives_the_features_and_links_third_party_crates_resolve_to() {
    let config = DEVICE_ONLY.to_owned();
    let metadata = json::parse(&metadata(&locked_workspace(), "/w", "/")).expect("json");
    let load_with = |config: &str, lock: &str| {
        let lock = lock.to_owned();
        let read = move |path: &str| {
            if path == "l.json" {
                Ok(lock.clone())
            } else {
                Ok("[package]\n".to_owned())
            }
        };
        load(&metadata, &read, Config::parse(config).expect("config"))
    };
    let files = generate(&load_with(&config, LOCK).expect("loads")).expect("generates");
    assert!(tagged(&files, "via_chain", "t_test"), "a transitive chain");
    assert!(
        tagged(&files, "via_default", "t_test"),
        "a resolved feature"
    );
    // `d` links `a`, which links `b`: the chain reaches it too.
    assert!(tagged(&files, "clean", "t_test"));
    let no_chain = LOCK.replace("\"id\":\"b 1.0.0\"", "\"id\":\"z 1.0.0\"");
    let files = generate(&load_with(&config, &no_chain).expect("loads")).expect("generates");
    assert!(!tagged(&files, "via_chain", "t_test"));
    assert!(tagged(&files, "via_default", "t_test"));
    assert!(!tagged(&files, "clean", "t_test"));
    // The lock is part of the input: missing, it is an error naming it, not a silent loss of tags.
    let missing = |path: &str| -> Result<String, String> {
        if path == "l.json" {
            Err("reading l.json: gone".to_owned())
        } else {
            Ok("[package]\n".to_owned())
        }
    };
    let error = load(&metadata, &missing, Config::parse(&config).expect("config"))
        .expect_err("a missing crate lock");
    assert!(error.contains("l.json"), "{error}");
}

#[test]
fn card_tests_on_a_non_member_or_that_changes_nothing_is_refused() {
    let error = try_generate(
        &third_party_workspace(),
        &format!("{DEVICE_ONLY}[packages.ghost]\ncard_tests = true\n"),
    )
    .expect_err("not a member");
    assert!(
        error.contains("ghost") && error.contains("not a workspace member"),
        "{error}"
    );
    let error = try_generate(
        &third_party_workspace(),
        &format!("{DEVICE_ONLY}[packages.plain]\ncard_tests = false\n"),
    )
    .expect_err("hides nothing");
    assert!(
        error.contains("plain") && error.contains("changes nothing"),
        "{error}"
    );
}

#[test]
fn card_tests_false_does_not_untag_what_the_device_member_tags() {
    // The producer's shape: `user`'s device-world tests link the device member and stay tagged; a `false` on it is refused as
    // changing nothing in the world where nothing else tags it.
    let config = "[device]\nmember = \"example-memory\"\nfeature = \"cuda\"\n[packages.user]\ncard_tests = false\n";
    let ws = workspace(&members(), "/w", "/").expect("loads");
    let mut ws = ws;
    ws.config = Config::parse(config).expect("config");
    let error = generate(&ws).expect_err("hides nothing");
    assert!(error.contains("changes nothing"), "{error}");
}

#[test]
fn a_device_workspace_with_third_party_dependencies_needs_the_crate_lock() {
    let metadata = json::parse(&metadata(&third_party_workspace(), "/w", "/")).expect("json");
    let read = |_: &str| Ok("[package]\n".to_owned());
    let config = Config::parse("[device]\nfeature = \"cuda\"\n").expect("config");
    let error = load(&metadata, &read, config).expect_err("no crate lock");
    assert!(
        error.contains("lib")
            && error.contains("gpu")
            && error.contains("lockfile")
            && error.contains("crate_lock"),
        "{error}"
    );
    // Without third-party dependencies there is nothing to resolve: no lock needed.
    let own = vec![Fixture {
        name: "solo",
        features: &[("cuda", &[])],
        deps: vec![],
        tests: &["t"],
    }];
    let metadata = json::parse(&self::metadata(&own, "/w", "/")).expect("json");
    let config = Config::parse("[device]\nfeature = \"cuda\"\n").expect("config");
    load(&metadata, &read, config).expect("loads without a crate lock");
}
