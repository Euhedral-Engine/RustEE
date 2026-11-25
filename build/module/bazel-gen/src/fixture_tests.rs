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
/// - `euhedral-memory`, with `cuda`;
/// - `user`, whose `cuda` turns on `euhedral-memory/cuda`, with the test `t`;
/// - `plain`, which has a `testing` feature its own dev-dependency on itself turns on, with the test `p`, linking neither.
fn members<'a>() -> Vec<Fixture<'a>> {
    vec![
        Fixture {
            name: "euhedral-memory",
            features: &[("cuda", &[]), ("default", &[])],
            deps: vec![],
            tests: &[],
        },
        Fixture {
            name: "user",
            features: &[("cuda", &["euhedral-memory/cuda"])],
            deps: vec![("euhedral-memory", false, Some("euhedral-memory"), &[])],
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

/// RustEE's device: `euhedral-memory/cuda` opens the card.
fn config() -> Config {
    Config {
        device: Some(Device {
            member: Some("euhedral-memory".to_owned()),
            feature: "cuda".to_owned(),
            external: Vec::new(),
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
    assert_eq!(features(World::Testing, "euhedral-memory"), ["default"]);
    assert_eq!(features(World::CudaTesting, "user"), ["cuda"]);
    assert_eq!(
        features(World::CudaTesting, "euhedral-memory"),
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
fn a_testing_world_with_the_device_feature_is_refused() {
    let mut members = members();
    // A dev-dependency asks for euhedral-memory/cuda: `cargo test --workspace` would open the card.
    members[2]
        .deps
        .push(("euhedral-memory", true, Some("euhedral-memory"), &["cuda"]));
    let ws = workspace(&members, "/w", "/").expect("the fixture loads");
    let error = generate(&ws).expect_err("the generator refuses");
    assert!(
        error.contains("testing world resolves euhedral-memory/cuda"),
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
    // An empty `default` turns on nothing: euhedral-memory without it is the same library.
    let mut empty = members();
    empty[2].deps.push((
        "euhedral-memory",
        false,
        Some("euhedral-memory"),
        &["!default"],
    ));
    generate(&workspace(&empty, "/w", "/").expect("the fixture loads"))
        .expect("an empty default may differ");
}

#[test]
fn unmodelled_manifests_are_refused() {
    let mut weak = members();
    weak[1].features = &[("cuda", &["euhedral-memory?/cuda"])];
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
        ("cuda", &["euhedral-memory/cuda"]),
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
    members[1].features = &[("cuda", &["euhedral-memory/gpu"])];
    let ws = workspace(&members, "/w", "/").expect("the fixture loads");
    let error = generate(&ws).expect_err("the generator refuses");
    assert!(
        error.contains("euhedral-memory with the feature `cuda`"),
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
        &|_| Ok("[package]\n".to_owned()),
        config(),
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
