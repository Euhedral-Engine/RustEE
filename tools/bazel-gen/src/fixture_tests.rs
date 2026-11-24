//! The generator end to end on a small workspace: feature-world resolution, the `gpu`/`external` tags, the refusals and
//! Windows paths.

use std::collections::BTreeMap;

use crate::emit::generate;
use crate::json;
use crate::model::{Workspace, load};
use crate::worlds::{World, resolve_world};

/// One dependency of the fixture: (name, dev, path member, features).
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
                format!(
                    r#"{{"name":{},"kind":{},"rename":null,"target":null,"optional":false,"uses_default_features":true,"features":{}{}}}"#,
                    json_str(name),
                    if *dev { "\"dev\"" } else { "null" },
                    list(features),
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

fn json_str(text: &str) -> String {
    format!("\"{}\"", text.replace('\\', "\\\\"))
}

fn workspace(members: &[Fixture], root: &str, sep: &str) -> Result<Workspace, String> {
    let metadata = json::parse(&metadata(members, root, sep))?;
    load(&metadata, &|_| Ok("[package]\n".to_owned()))
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

    let metadata =
        metadata(&members(), "/w", "/").replace(r#""rename":null"#, r#""rename":"other""#);
    let error = load(&json::parse(&metadata).expect("json"), &|_| {
        Ok(String::new())
    })
    .expect_err("renames");
    assert!(error.contains("renamed"), "{error}");
}

#[test]
fn windows_paths_generate_the_same_files() {
    let windows = workspace(&members(), r"C:\w", r"\").expect("the Windows fixture loads");
    assert_eq!(generate(&windows).expect("generates"), files(&members()));
}
