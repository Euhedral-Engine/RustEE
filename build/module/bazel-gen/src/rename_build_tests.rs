//! Renamed dependencies and build scripts, on workspaces written as `cargo metadata` JSON.

use std::collections::BTreeMap;

use crate::config::Config;
use crate::emit::generate;
use crate::json;
use crate::model::load;

/// One dependency as `cargo metadata` prints it. `kind` is `null`, `"dev"` or `"build"`.
fn dep(name: &str, rename: Option<&str>, kind: &str, path: Option<&str>, optional: bool) -> String {
    let rename = rename.map_or("null".to_owned(), |r| format!("\"{r}\""));
    let kind = if kind == "null" {
        kind.to_owned()
    } else {
        format!("\"{kind}\"")
    };
    let path = path
        .map(|p| format!(",\"path\":\"/w/{p}\""))
        .unwrap_or_default();
    format!(
        r#"{{"name":"{name}","rename":{rename},"kind":{kind},"target":null,"optional":{optional},"uses_default_features":true,"features":[]{path}}}"#
    )
}

/// One member in `dir` (its name), with a library and a test `t`, and a build script when `build`.
fn package(
    name: &str,
    features: &str,
    deps: &[String],
    build: bool,
    links: Option<&str>,
) -> String {
    let target = |kind: &str, tname: &str, src: &str| {
        let crate_type = if kind == "lib" { "lib" } else { "bin" };
        format!(
            r#"{{"name":"{tname}","kind":["{kind}"],"crate_types":["{crate_type}"],"src_path":"/w/{name}/{src}","edition":"2024","test":true,"doctest":{}}}"#,
            kind == "lib"
        )
    };
    let mut targets = vec![
        target("lib", &name.replace('-', "_"), "src/lib.rs"),
        target("test", "t", "tests/t.rs"),
    ];
    if build {
        targets.push(target("custom-build", "build-script-build", "build.rs"));
    }
    let links = links
        .map(|l| format!(",\"links\":\"{l}\""))
        .unwrap_or_default();
    format!(
        r#"{{"name":"{name}","version":"0.1.0","manifest_path":"/w/{name}/Cargo.toml","features":{{{features}}},"dependencies":[{}],"targets":[{}],"metadata":null{links}}}"#,
        deps.join(","),
        targets.join(",")
    )
}

fn generated(packages: &[String], config: &str) -> Result<BTreeMap<String, String>, String> {
    let metadata = json::parse(&format!(
        r#"{{"workspace_root":"/w","packages":[{}]}}"#,
        packages.join(",")
    ))?;
    let config = Config::parse(config)?;
    generate(&load(&metadata, &|_| Ok("[package]\n".to_owned()), config)?)
}

fn rule<'a>(file: &'a str, name: &str) -> &'a str {
    let at = file
        .find(&format!("    name = \"{name}\",\n"))
        .unwrap_or_else(|| panic!("no rule {name} in\n{file}"));
    let start = file[..at - 1].rfind('\n').map_or(0, |i| i + 1);
    &file[start..at + file[at..].find("\n)\n").expect("a rule ends") + 3]
}

#[test]
fn a_renamed_member_dependency_is_aliased_in_the_library_and_the_tests() {
    let packages = [
        package("base", "", &[], false, None),
        package(
            "user",
            "",
            &[
                dep("base", Some("other-base"), "null", Some("base"), false),
                dep("base", Some("dev-base"), "dev", Some("base"), false),
            ],
            false,
            None,
        ),
    ];
    let error = generated(&packages, "").expect_err("two keys of one label");
    assert!(
        error.contains("`dev-base`") && error.contains("`other-base`"),
        "{error}"
    );

    let packages = [
        package("base", "", &[], false, None),
        package(
            "user",
            "",
            &[dep("base", Some("other-base"), "null", Some("base"), false)],
            false,
            None,
        ),
    ];
    let files = generated(&packages, "").expect("generates");
    let user = &files["user/BUILD.bazel"];
    let lib = rule(user, "user");
    assert!(lib.contains("deps = [\"//base:base\"]"), "{lib}");
    assert!(
        lib.contains("aliases = {\n        \"//base:base\": \"other_base\",\n    }"),
        "{lib}"
    );
    let test = rule(user, "t_test");
    assert!(test.contains("\"//base:base\": \"other_base\""), "{test}");
}

#[test]
fn a_renamed_crates_universe_dependency_is_aliased() {
    let packages = [package(
        "user",
        "",
        &[dep("serde", Some("ser"), "null", None, false)],
        false,
        None,
    )];
    let files = generated(&packages, "").expect("generates");
    let lib = rule(&files["user/BUILD.bazel"], "user");
    assert!(lib.contains("deps = [\"@crates//:serde\"]"), "{lib}");
    assert!(lib.contains("\"@crates//:serde\": \"ser\""), "{lib}");
    // The key's own spelling, `-` as `_`, is the name; a key equal to the package's crate name needs none.
    let packages = [package(
        "user",
        "",
        &[dep("serde-json", Some("serde_json"), "null", None, false)],
        false,
        None,
    )];
    let files = generated(&packages, "").expect("generates");
    assert!(!rule(&files["user/BUILD.bazel"], "user").contains("aliases"));
}

#[test]
fn a_renamed_optional_dependency_is_turned_on_by_its_key() {
    let packages = [package(
        "user",
        r#""default":["fast"],"fast":["dep:ser"]"#,
        &[dep("serde", Some("ser"), "null", None, true)],
        false,
        None,
    )];
    let files = generated(&packages, "").expect("generates");
    let lib = rule(&files["user/BUILD.bazel"], "user");
    assert!(lib.contains("\"@crates//:serde\": \"ser\""), "{lib}");
    assert!(lib.contains("\"fast\""), "{lib}");
}

const GIT: &str = "[external.crates]\next = \"@m//ext\"\n";

#[test]
fn a_renamed_git_dependency_picks_its_world_by_its_key_and_two_keys_of_one_label_are_refused() {
    let both = [
        dep("ext", None, "null", None, false),
        dep("ext", Some("ext-api"), "null", None, false),
    ];
    let packages = [package("user", "", &both, false, None)];
    // One entry: both keys reach one label in every world.
    let error = generated(&packages, GIT).expect_err("one label, two names");
    assert!(
        error.contains("`ext`") && error.contains("`ext-api`"),
        "{error}"
    );
    assert!(error.contains("@m//ext:ext"), "{error}");

    // The renamed key has its own entry: `ext-api` is the fake world of the package in each of ours.
    let config = format!(
        "{GIT}ext-api = {{ package = \"@m//ext\", worlds = {{ production = \"_fake\", testing = \"_fake\" }} }}\n"
    );
    let files = generated(&packages, &config).expect("each key has its own label");
    let user = &files["user/BUILD.bazel"];
    let lib = rule(user, "user");
    assert!(lib.contains("\"@m//ext:ext\""), "{lib}");
    assert!(lib.contains("\"@m//ext:ext_fake\": \"ext_api\""), "{lib}");
    assert!(!lib.contains("\"@m//ext:ext\": \"ext_api\""), "{lib}");
    // The testing world's library takes the `_testing`-less fake label of its own entry.
    assert!(rule(user, "user_testing").contains("\"@m//ext:ext_fake\": \"ext_api\""));

    // A renamed key without its own entry shares the package's.
    let packages = [package(
        "user",
        "",
        &[dep("ext", Some("ext-api"), "null", None, false)],
        false,
        None,
    )];
    let files = generated(&packages, GIT).expect("generates");
    let lib = rule(&files["user/BUILD.bazel"], "user");
    assert!(lib.contains("\"@m//ext:ext\": \"ext_api\""), "{lib}");
}

const SCRIPT_CONFIG: &str = "[packages.gpu]\nbuild_script_env = { CUDA_HOME = \"/opt/cuda\" }\n\
    build_script_data = [\"kernels/**\"]\nbuild_script_tags = [\"external\", \"gpu\"]\n";

#[test]
fn a_package_with_a_build_script_gets_a_cargo_build_script_its_targets_depend_on() {
    let packages = [
        package("base", "", &[], false, None),
        package(
            "gpu",
            r#""cuda":[],"default":["cuda"]"#,
            &[
                dep("base", None, "null", Some("base"), false),
                dep("base", Some("build-base"), "build", Some("base"), false),
                dep("cc", None, "build", None, false),
            ],
            true,
            Some("cudart"),
        ),
    ];
    let files = generated(&packages, SCRIPT_CONFIG).expect("generates");
    let gpu = &files["gpu/BUILD.bazel"];
    assert!(
        gpu.contains("load(\"@rules_rust//cargo:defs.bzl\", \"cargo_build_script\")"),
        "{gpu}"
    );
    let script = rule(gpu, "build_script");
    assert!(script.starts_with("cargo_build_script("), "{script}");
    for want in [
        "srcs = [\"build.rs\"]",
        "crate_features = [\n        \"cuda\",\n        \"default\",\n    ]",
        "\"//base:base\",\n        \"@crates//:cc\"",
        "\"//base:base\": \"build_base\"",
        "build_script_env = {\n        \"CUDA_HOME\": \"/opt/cuda\",\n    }",
        "\"kernels/**\"",
        "\"external\",\n        \"gpu\",",
        "links = \"cudart\"",
    ] {
        assert!(script.contains(want), "{want} not in\n{script}");
    }
    // The build script has only the build-dependencies; the library only the normal ones and the script.
    let lib = rule(gpu, "gpu");
    assert!(lib.contains("\":build_script\""), "{lib}");
    assert!(
        !lib.contains("@crates//:cc") && !lib.contains("build_base"),
        "{lib}"
    );
    assert!(rule(gpu, "t_test").contains("\":build_script\""));
    // The testing world has the same features and dependencies: the library and the script alias the first.
    assert!(rule(gpu, "build_script_testing").starts_with("alias("));
    assert!(rule(gpu, "gpu_testing").starts_with("alias("));
    // The root package has no build script load.
    assert!(!files["base/BUILD.bazel"].contains("cargo_build_script"));
}

#[test]
fn build_script_settings_of_a_package_without_one_are_refused() {
    let packages = [package("gpu", "", &[], false, None)];
    let error = generated(&packages, SCRIPT_CONFIG).expect_err("no build script");
    assert!(error.contains("gpu has no build script"), "{error}");
}

#[test]
fn a_build_script_that_is_not_build_rs_is_refused() {
    let metadata = r#"{"workspace_root":"/w","packages":[{"name":"p","version":"0.1.0","manifest_path":"/w/p/Cargo.toml","features":{},"dependencies":[],"targets":[{"name":"build-script-build","kind":["custom-build"],"crate_types":["bin"],"src_path":"/w/p/build/main.rs","edition":"2024"}],"metadata":null}]}"#;
    let error = load(
        &json::parse(metadata).expect("json"),
        &|_| Ok(String::new()),
        Config::default(),
    )
    .expect_err("another file");
    assert!(error.contains("build/main.rs"), "{error}");
}

/// Members `a` and `b` in a scratch workspace holding `files` (root-relative path, content).
fn path_workspace(
    tag: &str,
    files: &[(&str, &str)],
) -> Result<(BTreeMap<String, String>, std::path::PathBuf), String> {
    let root = std::env::temp_dir().join(format!("bazel-gen-path-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    for (path, content) in files {
        let at = root.join(path);
        std::fs::create_dir_all(at.parent().unwrap()).unwrap();
        std::fs::write(at, content).unwrap();
    }
    let packages = [
        package("a", "", &[], false, None),
        package("b", "", &[], false, None),
    ];
    let metadata = json::parse(&format!(
        r#"{{"workspace_root":"/w","packages":[{}]}}"#,
        packages.join(",")
    ))?;
    let mut ws = load(
        &metadata,
        &|_| Ok("[package]\n".to_owned()),
        Config::parse("")?,
    )?;
    let found = crate::find_path_modules(&mut ws, &root);
    let _ = std::fs::remove_dir_all(&root);
    found?;
    Ok((generate(&ws)?, root))
}

#[test]
fn a_path_into_another_member_is_exported_and_listed_in_srcs() {
    let (files, _) = path_workspace(
        "member",
        &[
            (
                "a/src/lib.rs",
                "#[path = \"../../b/src/shared.rs\"]\nmod shared;\n",
            ),
            (
                "a/tests/t.rs",
                "#[path = \"../../b/src/shared.rs\"]\nmod shared;\n",
            ),
            ("b/src/lib.rs", ""),
            ("b/src/shared.rs", ""),
        ],
    )
    .expect("generates");
    let a = &files["a/BUILD.bazel"];
    assert!(rule(a, "a").contains("\"//b:src/shared.rs\""), "{a}");
    assert!(rule(a, "t_test").contains("\"//b:src/shared.rs\""), "{a}");
    let b = &files["b/BUILD.bazel"];
    assert!(
        b.contains("exports_files([\"src/shared.rs\"], visibility = [")
            && b.contains("\"//a:__pkg__\""),
        "{b}"
    );
    assert!(!files["BUILD.bazel"].contains("shared.rs"));
}

#[test]
fn a_path_into_a_non_member_directory_belongs_to_the_root_package() {
    let (files, _) = path_workspace(
        "root",
        &[
            ("a/src/lib.rs", "#[path = \"../../extra/x.rs\"]\nmod x;\n"),
            ("b/src/lib.rs", ""),
            ("extra/x.rs", ""),
        ],
    )
    .expect("generates");
    assert!(rule(&files["a/BUILD.bazel"], "a").contains("\"//:extra/x.rs\""));
    assert!(
        files["BUILD.bazel"]
            .contains("exports_files([\"extra/x.rs\"], visibility = [\"//a:__pkg__\"])"),
        "{}",
        files["BUILD.bazel"]
    );
    assert!(!files.contains_key("extra/BUILD.bazel"));
    // The stale test reads the file too.
    assert!(rule(&files["BUILD.bazel"], "workspace_sources").contains("\"extra/x.rs\""));
}

#[test]
fn a_path_that_bazel_cannot_see_is_refused_naming_the_file() {
    let source = ("a/src/lib.rs", "#[path = \"../../extra/x.rs\"]\nmod x;\n");
    let error = path_workspace(
        "ignored",
        &[source, ("extra/x.rs", ""), (".bazelignore", "extra\n")],
    )
    .expect_err("ignored");
    assert!(
        error.contains("extra/x.rs") && error.contains("`extra`") && error.contains("narrow"),
        "{error}"
    );
    let error = path_workspace(
        "outside",
        &[("a/src/lib.rs", "#[path = \"../../../x.rs\"]\nmod x;\n")],
    )
    .expect_err("outside");
    assert!(
        error.contains("a/src/lib.rs") && error.contains("leaves the workspace"),
        "{error}"
    );
    let error = path_workspace(
        "other-package",
        &[source, ("extra/x.rs", ""), ("extra/BUILD.bazel", "")],
    )
    .expect_err("a package the generator does not write");
    assert!(
        error.contains("extra/x.rs") && error.contains("not a workspace member"),
        "{error}"
    );
}

#[test]
fn what_a_foreign_file_includes_and_what_tests_name_in_src_is_carried_along() {
    let (files, _) = path_workspace(
        "include",
        &[
            (
                "a/src/lib.rs",
                "#[path = \"../../b/src/shared.rs\"]\nmod shared;\n",
            ),
            ("a/src/helper.rs", ""),
            (
                "a/tests/t.rs",
                "#[path = \"../src/helper.rs\"]\nmod helper;\nfn main() {}\n",
            ),
            ("b/src/lib.rs", ""),
            (
                "b/src/shared.rs",
                "const D: &[u8] = include_bytes!(\"../data/d.bin\");\n",
            ),
            ("b/data/d.bin", ""),
        ],
    )
    .expect("generates");
    let a = &files["a/BUILD.bazel"];
    assert!(rule(a, "a").contains("\"//b:data/d.bin\""), "{a}");
    assert!(rule(a, "t_test").contains("\"//a:src/helper.rs\""), "{a}");
    assert!(!rule(a, "a").contains("//a:src/helper.rs"), "{a}");
    assert!(files["b/BUILD.bazel"].contains("\"data/d.bin\""));
}

#[test]
fn literal_includes_of_every_source_file_become_compile_data() {
    let (files, _) = path_workspace(
        "includes",
        &[
            (
                "a/src/lib.rs",
                "const D: &[u8] = include_bytes!(\"../data/d.bin\");\nconst J: &str = include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/data/j.json\"));\nconst F: &str = include_str!(\"../../b/fixtures/f.json\");\nconst R: &str = include_str!(\"../../extra/r.txt\");\nconst V: &str = include_str!(concat!(env!(\"OUT_DIR\"), \"/v\"));\n",
            ),
            ("a/tests/t.rs", "const T: &str = include_str!(\"../data/j.json\");\nconst U: &str = include_str!(\"fixtures/t.txt\");\n"),
            ("a/data/d.bin", ""),
            ("a/data/j.json", ""),
            ("a/tests/fixtures/t.txt", ""),
            ("b/src/lib.rs", ""),
            ("b/fixtures/f.json", ""),
            ("extra/r.txt", ""),
        ],
    )
    .expect("generates");
    let a = rule(&files["a/BUILD.bazel"], "a");
    for want in [
        "data/d.bin",
        "data/j.json",
        "//b:fixtures/f.json",
        "//:extra/r.txt",
    ] {
        assert!(a.contains(&format!("\"{want}\"")), "{want} in\n{a}");
    }
    assert!(!a.contains("OUT_DIR") && !a.contains("/v\""), "{a}");
    let t = rule(&files["a/BUILD.bazel"], "t_test");
    assert!(
        t.contains("\"//b:fixtures/f.json\"")
            && t.contains("\"data/j.json\"")
            && !t.contains("t.txt"),
        "{t}"
    );
    assert!(
        files["b/BUILD.bazel"].contains("exports_files([\"fixtures/f.json\"]"),
        "{}",
        files["b/BUILD.bazel"]
    );
    assert!(files["BUILD.bazel"].contains("exports_files([\"extra/r.txt\"]"));
}

#[test]
fn an_include_that_cannot_be_resolved_is_refused_naming_the_file() {
    let lib = |target: &'static str| -> Vec<(&'static str, String)> {
        vec![
            (
                "a/src/lib.rs",
                format!("const X: &str = include_str!(\"{target}\");\n"),
            ),
            ("extra/r.txt", String::new()),
        ]
    };
    let run = |tag: &str, target: &'static str, ignore: Option<&'static str>| {
        let mut files = lib(target);
        if let Some(entry) = ignore {
            files.push((".bazelignore", entry.to_owned()));
        }
        let files: Vec<(&str, &str)> = files.iter().map(|(p, c)| (*p, c.as_str())).collect();
        path_workspace(tag, &files).expect_err("refused")
    };
    let error = run("inc-missing", "missing.txt", None);
    assert!(
        error.contains("a/src/missing.txt") && error.contains("does not exist"),
        "{error}"
    );
    let error = run("inc-outside", "../../../x.txt", None);
    assert!(
        error.contains("a/src/lib.rs") && error.contains("leaves the workspace"),
        "{error}"
    );
    let error = run("inc-ignored", "../../extra/r.txt", Some("extra\n"));
    assert!(
        error.contains("extra/r.txt") && error.contains("`extra`"),
        "{error}"
    );
}

#[test]
fn an_integration_test_gets_the_path_of_each_binary_of_its_package() {
    let tool = r#"{"name":"tool","kind":["bin"],"crate_types":["bin"],"src_path":"/w/a/src/main.rs","edition":"2024","test":true,"doctest":false}"#;
    let a = package("a", "", &[], false, None).replacen(
        "\"targets\":[",
        &format!("\"targets\":[{tool},"),
        1,
    );
    let files = generated(&[a], "").expect("generates");
    let build = &files["a/BUILD.bazel"];
    let test = rule(build, "t_test");
    assert!(
        test.contains("\"CARGO_BIN_EXE_tool\": \"$(rootpath :tool)\"")
            && test.contains("data = [\":tool\"]"),
        "{test}"
    );
    // Only integration tests: the unit tests and the binary itself do not get it.
    assert!(!rule(build, "tool").contains("CARGO_BIN_EXE"), "{build}");
    assert!(
        !rule(build, "unit_test").contains("CARGO_BIN_EXE"),
        "{build}"
    );
}

#[test]
fn a_renamed_external_dependency_that_turns_on_the_device_feature_is_refused_in_the_testing_world()
{
    let cuda = dep("ext", Some("gpu-cuda"), "null", None, false)
        .replace("\"features\":[]", "\"features\":[\"cuda\"]");
    let packages = [
        package("m", r#""cuda":[]"#, &[], false, None),
        package("user", "", &[cuda], false, None),
    ];
    let config = "[device]\nmember = \"m\"\nfeature = \"cuda\"\nexternal = [\"ext\"]\n\n\
        [external.crates]\next = \"@m//ext\"\ngpu-cuda = \"@m//ext\"\n";
    let error = generated(&packages, config)
        .expect_err("the renamed key opens the card in the testing world");
    assert!(error.contains("testing world resolves"), "{error}");
}

#[test]
fn a_renamed_dev_dependency_is_not_given_to_the_doctests_as_aliases() {
    let packages = [
        package("base", "", &[], false, None),
        package(
            "user",
            "",
            &[dep("base", Some("dev-base"), "dev", Some("base"), false)],
            false,
            None,
        ),
    ];
    let error = generated(&packages, "").expect_err("a doctest cannot name the dependency");
    assert!(
        error.contains("doctests") && error.contains("`dev_base`") && error.contains("aliases"),
        "{error}"
    );
}

#[test]
fn an_integration_test_runs_the_binary_built_in_its_own_world() {
    let tool = r#"{"name":"tool","kind":["bin"],"crate_types":["bin"],"src_path":"/w/a/src/main.rs","edition":"2024","test":true,"doctest":false}"#;
    let a = package("a", r#""cuda":[]"#, &[], false, None).replacen(
        "\"targets\":[",
        &format!("\"targets\":[{tool},"),
        1,
    );
    let config = "[device]\nmember = \"a\"\nfeature = \"cuda\"\n";
    let files = generated(&[a], config).expect("generates");
    let build = &files["a/BUILD.bazel"];
    let plain = rule(build, "t_test");
    assert!(
        plain.contains("data = [\":tool\"]") && !plain.contains("gpu"),
        "{plain}"
    );
    let cuda = rule(build, "t_cuda_test");
    assert!(
        cuda.contains("data = [\":tool_cuda_testing\"]")
            && cuda.contains("\"CARGO_BIN_EXE_tool\": \"$(rootpath :tool_cuda_testing)\"")
            && cuda.contains("\"gpu\""),
        "{cuda}"
    );
    let bin = rule(build, "tool_cuda_testing");
    assert!(bin.contains("crate_features = [\"cuda\"]"), "{bin}");
}

#[test]
fn a_build_script_tag_set_is_derived_from_gpu_and_from_absolute_paths_in_its_environment() {
    let build = |config: &str| {
        let files = generated(&[package("gpu", "", &[], true, None)], config).expect("generates");
        rule(&files["gpu/BUILD.bazel"], "build_script").to_owned()
    };
    let script = build(
        "[packages.gpu]\nbuild_script_env = { CUDA_HOME = \"/opt/cuda\" }\nbuild_script_tags = [\"gpu\"]\n",
    );
    assert!(
        script.contains(
            "tags = [\n        \"external\",\n        \"gpu\",\n        \"no-cache\",\n        \"no-remote-cache\",\n    ]"
        ),
        "{script}"
    );
    // A relative value and no gpu tag derive nothing.
    let script = build(
        "[packages.gpu]\nbuild_script_env = { MODE = \"fast\" }\nbuild_script_tags = [\"x\"]\n",
    );
    assert!(
        script.contains("tags = [\"x\"]") && !script.contains("no-cache"),
        "{script}"
    );
}

#[test]
fn an_own_file_outside_the_source_directories_is_gathered_for_the_stale_test() {
    let (files, _) = path_workspace(
        "own-files",
        &[
            (
                "a/src/lib.rs",
                "const S: &str = include_str!(\"../schema.sql\");\n",
            ),
            (
                "a/build.rs",
                "const K: &str = include_str!(\"kernels/x\");\n",
            ),
            (
                "a/tests/t.rs",
                "const T: &str = include_str!(\"../data/t.txt\");\n",
            ),
            ("a/schema.sql", ""),
            ("a/kernels/x", ""),
            ("a/data/t.txt", ""),
            ("b/src/lib.rs", ""),
        ],
    )
    .expect("generates");
    let a = &files["a/BUILD.bazel"];
    let sources = rule(a, "sources");
    for want in ["schema.sql", "kernels/x", "data/t.txt"] {
        assert!(
            sources.contains(&format!("\"{want}\"")),
            "{want} in\n{sources}"
        );
    }
    assert!(!rule(&files["b/BUILD.bazel"], "sources").contains("schema"));
}
