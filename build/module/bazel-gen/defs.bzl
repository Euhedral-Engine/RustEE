"""The shared Rust BUILD generator's rules (README.md; build/CONTRACTS.md C6). The generator writes them into a workspace's
root BUILD.bazel when its bazel_gen.toml says `[root] gazelle = true` (the default):

    load("@euhedral_build//bazel-gen:defs.bzl", "bazel_gen", "bazel_gen_stale_test")

    bazel_gen(name = "gazelle")
    bazel_gen_stale_test(
        name = "bazel_gen_stale_test",
        config = "bazel_gen.toml",
        lock = "MODULE.bazel.lock",
    )

`bazel run //:gazelle` writes the BUILD files (`-- -mode=diff` only checks them); `bazel test //...` runs the stale test,
which fails when a BUILD file, or the crate hub's lock, is out of date. The lock is MODULE.bazel.lock (its record of the
manifests) for a hub without `lockfile`, or the hub's cargo-bazel-lock.json (`[root] crate_lock`), whose crates are compared
with Cargo.lock's packages.
"""

_CARGO = Label("@rules_rust//cargo:current_cargo_toolchain")
_GENERATOR = Label("//bazel-gen:euhedral-bazel-gen")

def _generator_impl(ctx, env):
    exe = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.symlink(output = exe, target_file = ctx.executable._generator, is_executable = True)
    cargo = ctx.file._cargo
    files = [cargo] + ctx.files.data
    env = dict(env)

    # Paths relative to where the binary starts, the runfiles root of the main repository (main.rs).
    env["CARGO"] = cargo.short_path
    if getattr(ctx.file, "config", None):
        files.append(ctx.file.config)
        env["BAZEL_GEN_CONFIG"] = ctx.file.config.short_path
    if getattr(ctx.file, "lock", None):
        files.append(ctx.file.lock)
        env["BAZEL_GEN_LOCK"] = ctx.file.lock.short_path
    runfiles = ctx.runfiles(files = files).merge(ctx.attr._generator[DefaultInfo].default_runfiles)
    return [
        DefaultInfo(executable = exe, runfiles = runfiles),
        RunEnvironmentInfo(environment = env),
    ]

_COMMON = {
    "data": attr.label_list(allow_files = True, doc = "Files the generator reads besides the manifests."),
    "_cargo": attr.label(default = _CARGO, allow_single_file = True),
    "_generator": attr.label(default = _GENERATOR, executable = True, cfg = "target"),
}

bazel_gen = rule(
    implementation = lambda ctx: _generator_impl(ctx, {}),
    attrs = _COMMON,
    executable = True,
    doc = "`bazel run` it to write the workspace's BUILD files; `-- -mode=diff` only checks them. It reads the root's " +
          "bazel_gen.toml from the workspace itself.",
)

_stale_test = rule(
    implementation = lambda ctx: _generator_impl(ctx, {"BAZEL_GEN_MODE": "diff"}),
    attrs = _COMMON | {
        "config": attr.label(allow_single_file = True, doc = "The workspace's bazel_gen.toml."),
        "lock": attr.label(allow_single_file = True, doc = "MODULE.bazel.lock or the hub's cargo-bazel-lock.json, to check against the manifests."),
    },
    test = True,
)

def bazel_gen_stale_test(name, config = None, lock = None, sources = "//:workspace_sources", data = [], **kwargs):
    """Fails when a BUILD file is not what the generator writes from the workspace's manifests.

    Args:
      name: the test's name.
      config: the workspace's bazel_gen.toml, when it has one.
      lock: MODULE.bazel.lock, or the hub's cargo-bazel-lock.json when bazel_gen.toml names it as `crate_lock`, to fail
        also when crate_universe's lock is behind the manifests or Cargo.lock.
      sources: every manifest, BUILD file and target file of the workspace (the root's generated filegroup).
      data: more files the generator reads.
      **kwargs: the test's common attributes (tags, size). The test always carries the tag `stale-build`, which the
        ci repo's bazel gate (`stale-build: tested`) selects it by.
    """
    kwargs["tags"] = sorted({tag: None for tag in kwargs.get("tags", []) + ["stale-build"]}.keys())
    data = [sources] + data
    if lock:
        data.append("//:Cargo.lock")
    _stale_test(
        name = name,
        config = config,
        lock = lock,
        data = data,
        **kwargs
    )
