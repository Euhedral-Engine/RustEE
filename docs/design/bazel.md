# Building and testing RustEE with Bazel (design)

Date: 2026-11-23. Status: proposed, revised after review. The build starts only after forge's rollover (phase 3)
merges; no step below merges before it, and merging any Bazel change into `main` needs the user's word.

## Why

The machine's repositories build and test with Bazel under one set of contracts (forge `build/CONTRACTS.md`, C1 to C10),
so that builds and test results are cached across worktrees, CI and repositories. forge's own Rust crate settled the
single-crate case: rules_rust 0.75.0, crate_universe `crate.from_cargo`, hand-written BUILD files with globs and
`all_crate_deps`, and a layout check that refuses what a glob cannot see.

That pattern does not carry over to RustEE. RustEE is a workspace of ten crates with:

- features, some of them on by default (`euhedral-io` is empty without `async-io`, so its tests would pass vacuously
  without it);
- self dev-dependencies that turn on test features (`euhedral-gpu` gets `fake` and `test-hooks`, `euhedral-graph` gets
  `testing`);
- examples with `required-features = ["cuda"]`;
- `tests/common` modules and kernel files that integration tests include (`include_str!`, `include_bytes!`);
- `cfg(loom)` dependencies, and `[lints]` tables with `check-cfg`;
- a CUDA backend (`euhedral-memory`, `euhedral-gpu`) and a benchmark crate with three binaries;
- no `rust-toolchain.toml` (the version is pinned in `mise.toml`).

`all_crate_deps` drops dev-dependencies and optional dependencies whose features are off, and a glob cannot know which
features a target needs. This document decides how RustEE's BUILD files are made, how each target gets its features,
where the toolchain is pinned, what runs under Bazel and what stays on Cargo, and how to prove that Bazel runs every test
Cargo runs. It ends with the build steps and the changes it proposes to the contracts.

Bazel is added; nothing is replaced. The gate keeps its fmt, workspace clippy, cuda clippy, tests, release stress tests,
loom, Miri shards, Windows and device tests on Cargo.

## Facts the design rests on

These were read from the workspace at the time of writing, and checked with a prototype (below):

- No crate has a `build.rs`. Nothing needs `cargo_build_script` today.
- `euhedral-memory` and `euhedral-gpu` load the CUDA driver (`libcuda.so.1`, `nvcuda.dll`) and NVRTC at run time
  (`dlopen`, `LoadLibrary`). They link nothing from the CUDA toolkit and compile no CUDA at build time. Kernels are
  compiled by NVRTC when a test runs; `tests/kernels/ops.ptx` is a checked-in file.
- `cargo metadata` lists every target, with its kind, root file, edition and `required-features`. Under
  `cargo test --workspace` (CI's `--workspace` run) the first-party features resolve to: `euhedral-gpu`
  `default, fake, test-hooks`; `euhedral-graph` `testing`; `euhedral-io` `async-io, default`; the rest none.
- `cargo test --workspace` runs 2307 tests: unit, integration, the unit tests of the benchmark binaries, and the
  doctests (the READMEs are doctests).
- Tests read three Cargo variables: `env!("CARGO_MANIFEST_DIR")` to read the workspace's sources
  (`euhedral-core/tests/hot_fix_ratchet.rs`, `euhedral-gpu/tests/layering.rs`), and `env!("CARGO_TARGET_TMPDIR")` or the
  test binary's own directory for scratch files (`euhedral-io` and `euhedral-gpu`, six sites).
- Miri's nightly is selected with `RUSTUP_TOOLCHAIN` (`scripts/ci/miri-toolchain`), which outranks a
  `rust-toolchain.toml`. The GitHub workspace job sets the version with `rustup default`, which a `rust-toolchain.toml`
  overrides (with the same version, so nothing changes).

## The prototype

The prototype ran on a copy of the workspace in a scratch directory, with Bazel 9.3.3, rules_rust 0.75.0 and the
machine's caches, and was deleted afterwards. It covered all ten crates, not only two or three, because the generator
treats every crate the same way.

- **Generator:** a throwaway Python script of about 300 lines (the product generator is a Rust crate, below). It
  reads `cargo metadata --no-deps` and writes one `BUILD.bazel` per crate. `bazel build //...` succeeds, with no
  warnings once two problems listed below were fixed. That is 1129 actions, covering the testing and cuda testing
  worlds and the CUDA examples, with a peak of 5.5 GB. The production world (below) was designed after the prototype
  and not built under Bazel; see the next item.
- **Production world, checked with Cargo:** `cargo clippy --workspace --lib -- -D warnings` builds every library with
  only its own defaults, as the production world does. Nine crates are clean. `euhedral-gpu` is not: with neither
  `fake` nor `cuda` it has no device backend, and 15 items are dead code (`api.rs`, `hold.rs`, `lanes.rs`, `spans.rs`,
  `tensor_map.rs`). No gate run builds that combination today. It is a lint state, not a fault, and step 2 covers it.
- **Tests:** `bazel test //...` passes all non-`gpu` targets. Six failed before two kinds of fixes:
  - the scratch-directory edits (step 2 below);
  - for the two source-reading tests, `rustc_env` and `data` (below).

  io_uring runs in the Linux sandbox.
- **Counting:** the test names were compared per binary (unit, each integration file, each bin, the doctests):
  - `cargo test --workspace -- --list` against the Bazel targets: 2307 tests each, the same names in every binary.
  - The CUDA world, `cargo test -p euhedral-gpu -p euhedral-memory --features cuda -- --list` against the
    `gpu`-tagged targets run with `--list`: 376 unit and integration tests each, the same names. Nothing was run on the
    card.
- **Stale check:** an `sh_test` ran the toolchain's `cargo metadata --no-deps --offline` inside the sandbox, then the
  generator in check mode. It passes on fresh files. It fails, naming `euhedral-hashing/BUILD.bazel`, when a new
  `tests/*.rs` file is added.
- **`CARGO_MANIFEST_DIR`:** rules_rust sets it at compile time to the sandbox's execution root, which no longer exists
  when the test runs. Setting `rustc_env = {"CARGO_MANIFEST_DIR": "<package>"}` (a path relative to the runfiles root,
  where Bazel tests start) and making the workspace's sources `data` makes `hot_fix_ratchet` pass unedited. The ratchet
  asserts exact counts per crate, so a missing source fails it rather than passing vacuously.
- **rustc flags:** `aquery` shows fastbuild passes `opt-level=0`, so `debug_assertions` are on as in `cargo test`.
  `-c opt` passes `opt-level=3`.

Two rules_rust behaviours the prototype found, which the generator must handle:

1. `rust_test(crate = ...)` and `rust_doc_test` do **not** inherit the library's `crate_features`. Without repeating
   them, `euhedral-gpu`'s unit tests compile without `fake` and its fake-device tests disappear. The generator writes
   `crate_features` on every target.
2. rules_rust passes no `--check-cfg` for features. A `rust_lint_config` with any `check-cfg` turns checking on, so the
   generator lists `feature` (the crate's features), `test` and `docsrs` as well as the crate's own names (`loom`).

## Decisions

### First-party BUILD files: a checked-in generator driven by `cargo metadata`

A Rust crate in the workspace, `tools/bazel-gen` (`euhedral-bazel-gen`, `publish = false`), reads
`cargo metadata --no-deps --format-version 1` and writes every crate's `BUILD.bazel`. The file starts with an
`@generated` line, and nobody edits it by hand. It is Rust because new tools here are Rust by default (Python is for
throwaway scripts), so Cargo builds and tests it like every other member and Bazel needs no Python toolchain. Its
dependencies are `cargo_metadata` (the metadata types), `cargo-platform` (parsing target `cfg` strings) and `toml`
(`[lints]`, `mise.toml`, `rust-toolchain.toml`), pinned in `Cargo.lock` like every dependency. It moves to forge once a
second Cargo workspace uses it.

- **Targets come from Cargo, not from globs:** one `rust_library` per lib, one `rust_test` for the lib's unit tests, one
  `rust_doc_test`, one `rust_test` per `[[test]]` target, one `rust_binary` per bin (plus a `rust_test` for its unit
  tests) and per example. The generator stops with a message on anything it does not model: a `custom-build` or
  `bench` target, a renamed dependency, a weak `dep?/feature` entry, a first-party dependency under `[target.*]`, a
  target `cfg` it does not know, or `euhedral-memory/cuda` in the testing world. None of these exists today. A new kind is a generator change, never a silent omission.
- **Sources within a target are globs:** `src/**/*.rs` minus the bins; `tests/<name>.rs` plus `tests/*/**/*.rs` (the
  `common` modules); `examples/<name>.rs` plus `examples/<name>/**`. Non-Rust files beside them (`README.md`, the
  kernels) are `compile_data`. A new module file therefore needs no new BUILD file. Only a `Cargo.toml` change or a new
  target file does.
- **Dependencies are explicit labels:** `//<crate>:<target>` for workspace crates, and `@crates//:<name>` for
  third-party ones. The generator does not use `all_crate_deps`, because crate_universe resolves workspace members'
  optional dependencies only in the default world (`euhedral-memory`'s `libc` and `windows-sys`, needed by `cuda`, are
  absent from it). A target `cfg` becomes a `select` on `@platforms//os:*`. `cfg(loom)` is skipped, because loom stays
  on Cargo. A third-party feature that only a non-default world needs is added with `crate.annotation(crate_features)`
  in `MODULE.bazel`. None is needed today.
- **Lints:** the `[lints]` table becomes a `rust_lint_config` on every target.
- **Per-target extras that Cargo cannot express** live in `[package.metadata.bazel]` in the crate's `Cargo.toml`, which
  Cargo ignores:
  - `test.<name>.workspace_sources = true` for `hot_fix_ratchet` and `layering`. It gives the test
    `data = ["//:workspace_sources"]` (one `sources` filegroup per crate, gathered at the root, because a glob cannot
    cross packages) and the package-relative `CARGO_MANIFEST_DIR`.
  - `test.<name>.cpus = N` becomes `tags = ["cpu:N"]`, so Bazel's scheduler reserves the cores a lattice test spins
    up. Candidates are the graph tests that read `EUHEDRAL_TEST_CPUS` (`credit_graph`, `lane_source_lattice`,
    `lane_source_liveness`). The generator step sets them from what those tests start.
- **Integration tests** get `CARGO_TARGET_TMPDIR` set to a path that does not exist (Cargo sets it and the code reads it
  at compile time). Under Bazel the tests use `TEST_TMPDIR` instead (step 2).
- **`//:gazelle`** is the crate's `rust_binary`. Run as `bazel run //:gazelle`, it calls the toolchain's cargo and writes
  into `BUILD_WORKSPACE_DIRECTORY`. `bazel run //:gazelle -- -mode=diff` prints the stale files and fails, as C6's
  command does. `cargo run -p euhedral-bazel-gen` does the same without Bazel. That is how the first BUILD files
  (including the generator's own) are written, and it is the recovery path when stale BUILD files keep `//:gazelle`
  itself from building. Labels and paths in the output always use `/`, also when the generator runs on Windows, so
  both systems write the same bytes. There are no Go or Python packages in RustEE for Gazelle itself to
  manage.
- **The stale check** is a `rust_test` of the crate (`//tools/bazel-gen:stale_test`), so it is inside
  `bazel test //...`. It runs `cargo metadata --no-deps --offline` with the cargo in `CARGO` and then the generator in
  check mode:
  - Under Bazel, `CARGO` is the toolchain's cargo (`@rules_rust//rust/toolchain:current_cargo_files` as `data`) and
    `CARGO_HOME` is under `TEST_TMPDIR`. Its `data` is every `Cargo.toml`, `Cargo.lock` and `cargo-bazel-lock.json` (both
    read by the lock check below), every `BUILD.bazel` and every file that target discovery scans (`src/`, `tests/`,
    `examples/`, `benches/`, `build.rs`). Without them in `data` the sandbox has no copy to read.
  - Under `cargo test`, Cargo sets `CARGO` itself, so the same test also fails the Cargo gate on a stale file.
    `affected.py` marks `euhedral-bazel-gen` affected by any member's `Cargo.toml` and by any change at all under a
    member's `src/`, `tests/`, `examples/` or `benches/` or to its `build.rs` (step 3). Only an added or removed target file can make a
    BUILD file stale, but `affected.py` cannot tell one from an edit: `changed_files` runs
    `git diff --name-only --no-renames`, and `--files` takes plain paths. Counting every change there keeps both
    inputs as they are. The alternative, `--name-status` with a status per path, would change `--files` (its callers
    pass no status) and every function that takes the file list, to save a test that takes seconds. Over-selecting
    only runs a cheap test; under-selecting would let a stale file through, so the Cargo side runs the test on every
    pull request that can make BUILD files stale, and some that cannot.
  - **The locks:** `crate.from_cargo` sets `lockfile = "//:cargo-bazel-lock.json"`, because Bazel refuses a
    non-root module's crate_universe repository without one (a downstream module could not depend on this one).
    With `lockfile` set, rules_rust 0.75.0 treats the `crate` extension as reproducible (`extensions.bzl`), so it
    leaves `MODULE.bazel.lock` (no `%crate`, `FILE:@@//...Cargo.toml` or `ENV:` lines remain), and crate_universe's own
    digest of `cargo-bazel-lock.json` hashes the lock, the config, the splicing manifest and the cargo and rustc
    versions, but not `Cargo.lock` and not the contents of a manifest. **A local Bazel build therefore does not notice a
    stale crate lock**: a `cargo update --precise`, a changed requirement or a feature change in a member's
    `Cargo.toml` all build without a word. Two checks of ours enforce it. The stale test compares every package of
    `Cargo.lock` that is not a workspace member (`name version`, whatever its source) with the crates of
    `cargo-bazel-lock.json`, both ways, and names the packages that differ; it compares packages and versions only, not
    features, and runs under Cargo and Bazel. The `bazel` CI job's step "The crate lock is current" repins a copy before
    `bazel test` and fails when `cargo-bazel-lock.json`, `MODULE.bazel.lock` or `Cargo.lock` changes, which catches
    what the package comparison does not (a feature or requirement change, a toolchain bump). The repin is
    deterministic (two runs give identical files). One command refreshes the locks after `cargo update`, an edit of a
    `Cargo.toml` or of `MODULE.bazel`: `CARGO_BAZEL_REPIN=1 bazel mod deps --lockfile_mode=update`. Only an edit of
    `MODULE.bazel` itself fails a local Bazel command (`--lockfile_mode=error` on `MODULE.bazel.lock`).

  Its failure message names the stale files and the command that fixes them. Regenerating is the duty of whoever
  changes a `Cargo.toml` or adds a target file. The root `README.md` says so in one line, and the stale test enforces
  it.
- **The pin check** (toolchain, below) is a second test of the same crate, so it also runs under both tools. Its
  `data` is `rust-toolchain.toml`, `mise.toml` and `MODULE.bazel`.
- **Line endings:** `.gitattributes` sets `BUILD.bazel`, `*.bzl`, `MODULE.bazel`, `MODULE.bazel.lock`, `cargo-bazel-lock.json`, `Cargo.toml`
  and `Cargo.lock` to `eol=lf`. Otherwise a Windows checkout with CRLF fails the byte comparison of the stale check and
  the locks' digests.
- **The feature resolver** (below) is the generator's own, for workspace crates only. It is proved once per world in
  the generator's pull request, with the command and the output in its description, and the comparison is then
  deleted:
  - the testing world against `cargo metadata`'s full resolve;
  - the production world against `cargo tree -e normal,features` per crate. `cargo metadata`'s resolve folds in
    dev-dependency features, so it cannot prove the production world.

Rejected:

- **Phase 3's hand-written globs with `all_crate_deps` and a layout check:** they cannot pass dev-dependencies, the
  test features of self dev-dependencies, or optional dependencies, and the layout check would refuse most of the
  workspace. Writing ten crates and two feature worlds by hand makes about 250 targets that drift from `Cargo.toml`.
- **`gazelle_rust`:** it infers dependencies from `use` paths in the sources. RustEE's dependencies, features and
  targets are already exact in `Cargo.toml`, and inferring them from sources knows nothing of features,
  `required-features` or self dev-dependencies, which are the hard part here. It would also be one more Rust plugin to
  build and pin outside the C4 table, and an unproven match with rules_rust 0.75.0 and Gazelle 0.54.0. It was not
  fetched to test, because nothing it does would replace the feature logic.
- **A Starlark macro that reads `Cargo.toml`:** Starlark cannot read a file at loading time, so the macro would need a
  repository rule or module extension that parses TOML. That is the same generator, written in a weaker language and
  run on every fetch, and its output would be invisible in review.
- **rules_rust's own tooling:** crate_universe (`from_cargo`, `crates_vendor`) generates BUILD files for third-party
  crates and offers only `all_crate_deps` and `crate_deps` for workspace members. rules_rust 0.75.0 ships no generator
  for first-party targets. Its `extract_cargo_lints` reads `[lints]`, but it builds a Rust tool and still emits no
  `check-cfg` for features, so the generator writes the `rust_lint_config` itself.
- **A Python generator (the prototype's form):** it would need rules_python and its hermetic interpreter in a Rust
  repository, and new tools here are Rust by default. A Rust crate is built and tested by both tools with no second
  toolchain.
- **Putting the generator in forge now:** only RustEE needs it, and its tests need a real workspace. It moves when a
  second workspace uses it.

### Features: three worlds, every test in the test world

A *world* is one consistent choice of features for every workspace crate. Within a test's dependency graph every crate
must come from the same world. Otherwise `euhedral-gpu`'s tests would link two different `euhedral_graph` crates (one
with `testing`, one without), and their types would not match.

| world | features | targets | tag |
|---|---|---|---|
| production | each crate's defaults, no dev-dependencies (what a consumer of the crate gets) | `//<crate>:<crate>` | none |
| testing | `cargo test --workspace`: defaults plus every member's dev-dependency features | `<crate>_testing` and every test | none |
| cuda testing | testing plus `cuda` on `euhedral-memory` and `euhedral-gpu` | `<crate>_cuda_testing`, their tests, the CUDA examples | `gpu`, `external` on every test that resolves `euhedral-memory/cuda` (What runs where) |

- A crate whose features and first-party dependencies are the same in two worlds gets one target, and the other
  world's name is an `alias`. Today the testing world has its own build of `euhedral-graph`, `euhedral-gpu`,
  `euhedral-memory` and `euhedral-net`. The cuda world has its own build of `euhedral-memory` and `euhedral-gpu`.
- **Defaults** are always on, so `euhedral-io` is never built empty.
- **Self dev-dependencies:** in the testing world a crate's own library already carries its test features (`fake`,
  `test-hooks`, `testing`), and its tests link it.
- **`required-features`:** an example or bin is generated in the first world that has its required features. The CUDA
  examples land in the cuda world, so `bazel build //...` compiles them as `cargo test --features cuda` does.
- **Never fewer features than Cargo:** the testing world is the union `cargo test --workspace` builds, so every crate
  has at least the features that `cargo test -p <crate>` gives it, and sometimes more:
  - `cargo test -p euhedral-net` and `cargo test -p euhedral-memory --features cuda` build `euhedral-graph` without
    `testing`;
  - under Bazel, the same tests link `euhedral-graph` with `testing`.

  The Bazel test set is therefore the workspace run's, not each per-crate run's. Feature minimality stays checked by
  Cargo: workspace clippy and the per-crate `cargo test -p` runs, plus `cargo clippy --workspace --lib` for the
  production world (step 2b).
- **`--features cuda` as the gate uses it:** the cuda world is built by every `bazel build //...` and `bazel test //...`
  (a compile check on Linux and Windows), and its tests run only under `--config=gpu`. The gate's CUDA host-side step
  in CI (no card, every device test skips) stays on Cargo. Under Bazel those host-side tests run with the device tests
  under a grant.
- **loom:** a `cfg`, not a feature, set with `RUSTFLAGS` for every crate. It stays on Cargo (see below).
- **No build setting and no transitions:** worlds are named targets. A consumer (C5, phase 5) picks
  `//euhedral-gpu:euhedral_gpu`, or later a generated `euhedral_gpu_cuda` production target, and never gets test hooks.

Rejected:

- **One world, the testing union, for everything:** simplest, but every library would carry `fake` and `test-hooks`,
  and a later consumer of the module would get test switches in production.
- **A `bool_flag` for `cuda` with `select` in `crate_features`:** then `//...` never compiles the CUDA code unless the
  flag is set, and the Windows job would not build it.
- **A configuration transition per test:** it needs a custom wrapper rule (or a rule set outside the C4 table) and
  gives the same result as named targets, with less visible output.

### Toolchain: add `rust-toolchain.toml`, check the pins

C10 says rules_rust follows `rust-toolchain.toml`. RustEE adds one at the root:

```toml
[toolchain]
channel = "1.99.1"
profile = "default"
```

The version is the one `mise.toml` pins today, and `profile = "default"` matches `mise.toml` (rustfmt and clippy
included).

- rules_rust 0.75.0 takes only a literal version, so `MODULE.bazel` repeats it twice: `rust.toolchain(versions = [...])`
  and `rust.host_tools(version = ...)`. The second is the cargo crate_universe splices with. It defaults to rules_rust's
  own version, which is the same today but moves when rules_rust is bumped.
- A check fails when any of `rust-toolchain.toml`, `mise.toml` and the two `MODULE.bazel` lines differ. It is a test of
  the generator crate, so it runs in `bazel test //...` and in `cargo test`. Until the Bazel files exist, it compares
  the first two.
- **Miri** keeps its dated nightly in `scripts/ci/miri-toolchain`. The Miri jobs select it with `RUSTUP_TOOLCHAIN`,
  which outranks `rust-toolchain.toml`, so they are unchanged.
- **rust-analyzer** reads `Cargo.toml` and `rust-toolchain.toml` through Cargo, as phase 3 decided. There is no
  `rust-project.json`.

Rejected:

- **Documenting an exception to C10:** keeping `mise.toml` as the only source would make RustEE the one Rust repository
  whose IDE toolchain and CI toolchain come from different files, and it gains nothing. The file costs one line in each
  of the two workflows' "root build file" lists. `scripts/ci/affected.py` already treats `rust-toolchain.toml` as a
  root build file, so CI needs no change to see it.

### What runs where

| what | Bazel | Cargo (unchanged) | why |
|---|---|---|---|
| unit, integration, bin and doc tests (debug) | `bazel test //...` | `cargo test` | the parity this design proves |
| CUDA examples, cuda-world compile | `bazel build //...` | `cargo clippy --features cuda` | compile check on both OSes |
| device tests (`gpu`) | `bazel test --config=gpu //...` under `system-run --gpu` | the device job | the card, under a grant |
| benchmarks' smoke tests and bins' unit tests | `bazel test //...` | `cargo test` | they are ordinary tests |
| benchmark runs, timing tests | not yet (see below) | by hand | C7 `exclusive` |
| release stress tests | no | `cargo test --release ... -- <filters>` | several name filters per binary; see below |
| loom | no | `RUSTFLAGS=--cfg loom cargo test --release` | see below |
| Miri | no | `cargo miri` shards | rules_rust has no Miri support; nightly toolchain |
| fmt, clippy | no | `cargo fmt`, `cargo clippy -D warnings` | the gate's lint stays one tool |
| Windows | `bazel test //... --config=ci` on `windows_x86_64` | GitHub Windows job | C8 |

- **GPU (C7):** the generator tags `gpu` and `external` every test whose resolved features include
  `euhedral-memory/cuda`, the feature that loads the driver (step 4). It decides this per test, from the features
  the test's graph resolves to, not from the world the test was generated in. So a test that reaches `cuda` through any
  path is tagged.
  - The generator refuses to write when the testing world resolves `euhedral-memory/cuda`. That would happen if a
    member's dev-dependency asked for it (say `euhedral-memory = { ..., features = ["cuda"] }`): Cargo would then
    merge the cuda world into the testing world, and untagged tests would open the card. The testing world is proved
    equal to `cargo metadata`'s resolve, where the same fact shows as `cuda` in `euhedral-memory`'s `features`.
  - `external` keeps Bazel from caching the result in any configuration: a device result depends on the card and the
    driver, which are not inputs.
  - `--config=gpu` runs both crates' unit, integration and doc tests with `cuda`, with the testing world's features.
    That is a superset, in features, of the device job's `cargo test -p <crate> --features cuda` (see Features). The
    cuda library targets are never `bazel test`ed without the grant.
  - The repository's `.bazelrc` passes the CUDA lookup environment through:
    `test:gpu --test_env=EUHEDRAL_NVRTC --test_env=CUDA_HOME --test_env=CUDA_PATH --test_env=LD_LIBRARY_PATH`, plus
    `--test_arg=--nocapture`.
  - **Skips fail the run:**
    - A device test that finds no driver, card or NVRTC returns early and reports `ok`.
    - Every such early return, and every half of a test that a capability check leaves out, goes through a helper
      (after step 5 fixes three tests in `euhedral-gpu/tests/cuda.rs` that skip without one):
      - `skip` (`euhedral-gpu/tests/common/skip.rs` and `euhedral-memory/tests/cuda.rs`) prints
        `DEVICE-TEST-SKIPPED:` when a driver, card or NVRTC is missing;
      - `skip_hardware` (`euhedral-gpu/tests/common/skip.rs`) prints `DEVICE-TEST-SKIPPED-HARDWARE:` when the card
        or driver lacks the feature under test. Two of step 5's three fixes go through it.

      The unit tests under `src/cuda` have no silent skip.
    - `test:gpu --test_env=EUHEDRAL_DEVICE_REQUIRED=1` makes both `skip` helpers panic instead, so the test fails
      where it skipped. `skip_hardware` stays a listed, non-failing skip.
    - Under Bazel this replaces `scripts/ci/device_check.py`, which reads Cargo's output. Its other check, that every
      `tests/cuda*.rs` binary ran, holds by construction, since every test target is generated. Cargo's device job
      keeps `device_check.py` unchanged.
  - Whether the Linux sandbox exposes `/dev/nvidia*` is checked first. The fallback is adding `no-sandbox` to those
    targets' tags.
- **No CUDA repository rule:** nothing in RustEE compiles or links against the toolkit, so the build stays hermetic and
  cacheable. Only the device tests touch the machine's CUDA, at run time, and they are uncached (above).
- **Windows (C8):** no target needs `target_compatible_with`. Every platform difference in RustEE is a `cfg` inside
  shared crates (io_uring, IoRing, `libc`, `windows-sys`), and the examples' Unix parts are `cfg(unix)` modules. The
  generator emits none, so every target is built on Windows.
  - The two source-reading tests need a runfiles tree, which Bazel does not create on Windows by default, so the
    repository's `.bazelrc` sets `common:windows --enable_runfiles`. That line only applies with
    `common --enable_platform_specific_config`, which is off by default in Bazel 9.3.3, so the `.bazelrc` sets that
    too.
  - crate_universe resolves for `x86_64-unknown-linux-gnu` and `x86_64-pc-windows-msvc` (`supported_platform_triples`).
- **Benchmarks (C7 `exclusive`):** the gate runs no benchmark today. The bins are built and their unit tests run. A
  later change may add `exclusive`-tagged targets for the throughput binaries and for the `credit_graph` timing test
  (with `env = {"EUHEDRAL_TEST_TIMING": "1"}`), run with `--config=system` under `system-run --system`. Nothing in this
  design needs them.
- **Release stress tests:** they are `cargo test --release` on chosen modules with several name filters per binary.
  Under Bazel that is a `-c opt` build of the testing world with per-target filter arguments. It is possible, but it
  would duplicate a Cargo step for no new coverage. It stays on Cargo, and can move later as `*_release_test` targets
  if the gate wants their results cached.
- **loom:** `--cfg loom` applies to every crate, third-party included, and swaps the libraries' atomics. Under Bazel it
  would be a `--config=loom` with `@rules_rust//rust/settings:extra_rustc_flag=--cfg=loom`, `-c opt` and a
  `config_setting` for the `cfg(loom)` dependencies. That is a fourth world, Linux-only, for a CI step that runs a few
  modules. It stays on Cargo.

### Counting: test names, per binary, once

The proof that `bazel test //...` runs what `cargo test` runs compares **names**, not totals, for every test binary.
Each lib, each integration file, each bin and each crate's doctests is one binary.

1. `cargo test --workspace --no-fail-fast -- --list --format=terse`. Each `Running` line starts a binary, and the
   lib's unit-test line names the crate. Then the same once per crate as the gate runs it: `cargo test -p <crate>`,
   and `-p euhedral-gpu --features cuda` and `-p euhedral-memory --features cuda` for the cuda world. A per-crate
   run can build a dependency with fewer features than the workspace run (Features). The per-crate comparison shows
   that no test exists only under those features.
2. `bazel test //... --test_arg=--list --test_arg=--format=terse`, then each `bazel-testlogs/<crate>/<target>/test.log`.
   `rust_doc_test` ignores `--list` and runs the doctests, which is harmless. Their names are read from its `test ...`
   lines and normalised to `file:line`.
   - For the cuda world, run each `gpu`-tagged binary from `bazel-bin` with `--list`, which opens no card. The cuda
     doctests are compared under the grant.
3. Compare the sets per (crate, binary). The prototype's result was 2307 = 2307 (testing world) and 376 = 376
   (cuda-world unit and integration tests), with every set equal. Per crate, testing world (Cargo = Bazel):

   | crate | tests |
   |---|---|
   | euhedral-benchmarks | 29 |
   | euhedral-core | 804 |
   | euhedral-data-structures | 331 |
   | euhedral-gpu | 208 |
   | euhedral-graph | 342 |
   | euhedral-hardware-utils | 171 |
   | euhedral-hashing | 43 |
   | euhedral-io | 193 |
   | euhedral-memory | 69 |
   | euhedral-net | 117 |
   | total | 2307 |

This runs once, names only, in the generator's pull request (command and table in its description), and the script is
then deleted. What keeps the counts equal afterwards is structural: every target comes from `cargo metadata`, every test
target repeats the world's features, and the stale test fails when the BUILD files no longer match `Cargo.toml`.
A per-crate count is not a test to keep (it would only restate the generator).

## Build steps

Each step is one pull request for one agent. The build starts after forge's rollover (phase 3) merges, and no step
merges before it.

1. **`rust-toolchain.toml` and the generator crate with the pin check.** This step adds:
   - the toolchain file (version from `mise.toml`);
   - `tools/bazel-gen` as a workspace member, with only its pin test for now. The test compares
     `rust-toolchain.toml` with `mise.toml`, and with the two `MODULE.bazel` lines once they exist;
   - its layer in `euhedral-gpu/tests/layering.rs` (`LAYERS`), since `every_member_has_a_layer` fails for a new
     member. It depends on no workspace crate and none depends on it, so it gets layer 0.

   `affected.py` already counts the toolchain file as a root build file. *Done:* the gate is green, the pin test fails
   on a mismatched pin, and the gate runs nothing else new. *Parallel with 2a and 2b.*
2. **(a) Tests write scratch files under `TEST_TMPDIR` when it is set.** The six sites:
   - `euhedral-io`: `src/uring.rs`, `src/win/live.rs`, `tests/key_frame.rs`, `tests/io_owned.rs`;
   - `euhedral-gpu`: `src/module_cache.rs`, `tests/cuda.rs`.

   Use one small helper per crate's tests. With the variable unset, behaviour under Cargo is unchanged. *Done:*
   `cargo test` of both crates green (cuda `--no-run`), the Windows job green. *Parallel with 1 and 2b.*
   
   **(b) The production world is lint-clean and guarded.** `euhedral-gpu` built with neither `fake` nor `cuda` has 15
   dead-code items. Gate the backend-only items on `any(feature = "fake", feature = "cuda")`. Then add
   `cargo clippy --workspace --lib -- -D warnings` to the clippy job of both workflows (twins). Bazel never fails a
   build on a warning, so this Cargo step is what guards the production world. *Done:* that step is in CI and green.
   *Parallel with 1 and 2a.*
3. **The Bazel skeleton.** Add:
   - `.bazelversion` (9.3.3) and `MODULE.bazel`: rules_rust, platforms and skylib at the table's
     versions, and no rules_python or rules_shell (no shell or Python target exists). It also holds:
     - `rust.toolchain` and `rust.host_tools` at the toolchain file's version;
     - `crate.from_cargo(name = "crates", manifests = ["//:Cargo.toml"], supported_platform_triples = [linux, windows])`;
     - the `euhedral_build` module, as forge does.
   - `.bazelrc` per C3, C7 and C8, plus `common --enable_platform_specific_config`, `--lockfile_mode=error`,
     `common:windows --enable_runfiles` and `try-import %workspace%/user.bazelrc`.
   - `.bazelignore` with `target` and `.claude`. Without `.claude`, `//...` descends into agent worktrees, even ones
     with their own `MODULE.bazel`.
   - In `.gitignore`: `/bazel-*` and `user.bazelrc`.
   - `.gitattributes` (LF for the Bazel files, `Cargo.toml` and `Cargo.lock`) `MODULE.bazel.lock` and `cargo-bazel-lock.json`. The pin test
     is extended to `MODULE.bazel`, and its failure message names `bazel mod deps`.
   - `scripts/ci/affected.py` and its tests: the root Bazel files (`MODULE.bazel`, `MODULE.bazel.lock`, `.bazelrc`,
     `.bazelversion`, `.bazelignore`, the root `BUILD.bazel`) affect everything (the full Cargo run and every Miri
     shard). A crate's `BUILD.bazel` affects that crate. Any member's `Cargo.toml`, and any change under a member's
     `src/`, `tests/`, `examples/` or `benches/` or to its `build.rs`, also affects `euhedral-bazel-gen` (its stale test). `--name-only`
     cannot tell an added file from an edited one, so every change there counts (see the stale check).

   *Done:* `bazel query @crates//...` resolves, forge's `check-repo.sh .` passes C1 to C4, and the affected-file tests
   cover the new files. *After 1.*
4. **The generator, the BUILD files and the stale check.** Add:
   - the generator in `tools/bazel-gen`, `//:gazelle`, the stale test with its lock check, and the generated
     `BUILD.bazel` of every crate in the three worlds. Every test whose resolved features include
     `euhedral-memory/cuda` is tagged `gpu` and `external` from the first generated file, and the generator refuses
     to write if the testing world resolves that feature, so `bazel test //...` never opens the card without a grant;
   - `[package.metadata.bazel]` for `hot_fix_ratchet` and `layering` and the `cpus` of the lattice tests, and the root
     `workspace_sources` filegroup;
   - in the root `BUILD.bazel`, `exports_files(["Cargo.toml", "Cargo.lock", "MODULE.bazel", "cargo-bazel-lock.json",
     "mise.toml", "rust-toolchain.toml"])`, so the stale and pin tests in `//tools/bazel-gen` can name those root
     files in their `data`;
   - the root README's line on regenerating.

   The pull request carries the one-time proofs:
   - the resolver against `cargo metadata` (testing world) and `cargo tree -e normal,features` (production world);
   - the name-by-name test count, workspace and per crate.

   *Done:*
   - `bazel build //...` and `bazel test //...` pass on Linux;
   - this query is empty:

     ```
     bazel query 'kind(test, rdeps(//..., attr(crate_features, "\bcuda\b", kind(rust_library, //...))))
       except (attr(tags, "\bgpu\b", //...) intersect attr(tags, "\bexternal\b", //...))'
     ```

     It selects every test that links any library built with `cuda` in its `crate_features` (the generator writes
     the feature there on every cuda-world target), whatever the target is named, and lists those lacking either
     tag. It does not rely on the refusal rule or on a target name. Listing only the tagged tests would not show an
     untagged one;
   - the counts are equal;
   - check-repo passes C6;
   - Cargo commands are unchanged.

   *After 2a, 2b and 3.*
5. **Device tests under Bazel.** Do the following:
   - check the sandbox's `/dev/nvidia*`;
   - add `test:gpu` (the CUDA `--test_env` lines, `EUHEDRAL_DEVICE_REQUIRED=1`, `--nocapture`);
   - make the two skip helpers panic when `EUHEDRAL_DEVICE_REQUIRED` is set;
   - route the silent early returns in `euhedral-gpu/tests/cuda.rs` through the helpers:
     - `missing_kernels_are_not_found` (line 494) and `arguments_that_do_not_fit_are_refused_before_launch` (line 727)
       return with no line when the driver is older than 13.1. That is `skip_hardware`.
     - `axpb_from_source_and_from_ptx_match_the_cpu_bit_for_bit` (line 570) drops its NVRTC half silently when NVRTC
       is missing (`skip`), and its image half silently when the driver is older than 13.1 (line 573,
       `skip_hardware`). It skips loudly only when both are missing.

     A dropped half returns nothing, so a search for `return` cannot find it. Silent skips are found by searching the
     capability checks instead: `driver_version()`, `Nvrtc::shared`, `arch()`, `CudaGpu::open`, `Device::open` and `Driver::load`/`Driver::shared`
     under `euhedral-gpu/tests` and `euhedral-memory/tests`, and reading each branch that leaves work out. That search,
     at the time of writing, finds only the four sites above. The other checks either pick one of two ways to load the
     kernels and skip loudly when both fail (`cuda_graph.rs`, `cuda_fault.rs`, `cuda_fault_drop.rs`,
     `cuda_launch_alloc.rs`, `cuda_lane_graph_alloc.rs`, `cuda_stage_alloc.rs`, `cuda_reopen.rs`,
     `common/cuda_support.rs`, `cuda.rs:768`), or already call a helper (`cuda.rs:449` and `:473`, the
     NVRTC-only tests, every device open, `cuda_programmatic.rs`, `cuda_tensor_map*.rs`, `euhedral-memory/tests/cuda.rs`;
     the bare `return` at `cuda.rs:701` follows the same check at line 691 and cannot be reached). Step 5
     repeats the search on its own head.

   *Done:* `system-run --gpu ... bazel test --config=gpu //...` runs every `gpu` target and none skipped, and a rerun
   is not served from the cache. *After 4; parallel with 6 and 7.*
6. **CI runs Bazel on Linux.** A `bazel` job in both workflows (twins kept by `test_workflow_twins.py`), under the
   arbiter, runs `bazel test //... --config=ci` (which includes the stale test) and `check-repo.sh`. *Done:*
   - the job is required in `ci-ok`;
   - a pull request that adds a test file without regenerating fails it;
   - the Cargo jobs are unchanged.

   As built: on Gitea the job runs on `arbiter-docker-bazel`, which gives it no token or secret and the commit at
   `/checkout` (copied in, since `actions/checkout` has no token), and it runs forge's C7 query itself, with
   `build:ci --local_resources=cpu=8` holding Bazel to the lease's threads. A Gitea-only `check-repo` job on
   `arbiter-docker` fetches forge's `main` with the job token (forge's repository is public, but the organisation's
   visibility is limited, so anonymous git is refused) and runs its `build/scripts/check-repo.sh --skip-c7 .`. A failed
   fetch fails the job, and no job runs a copy from this repository. The same job fails when forge's script no longer
   builds the C7 query the `bazel` job runs. Its steps that hold the token or run forge's code use only `/usr/bin:/bin`,
   since `/cache` comes first on the image's PATH and earlier pull requests' jobs can write it. GitHub's twin runs the
   same steps on `ubuntu-latest` (Bazelisk, checked against `.bazelversion`), except on a run limited to one OS. Both
   jobs run when `changes` reports any crate affected. Every Rust, Cargo, BUILD, MODULE, `.bazel*` and generator file
   does that, as does a file that a member's BUILD file names by label (the audit doc that `hot_fix_ratchet` reads).
   `test_workflow_twins.py` pins all of it.

   A pull request's `bazel` job uses the shared `ci-bazel` store, where an earlier pull request can have planted
   cached action and test results, so its green result is not a trust signal on its own. The enforced gate is the run
   on each push to `main`. The runner gives that run RustEE's own main stores (`ci-bazel-main-<id>`,
   `ci-cache-main-<id>`), which only jobs of protected refs write, that is merged code; the runner chooses the store
   from Gitea's task context, never from the workflow. Test results stay cached (`--cache_test_results` is not turned
   off): the main store is what makes them trustworthy there.

   *After 4; parallel with 5.*
7. **The Windows job runs Bazel.** The GitHub proxy runs `bazel test //... --config=ci` on
   `@euhedral_build//platforms:windows_x86_64`, with the shared `euhedral/ci` Windows workflow. *Done:* that job is
   green with every target built and no `target_compatible_with` added. *After 6 (both edit the twin workflows, so they
   go in order) and the shared workflow; parallel with 5.*
8. **Cache measurement.** Measure a warm rerun of `bazel test //...` and a run in a second worktree served from the
   shared disk cache. The numbers go in this document, and the step ends the build's acceptance criteria. *After 7.*

**Measured (2026-11-24, commit `5ecb1d8`, Bazel 9.3.3 under Bazelisk v1.28.1, this desktop, each run under a `system-run --system` grant).** Each run is `bazel test //...` over the 112 non-device tests, all passing; the times are Bazel's own `Elapsed time`. A wall clock around the call is not a cache number here: it was dominated by the wait for the whole-machine grant behind CI jobs. The second-worktree runs start from an empty output base (a new worktree, then `bazel clean --expunge` before the second and third run) and take every action from the shared disk cache.

| Run | `Elapsed time` (s), 3 runs | Actions | Tests |
|---|---|---|---|
| Warm rerun in the same worktree | 2.275, 2.009, 1.991 | none run | 0 of 112 executed, all cached |
| Second worktree, empty output base | 2.350, 2.442, 2.570 | 1286 processes: 508 disk cache hit, 890 internal | 0 of 112 executed, all cached |

Later, outside the build: `exclusive` benchmark and timing targets; release stress or loom under Bazel if the gate wants
them cached; a `<crate>_cuda` production target when a consumer module needs one (C5).

## Risks

- **The production world is checked by no gate run today.** Workspace clippy and `cargo test` unify dev-dependency
  features, so no gate run compiles a library with only its defaults. That is how `euhedral-gpu`'s dead code went
  unseen. `bazel build //...` compiles that world but never fails on a warning, so it is no guard. Step 2b's
  `cargo clippy --workspace --lib -- -D warnings` is the guard.
- **`exclusive` already means something in Bazel:** the built-in tag runs the test alone, with no other test in
  parallel. C7's meaning (needs the quiet machine) agrees with it, but the arbiter's `--system` grant, not the tag,
  is what keeps other jobs off the machine; the tag alone only serialises within one Bazel invocation.

- **Bazel runs test binaries in parallel; Cargo runs them one at a time.** The prototype's single run passed, including
  the lattice and `credit_graph` tests, but tests that size their thread counts from the CPUs the process may use now
  share the machine with each other. A test that fails only under Bazel is a test to fix (it assumes it has the machine
  to itself), not a reason to serialise the suite.
- **Generator drift from Cargo's resolver:** a feature form the resolver does not model could produce a world with
  fewer features. It stops on renamed dependencies, weak `?/` features, first-party dependencies under `[target.*]`,
  unknown target `cfg`s and `cuda` in the testing world, and the one-time proofs cover today's manifests. A new feature form is a generator change, proved
  again the same way.
- **Third-party features per world:** crate_universe builds each third-party crate once, with the features of the
  default resolve. A dependency feature that only the cuda world needs would be missing. Today the `windows-sys`
  features `euhedral-memory`'s `cuda` needs are already enabled by `euhedral-hardware-utils`. The cure, if needed, is
  `crate.annotation(crate_features = ...)`.
- **The device sandbox:** if the Linux sandbox hides `/dev/nvidia*`, the `gpu` targets need `no-sandbox`. That is
  step 5's first check.
- **Windows runfiles:** tests that read `data` at run time need `--enable_runfiles` (applied only with
  `--enable_platform_specific_config`) and symlink rights on the runner. They are found by the generator, not by hand:
  - every test with `data` (the `workspace_sources` key, today `hot_fix_ratchet` and `layering`);
  - every test that reads `compile_data` at run time rather than through `include_*!`. Step 4 lists them with a search
    for `CARGO_MANIFEST_DIR` and `fs::read` under `tests/`, and gives them the same key.

  If the runner lacks symlink rights, those tests read their files through the runfiles manifest instead, a
  test-only change.
- **The lock:** see "The locks" above: one repin command refreshes `cargo-bazel-lock.json` and `MODULE.bazel.lock`; the
  stale test and the `bazel` CI job's repin step fail on a stale one.

## Using RustEE from another Bazel module

A module depends on RustEE with `bazel_dep(name = "euhedral_rustee")` and a `git_override` (or `local_path_override`).
Three things follow from how Bazel loads a non-root module:

- `euhedral_build` is a `local_path_override` in RustEE's `MODULE.bazel` (container CI has no registry), and Bazel ignores
  overrides outside the root module. The consumer declares `bazel_dep(name = "euhedral_build", version = "0.1.0")` and its
  own override (`build/module` of this repository, or the registry once it publishes the module).
- The consumer's root needs a `BUILD.bazel`, even an empty one: crate_universe reads `//:MODULE.bazel` of the root and fails
  with "Unable to load package for //:MODULE.bazel" without it.
- Targets are named with underscores: `@euhedral_rustee//euhedral-core:euhedral_core`. The consumer's build resolves the
  third-party crates from this repository's `cargo-bazel-lock.json`; it does not run crate_universe's repin.
- rules_rust takes the Rust toolchain from the root module only: the consumer's build uses the toolchain its own
  `MODULE.bazel` registers, not RustEE's pin (1.99.1). A consumer that needs RustEE's compiler registers the same version.

## Proposed changes to forge's CONTRACTS.md

Text for the forge coordinator; RustEE does not edit forge.

- **C6, the Rust row:**

  > Rust: single crates without features or dev-dependencies use hand-written globs and `all_crate_deps` with a layout
  > check (phase 3). A workspace with features, dev-dependencies or examples generates its BUILD files with a
  > checked-in Rust generator crate driven by `cargo metadata --no-deps`. It is exposed as `//:gazelle`
  > (`bazel run //:gazelle` writes, `-- -mode=diff` fails on stale files). Its stale-check `rust_test` runs in
  > `bazel test //...` with the toolchain's cargo, and in `cargo test` (phase 4, RustEE). The generator moves to forge
  > once a second workspace uses it.

  The check line adds "or a stale-check test in `//...`" next to the Gazelle diff.
- **C3** (repository layout): `.bazelignore` lists `.claude` (agent worktrees), and `.gitignore` lists `/bazel-*`.
  Generated BUILD files are LF in `.gitattributes`, so a Windows checkout passes a byte-exact stale check.
- **C7:**

  > `gpu` tests are also tagged `external`, so no configuration caches a device result (it depends on the card and the
  > driver, which are not inputs). `test:gpu` passes the CUDA lookup environment (`CUDA_HOME`, `CUDA_PATH`,
  > `LD_LIBRARY_PATH` and the repo's own variables) with `--test_env`.

  The `gpu` row's "how it runs" adds:

  > a device test that skips for lack of a device must fail the run. RustEE does this with
  > `test:gpu --test_env=EUHEDRAL_DEVICE_REQUIRED=1`, which turns its skip helpers into panics.
- **C8:**

  > A repo whose tests read `data` as a directory tree sets `common:windows --enable_runfiles`, together with
  > `common --enable_platform_specific_config` (off by default in Bazel 9.3.3; without it no `:windows` line applies).
- **C10, the toolchain:**

  > rules_rust's toolchain follows `rust-toolchain.toml`; `MODULE.bazel` repeats the version in `rust.toolchain` and in
  > `rust.host_tools`, and a check test fails when they and `mise.toml` differ.
- **C10, CUDA** (replaces the CUDA line):

  > CUDA: a repo that compiles or links against the CUDA toolkit at build time uses the local toolkit through a
  > repository rule, its actions cached per machine (no repo does today). A repo that loads the driver and NVRTC at
  > run time (RustEE) needs no repository rule: its build is hermetic, and only its `gpu` tests touch the machine's
  > CUDA, uncached (C7).
- **Open questions, phase 4:**

  > answered: RustEE has no `build.rs` (the generator refuses `custom-build` targets until one appears, then emits
  > `cargo_build_script`); there is no CUDA repository rule (run-time loading); Windows runs `//...` with no
  > `target_compatible_with`. See RustEE `docs/design/bazel.md`.

## Corrections to the build's task text

The build task says RustEE has eleven crates and `build.rs` scripts. It has ten crates (`Cargo.toml`'s `members`) and
no build script. The CUDA task's "local-toolkit repository rule" has nothing to compile against: there is no
`build.rs`, no `#[link]`, and the NVRTC kernels have no `#include`. That plan and C10's exception are dropped. Its done
condition becomes "the cuda world builds under Bazel and its `gpu` tests pass under `system-run --gpu` with no skip".
