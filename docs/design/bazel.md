# Building and testing RustEE with Bazel (design)

Date: 2026-11-24. Status: proposed, for review before the build starts.

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

- **Generator:** a Python script, about 300 lines, that reads `cargo metadata --no-deps` and writes one `BUILD.bazel`
  per crate. `bazel build //...` succeeds, with no warnings once two problems listed below were fixed. That is 1129
  actions, covering both feature worlds and the CUDA examples, with a peak of 5.5 GB.
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

`scripts/bazel/gen_build.py` reads `cargo metadata --no-deps --format-version 1` and writes every crate's `BUILD.bazel`.
The file starts with an `@generated` line, and nobody edits it by hand.

- **Targets come from Cargo, not from globs:** one `rust_library` per lib, one `rust_test` for the lib's unit tests, one
  `rust_doc_test`, one `rust_test` per `[[test]]` target, one `rust_binary` per bin (plus a `rust_test` for its unit
  tests) and per example. A target kind the generator does not handle (`custom-build`, `bench`, a renamed dependency, a
  target `cfg` it does not know) stops it with a message, so a new kind of target is a generator change, never a silent
  omission.
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
  Cargo ignores. Today that is one key: `test.<name>.workspace_sources = true` for `hot_fix_ratchet` and `layering`.
  It gives the test `data = ["//:workspace_sources"]` (one `sources` filegroup per crate, gathered at the root, because
  a glob cannot cross packages) and the package-relative `CARGO_MANIFEST_DIR`.
- **Integration tests** get `CARGO_TARGET_TMPDIR` set to a path that does not exist (Cargo sets it and the code reads it
  at compile time). Under Bazel the tests use `TEST_TMPDIR` instead (step 2).
- **`//:gazelle`** is a `py_binary` that runs the generator through the toolchain's cargo and writes into
  `BUILD_WORKSPACE_DIRECTORY`. `bazel run //:gazelle -- -mode=diff` prints the stale files and fails, as C6's command
  does. There are no Go or Python packages in RustEE for Gazelle itself to manage. The `scripts/ci` Python tests stay
  on CI's Python.
- **`//:build_files_test`** is the stale check inside `bazel test //...`. It is a `py_test` that runs
  `cargo metadata --no-deps --offline` with the toolchain's cargo (`@rules_rust//rust/toolchain:current_cargo_files`,
  `CARGO_HOME` under `TEST_TMPDIR`) and then the generator in check mode. Its `data` is every `Cargo.toml`, every
  `BUILD.bazel` and every file target discovery scans (`src/`, `tests/`, `examples/`, `benches/`, `build.rs`).
- **Python:** rules_python 2.5.1 (the C4 table's version) with its hermetic interpreter (C10), standard library only
  (`json`, `tomllib`).
- **The feature resolver** (below) is the generator's own, for workspace crates only. It is proved once against
  `cargo metadata`'s full resolve for each world. The command and the output go in the pull request, and the comparison
  script is deleted, as for any parity proof.

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

### Features: three worlds, every test in the test world

A *world* is one consistent choice of features for every workspace crate. Within a test's dependency graph every crate
must come from the same world. Otherwise `euhedral-gpu`'s tests would link two different `euhedral_graph` crates (one
with `testing`, one without), and their types would not match.

| world | features | targets | tag |
|---|---|---|---|
| production | each crate's defaults, no dev-dependencies (what a consumer of the crate gets) | `//<crate>:<crate>` | none |
| testing | `cargo test --workspace`: defaults plus every member's dev-dependency features | `<crate>_testing` and every test | none |
| cuda testing | testing plus `cuda` on `euhedral-memory` and `euhedral-gpu` | `<crate>_cuda_testing`, their tests, the CUDA examples | `gpu` |

- A crate whose features and first-party dependencies are the same in two worlds gets one target, and the other
  world's name is an `alias`. Today the testing world has its own build of `euhedral-graph`, `euhedral-gpu`,
  `euhedral-memory` and `euhedral-net`. The cuda world has its own build of `euhedral-memory` and `euhedral-gpu`.
- **Defaults** are always on, so `euhedral-io` is never built empty.
- **Self dev-dependencies:** in the testing world a crate's own library already carries its test features (`fake`,
  `test-hooks`, `testing`), and its tests link it.
- **`required-features`:** an example or bin is generated in the first world that has its required features. The CUDA
  examples land in the cuda world, so `bazel build //...` compiles them as `cargo test --features cuda` does.
- **Never fewer features than Cargo:** the testing world is the union `cargo test --workspace` builds, so every crate
  has at least the features that `cargo test -p <crate>` gives it, and sometimes more (`euhedral-net`'s tests see
  `euhedral-graph` with `testing`). Feature minimality stays checked by Cargo: workspace clippy and the per-crate
  `cargo test -p` runs.
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
- A check fails when any of `rust-toolchain.toml`, `mise.toml` and the two `MODULE.bazel` lines differ. It runs in
  `bazel test //...` (part of `//:build_files_test`, or a `toolchain_pins_test` beside it) and in Cargo's CI as a step
  that needs no Bazel. Until the Bazel files exist, it compares the first two.
- **Miri** keeps its dated nightly in `scripts/ci/miri-toolchain`. `RUSTUP_TOOLCHAIN` outranks the file, so the Miri
  jobs are unchanged.
- **rust-analyzer** reads `Cargo.toml` and `rust-toolchain.toml` through Cargo, as phase 3 decided. There is no
  `rust-project.json`.

Rejected:

- **Documenting an exception to C10:** keeping `mise.toml` as the only source would make RustEE the one Rust repository
  whose IDE toolchain and CI toolchain come from different files, and it gains nothing. The file costs one line in each
  of the two workflows' "root build file" lists, which already name `rust-toolchain`.

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

- **GPU (C7):** every cuda-world test is tagged `gpu`, so `--config=gpu` runs exactly the device job's set (both
  crates' unit, integration and doc tests with `cuda`). The cuda library targets are never `bazel test`ed without the
  grant.
  - The repository's `.bazelrc` adds `test:gpu --cache_test_results=no`: a device result depends on the card and the
    driver, which are not inputs.
  - It also adds `test:gpu --test_env=EUHEDRAL_NVRTC` (unset is fine: the search finds `/usr/local/cuda`) and
    `--test_arg=--nocapture`.
  - A device test that finds no card or NVRTC returns early and reports `ok`. `scripts/ci/device_check.py` gains a mode
    that reads `bazel-testlogs/**/test.log` and fails on such a skip, as it does for Cargo's output.
  - Whether the Linux sandbox exposes `/dev/nvidia*` is checked first. The fallback is `tags = ["gpu", "no-sandbox"]`
    on those targets.
- **No CUDA repository rule:** nothing in RustEE compiles or links against the toolkit, so the build stays hermetic and
  cacheable. Only the device tests touch the machine's CUDA, at run time, and they are uncached (above).
- **Windows (C8):** no target needs `target_compatible_with`. Every platform difference in RustEE is a `cfg` inside
  shared crates (io_uring, IoRing, `libc`, `windows-sys`), and the examples' Unix parts are `cfg(unix)` modules. The
  generator emits none, so every target is built on Windows.
  - The two source-reading tests need a runfiles tree, which Bazel does not create on Windows by default, so the
    repository's `.bazelrc` sets `common:windows --enable_runfiles`.
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

1. `cargo test --workspace --no-fail-fast -- --list --format=terse` (and the same with
   `-p euhedral-gpu -p euhedral-memory --features cuda` for the cuda world). Each `Running` line starts a binary, and
   the lib's unit-test line names the crate.
2. `bazel test //... --test_arg=--list --test_arg=--format=terse`, then each `bazel-testlogs/<crate>/<target>/test.log`.
   `rust_doc_test` ignores `--list` and runs the doctests, which is harmless. Their names are read from its `test ...`
   lines and normalised to `file:line`.
   - For the cuda world, run each `gpu`-tagged binary from `bazel-bin` with `--list`, which opens no card. The cuda
     doctests are compared under the grant.
3. Compare the sets per (crate, binary). The prototype's result was 2307 = 2307 (default world) and 376 = 376
   (cuda-world unit and integration tests), with every set equal.

This runs once, in the generator's pull request (command and table in its description), and the script is then
deleted. What keeps the counts equal afterwards is structural: every target comes from `cargo metadata`, every test
target repeats the world's features, and `//:build_files_test` fails when the BUILD files no longer match `Cargo.toml`.
A per-crate count is not a test to keep (it would only restate the generator).

## Build steps

Each step is one pull request for one agent. Steps 1 and 2 touch only Cargo-side files and could merge before forge's
phase 3. Whether they may is the user's decision, since the Bazel build itself waits for phase 3.

1. **`rust-toolchain.toml` and the pin check.** Add the file (version from `mise.toml`) and a check of the two (a
   Python test in `scripts/ci`, run by CI's script-test step). Add `rust-toolchain.toml` to both workflows' root-file
   lists if it is not matched yet. *Done:* the gate is green and unchanged in what it runs. *Parallel with 2.*
2. **Tests write scratch files under `TEST_TMPDIR` when it is set.** The six sites:
   - `euhedral-io`: `src/uring.rs`, `src/win/live.rs`, `tests/key_frame.rs`, `tests/io_owned.rs`;
   - `euhedral-gpu`: `src/module_cache.rs`, `tests/cuda.rs`.

   Use one small helper per crate's tests, as in forge's crate. With the variable unset, behaviour under Cargo is
   unchanged. *Done:* `cargo test` of both crates green (cuda `--no-run`), the Windows job green. *Parallel with 1.*
3. **The Bazel skeleton.** Add:
   - `.bazelversion` (9.3.3) and `MODULE.bazel`: rules_rust, rules_python, rules_shell, platforms and skylib at the
     table's versions; `rust.toolchain` and `rust.host_tools` at the toolchain file's version;
     `crate.from_cargo(name = "crates", manifests = ["//:Cargo.toml"], supported_platform_triples = [linux, windows])`;
     the `euhedral_build` module as forge does;
   - `.bazelrc` per C3, C7 and C8, plus `--lockfile_mode=error`, `test:gpu --cache_test_results=no`,
     `common:windows --enable_runfiles`, and `try-import %workspace%/user.bazelrc`;
   - `.bazelignore` (`target`), `user.bazelrc` in `.gitignore`, `MODULE.bazel.lock`, and the pin check extended to
     `MODULE.bazel`.

   *Done:* `bazel query @crates//...` resolves, and forge's `check-repo.sh .` passes C1 to C4. *After forge's phase 3;
   after 1.*
4. **The generator, the BUILD files and the stale check.** Add:
   - `scripts/bazel/gen_build.py`, `//:gazelle` and `//:build_files_test`, and the generated `BUILD.bazel` of every
     crate in the three worlds;
   - `[package.metadata.bazel]` for `hot_fix_ratchet` and `layering`, and the root `workspace_sources` filegroup.

   The pull request carries both one-time proofs: the feature resolver against `cargo metadata` for each world, and the
   name-by-name test count. *Done:*
   - `bazel build //...` is warning-free and `bazel test //...` passes on Linux;
   - the counts are equal;
   - check-repo passes C6;
   - Cargo commands are unchanged.

   *After 2 and 3.*
5. **Device tests under Bazel.** Do the following:
   - check the sandbox's `/dev/nvidia*`;
   - add `test:gpu` (`EUHEDRAL_NVRTC`, `--nocapture`);
   - extend `device_check.py` to read Bazel's test logs.

   *Done:* `system-run --gpu ... bazel test --config=gpu //...` runs every `gpu` target, and the check confirms no
   device test skipped. *After 4; parallel with 6 and 7.*
6. **The Windows job runs Bazel.** The GitHub proxy runs `bazel test //... --config=ci` on
   `@euhedral_build//platforms:windows_x86_64`, with the shared `euhedral/ci` Windows workflow. *Done:* that job is
   green with every target built and no `target_compatible_with` added. *After 4 and the shared workflow; parallel
   with 5 and 7.*
7. **CI runs Bazel on Linux.** A `bazel` job in both workflows, twins kept by `test_workflow_twins.py`, runs
   `bazel test //... --config=ci` (which includes the stale check) and `check-repo.sh`, under the arbiter. *Done:*
   - the job is required in `ci-ok`;
   - a pull request that adds a test file without regenerating fails it;
   - the Cargo jobs are unchanged.

   *After 4; parallel with 5 and 6.*
8. **Cache measurement.** Measure a warm rerun of `bazel test //...` and a run in a second worktree served from the
   shared disk cache. The numbers go in this document, and the step ends the build's acceptance criteria. *After 7.*

Later, outside the build: `exclusive` benchmark and timing targets; release stress or loom under Bazel if the gate wants
them cached; a `<crate>_cuda` production target when a consumer module needs one (C5).

## Risks

- **Bazel runs test binaries in parallel; Cargo runs them one at a time.** The prototype's single run passed, including
  the lattice and `credit_graph` tests, but tests that size their thread counts from the CPUs the process may use now
  share the machine with each other. A test that fails only under Bazel is a test to fix (it assumes it has the machine
  to itself), not a reason to serialise the suite.
- **Generator drift from Cargo's resolver:** a feature form the resolver does not model (weak `?/` features across
  crates, renamed dependencies) could produce a world with fewer features. It stops on renamed dependencies and unknown
  target `cfg`s, and the one-time proof covers today's manifests. A new feature form is a generator change, proved
  again the same way.
- **Third-party features per world:** crate_universe builds each third-party crate once, with the features of the
  default resolve. A dependency feature that only the cuda world needs would be missing. Today the `windows-sys`
  features `euhedral-memory`'s `cuda` needs are already enabled by `euhedral-hardware-utils`. The cure, if needed, is
  `crate.annotation(crate_features = ...)`.
- **The device sandbox:** if the Linux sandbox hides `/dev/nvidia*`, the `gpu` targets need `no-sandbox`. That is
  step 5's first check.
- **Windows runfiles:** the source-reading tests need `--enable_runfiles` (and symlink rights on the runner). If the
  runner lacks them, the two tests read their files through the runfiles manifest instead, a test-only change.
- **The lock:** `MODULE.bazel.lock` must be refreshed after every `Cargo.lock` change (`--lockfile_mode=error` fails
  the Bazel job until it is). That job's message must say so, as forge's does.

## Proposed changes to forge's CONTRACTS.md

Text for the forge coordinator; RustEE does not edit forge.

- **C6, the Rust row:**

  > Rust: single crates without features or dev-dependencies use hand-written globs and `all_crate_deps` with a layout
  > check (phase 3). A workspace with features, dev-dependencies or examples generates its BUILD files with a
  > checked-in generator driven by `cargo metadata --no-deps`, exposed as `//:gazelle` (`bazel run //:gazelle` writes,
  > `-- -mode=diff` fails on stale files). A stale-check test inside `bazel test //...` runs the toolchain's cargo in the
  > sandbox (phase 4, RustEE).

  The check line adds "or a stale-check test in `//...`" next to the Gazelle diff.
- **C7:**

  > `test:gpu` also sets `--cache_test_results=no`: a device result depends on the card and the driver, which are not
  > inputs.

  The `gpu` row's "how it runs" adds: "a device test that skips for lack of a device must fail the run (the repo's own
  check of the test logs)".
- **C8:**

  > A repo whose tests read `data` as a directory tree sets `common:windows --enable_runfiles`.
- **C10, the toolchain:**

  > rules_rust's toolchain follows `rust-toolchain.toml`; `MODULE.bazel` repeats the version in `rust.toolchain` and in
  > `rust.host_tools`, and a check test fails when they and `mise.toml` differ.
- **C10, CUDA** (replaces the CUDA line):

  > CUDA: a repo that compiles or links against the CUDA toolkit at build time uses the local toolkit through a
  > repository rule, the documented exception (actions cached per machine). A repo that loads the driver and NVRTC at
  > run time (RustEE) needs no repository rule: its build is hermetic, and only its `gpu` tests touch the machine's
  > CUDA, uncached (C7).
- **Open questions, phase 4:**

  > answered: RustEE has no `build.rs` (the generator refuses `custom-build` targets until one appears, then emits
  > `cargo_build_script`); there is no CUDA repository rule (run-time loading); Windows runs `//...` with no
  > `target_compatible_with`. See RustEE `docs/design/bazel.md`.

## Corrections to the build's task text

The build task says RustEE has eleven crates and `build.rs` scripts. It has ten crates (`Cargo.toml`'s `members`) and
no build script. The CUDA task's "local-toolkit repository rule" has nothing to compile against (above). Its done
condition becomes "the cuda world builds under Bazel and its `gpu` tests pass under `system-run --gpu` with no skip".
