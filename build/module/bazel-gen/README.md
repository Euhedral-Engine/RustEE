# euhedral-bazel-gen

Generates a Cargo workspace's Bazel build files from its Cargo manifests, so the two builds cannot drift apart
(build/CONTRACTS.md, C6). It ships in the `euhedral_build` module as `@euhedral_build//bazel-gen`; every Rust repo on the
forge runs the same generator, configured by its own `bazel_gen.toml`. Its tests live in forge's `build/bazel-gen/`.

## Adopting it

The repo depends on `euhedral_build` and `rules_rust` (C4) and has a `crate.from_cargo` hub over its workspace. Its first
root `BUILD.bazel` is these two lines; the first `bazel run //:gazelle` replaces it with the generated one:

```starlark
load("@euhedral_build//bazel-gen:defs.bzl", "bazel_gen")

bazel_gen(name = "gazelle")
```

The generated root file holds `//:gazelle` and `//:bazel_gen_stale_test`, so `bazel test //...` runs the stale check.

## Use

```text
bazel run //:gazelle                 # write every BUILD.bazel
bazel run //:gazelle -- -mode=diff   # write nothing; fail and name the stale files
bazel test //:bazel_gen_stale_test   # the same check as a test (part of //...)
```

Run it after adding, removing or renaming a crate, a target, a dependency or a feature, and commit the BUILD files. After
editing a `Cargo.toml` or `Cargo.lock`, also repin the crate hub and commit its lock; the stale test checks that lock
unless `lock_check = false`, in one of two modes:
- a hub without a `lockfile` (`crate.from_cargo(cargo_lockfile = ..., manifests = [...])`): `MODULE.bazel.lock` records
  each manifest's digest, and the test checks each recorded manifest and `Cargo.lock` against the workspace. Repin with
  `bazel mod deps --lockfile_mode=update`.
- a hub with `lockfile = "//:cargo-bazel-lock.json"` (needed when another module uses the repo: rules_rust refuses a
  non-root hub without one): `MODULE.bazel.lock` records no manifests, so set `[root] crate_lock` to that file; the test
  then checks that every package of `Cargo.lock` but the members, from any source, is a crate of the crate lock and the
  other way round. Repin with `CARGO_BAZEL_REPIN=1 bazel mod deps --lockfile_mode=update`.

Never edit a generated file by hand, and add no BUILD file of your own inside the workspace: the stale test fails on
either.

Arguments, each also read from the environment: `-mode=fix|diff` (`BAZEL_GEN_MODE`), `-config=PATH`
(`BAZEL_GEN_CONFIG`, instead of the root's `bazel_gen.toml`), `-lock=PATH` (`BAZEL_GEN_LOCK`, also check that lock).

## `bazel_gen.toml`

Optional; every key has a default, and an unknown key or a value of the wrong type is refused. The file is not called
`bazel-gen.toml` because Bazel leaves the root's `bazel-*` entries out of the execroot, so the stale test could not read
it. The examples are in forge's `build/bazel-gen/examples/`.

```toml
header = "# @generated ..."    # the first line of every file
crates = "crates"              # the crate_universe hub of third-party crates (`@crates//:name`)

[root]
exports = ["Cargo.lock", "Cargo.toml", "MODULE.bazel", "MODULE.bazel.lock"]  # exports_files; add rust-toolchain.toml etc.
gazelle = true                 # emit //:gazelle and //:bazel_gen_stale_test
lock_check = true              # the stale test also checks the crate hub's lock (see Use)
crate_lock = "cargo-bazel-lock.json"  # the hub's `lockfile`, if it has one: check it instead of MODULE.bazel.lock
filegroups = { testdata = ["testdata/**"] }  # root filegroups, for tests that name them

[device]                       # the graphics-card feature(s): tags derived from them; a device world only with member or [external.crates]
member = "example-memory"     # the member whose feature opens the card (the producer); a consumer of it has none
external = ["example-memory"] # external crates (checked against [external.crates]); kept for existing configurations
feature = "cuda"
also = ["vulkan"]              # more device features; a crate resolved with any of them opens the card

[external.crates]              # git dependencies built by another Bazel module
example-core = "@euhedral_example//example-core"   # -> @euhedral_example//example-core:example_core<suffix>
example-gpu = { package = "@euhedral_example//example-gpu", worlds = { device = "_testing" } }  # per crate

[external.worlds]              # the label suffix of each world's build of an external crate
production = ""
testing = "_testing"
device = "_cuda_testing"

[packages.my-crate]
compile_data = ["kernels/**"]              # globs for every target's compile_data (include_str! and the like)
test_data = ["//:testdata", "Cargo.toml"]  # labels for the tests' data and compile_data; sets CARGO_MANIFEST_DIR
build_script_env = { CUDA_HOME = "/opt/cuda" }   # the build script's `build_script_env` (what it reads from the machine)
build_script_data = ["kernels/**"]         # globs of package files the build script reads: its `data`
build_script_tags = ["external", "gpu"]    # tags of the build script target only (a package without a build.rs refuses all three)
card_tests = false                         # true: this package's tests (and its dependents') open the card; false: its own never do
```

A renamed git dependency (`k = { workspace = true, features = [...] }` over a workspace `k = { package = "p", git = ... }`)
picks its world through its own entry: an `[external.crates]` entry named by the key (`k = { package = "@m//p", worlds =
{ ... } }`), else the package's entry. That is how one package appears in two feature sets: the package's entry and the
key's entry give two suffixes. The target is always the package's crate name, never the key's.

**The graphics card.** `[device] feature` (and `also`) name the features that open it. The generator derives which tests do
and tags them; the configuration names no crate. A test, binary test or doc test opens the card, in every world, when its
resolved graph links a crate that is resolved with a device feature on: any member (its own `cuda`, a dependency's request,
a `default` that includes it), a crate of another module, or a third-party crate:

```toml
[device]
feature = "cuda"      # nothing else: a test that links any crate resolved with `cuda` is tagged gpu and external
```

**Third-party crates need the crate hub's lock.** When `[device]` is declared and any member depends on a third-party
crate that `[external.crates]` does not map, `[root] crate_lock` must name crate_universe's own lock, or the generator
refuses: Cargo.toml of a member says only what a member asks of a crate, not which features Cargo unified onto it
(defaults, features that enable features) or which crates it links, so card tests would be left untagged. The recipe:
1. in `MODULE.bazel`, `crate.from_cargo(..., lockfile = "//:cargo-bazel-lock.json")`;
2. `CARGO_BAZEL_REPIN=1 bazel mod deps --lockfile_mode=update` writes the file (commit it; repin after every `Cargo.toml` or
   `Cargo.lock` edit, as in Use);
3. `[root] crate_lock = "cargo-bazel-lock.json"` in `bazel_gen.toml`, and `lock = "cargo-bazel-lock.json"` in the root's
   `bazel_gen_stale_test` (the generator writes the call from `bazel_gen.toml`, but name the file in `data`/`lock` of any
   hand-written one): the stale test reads it, and a missing file is an error naming it.

The generator then reads each crate's resolved features (defaults and enabled features included) and the crates it links, so
a test that links A is tagged when A links B that has the feature, together with the members' own requests. A workspace with
`[device]` and no third-party dependencies needs no lock. Over-tagging is the safe error.

With neither `member` nor `[external.crates]` there is no device world to build; with them the device world adds the
features to every member that declares them. The testing world refuses the device member, or a crate of another module
(its `_testing` label has no device build), resolved with a device feature. A declared feature that nothing resolves is
refused (stale).

Two per-package overrides, for what the generator cannot see; both name a workspace member:
- `[packages.p] card_tests = true`: the package's tests, and those of every package whose graph links it, open the card
  (a crate that loads the driver directly and declares no device feature).
- `[packages.p] card_tests = false`: the package's own tests never do, though they link a crate resolved with a device
  feature (a package that only names the type). It undoes only the tags derived from such crates: the device member's and
  the device world's tags stay. It is an unchecked declaration: nothing stops such a test from opening the card, and a test
  that skips when it finds no card passes. A `false` on a package nothing tags is refused.

A consumer chooses the feature world of an external crate it takes with `default-features = false` and its own features
by the suffix: the production world links `<label>`, the testing world `<label>_testing`, and the device world
`<label>_cuda_testing` (or the crate's own `worlds`). A member's `x/f` on an external crate is recorded; on any other
third-party crate it is refused.

## What it writes

It reads `cargo metadata --no-deps` and each member's `[lints]` (or, with `[lints] workspace = true`, the root's
`[workspace.lints]`), which the metadata does not carry, and writes one `BUILD.bazel` per member and the root's. Each
member's file has:

- `lints`, a `rust_lint_config` with the lint tables and the cfg names Cargo declares (`feature` with the crate's
  features, `test`, `docsrs`, and the names of `unexpected_cfgs`'s `check-cfg`);
- `sources`, the manifest and target files, gathered at the root's `//:workspace_sources`;
- the library in each feature world: `<crate>` (production: the crate's defaults, as a consumer gets it),
  `<crate>_testing` (`cargo test --workspace`'s features) and, with `[device]`, `<crate>_<feature>_testing` (the
  testing world plus the device feature on every member that has it). A world whose features and dependency targets
  equal the previous world's is an `alias` of it;
- the tests in each test world that needs its own: `unit_test`, `doc_test`, `<test>_test` and `<bin>_bin_test`, with
  `_<feature>` before `_test` in the device world (`unit_cuda_test`);
- the binaries in the first world with their required features, and the examples in the first test world with theirs.

**Renamed dependencies.** A dependency whose key differs from its package (`package = "..."`, in any of the three
tables, through `workspace = true` too) gets rules_rust's `aliases`: the label of the real target mapped to the key with
`-` as `_`, as Cargo names the crate in code. It holds for members, crate_universe crates and mapped git dependencies.
`dep:k` and `k/f` in features refer to the key. A rename is not conditional on the OS (a `[target.*]` rename lists the alias
on every OS; the label is in `deps` only on its own).

**Build scripts.** A package with `build.rs` gets a `cargo_build_script` named `build_script` (`build_script_testing`,
`build_script_<feature>_testing`: an alias of an earlier world's unless features or build-dependencies differ), with
`srcs`, the package's features in that world as `crate_features`, its `[build-dependencies]` as `deps` and `aliases`
(mapped like normal ones, renames included), `links` from the manifest, and `build_script_env`, `data` and `tags` from
`bazel_gen.toml`. The package's library, binaries, tests and examples have it in `deps`, so `OUT_DIR` and the script's
directives reach them as rules_rust gives them. The script's inputs that Cargo cannot say (`kernels/**`, the variables it
reads) are `build_script_data` and `build_script_env`. Only a script at `build.rs` is modelled. The script's tags are
its own: the library and tests do not inherit them. Two are derived, not configured: `external` on every target that has
`gpu`, and `no-cache` with `no-remote-cache` on a script whose `build_script_env` has an absolute path as a value
(`CUDA_HOME = "/opt/cuda"`: the script reads a tree outside the workspace that the action's inputs do not name, so a cached
result could outlive a change to it). The script still runs, and still finds the tree.

**`#[path]` across packages.** A `#[path = "..."]` in a package's `src/`, `tests/` or `examples/` is resolved as rustc
does (against the file's directory, plus the module names inside an inline `mod name { }`, and the file's stem first
when it is not a crate root or `mod.rs`), and so is one inside a file it names. A file of another member, or of no member
(it belongs to the nearest Bazel package above it, the root one when no BUILD file sits between), is listed by label in
the `srcs` of every target of the consuming package, and its owning package's generated BUILD file gets an `exports_files`
for exactly it, visible to the consumers and to the root (the stale test reads it). The generator writes no BUILD file in
a non-member directory. A path that leaves the workspace, a file under a `.bazelignore` entry (narrow `.bazelignore`:
list the other subdirectories instead of the directory) and a file in a Bazel package the generator does not write are
refused, naming the file. Every `include_str!`, `include_bytes!` and `include!` of a string literal (or of `concat!(env!("CARGO_MANIFEST_DIR"), "<literal>")`) in a member's `src/`, `tests/`, `examples/`, `benches/` or `build.rs` is resolved the same way, from the including file's directory: a file of the same package outside the includer's directory is a plain `compile_data` entry (`srcs` for `include!` of a source), one of another package or of the root is exported and listed by label, and a missing, ignored or outside file is refused naming the includer. A non-literal argument (`concat!(env!("OUT_DIR"), ...)`, a macro) is not read: name its files with the package's `compile_data` key. A `src/` file that `tests/` or `examples/` name is listed in their `srcs`. Only an attribute that starts a line is read.

**Binaries in integration tests.** A package with `[[bin]]` targets gives each integration test (`tests/*`) `data = [":<bin>"]`
and `rustc_env` `CARGO_BIN_EXE_<bin>` = `$(rootpath :<bin>)`, so `env!("CARGO_BIN_EXE_<bin>")` is the binary's path
relative to the runfiles root, where a test starts. It is relative, not absolute as under Cargo: a test that changes
directory or symlinks to it must make the path absolute first.

**Tests and the stale test.** Under Bazel a test runs with `CARGO_TARGET_TMPDIR=unused-under-bazel`: it must read
`TEST_TMPDIR` at run time, not `env!("CARGO_TARGET_TMPDIR")`. `bazel_gen_stale_test` always carries the tag `stale-build`
(set in `defs.bzl`), which the ci repo's bazel gate selects with `stale-build: tested`.

**Tags.** A test whose graph links a crate resolved with a device feature (any world), a package marked
`card_tests = true`, or (in the device world) any crate of
`[external.crates]` (its device build may open the card), opens the graphics card: it is tagged `gpu` and `external`, so a plain `bazel test //...` skips it
and `--config=gpu` runs it.

**Refusals.** The generator writes nothing and names the cause when:
- `device.member` does not declare the device feature;
- `[device]` is declared, a member depends on a third-party crate, and `[root] crate_lock` is not set;
- `[device]` declares a feature that no member or dependency is resolved with in the last world built (stale), or the
  testing world resolves a device feature for the device member or a crate of another module;
- `card_tests` on a name that is not a member, or `card_tests = false` on a package nothing would tag;
- a member built alone would link another member with other features than the production world gives it (an empty
  `default` excepted, which turns on nothing);
- a manifest uses what it does not model: a build script other than `build.rs`, weak (`x?/f`) or
  unmapped third-party (`crate/f`) feature entries, first-party dependencies under `[target.*]`, target triples or
  unknown cfg names (`loom` is known, and false: Bazel never sets it), a `harness` key, a `[lib] name` other than the
  package's, a dotted key in a `[lints.<tool>]` table, `[lints] workspace = true` beside lint tables, target layouts
  other than Cargo's defaults, required features no world has, or two targets with one Bazel name.

Paths in the metadata may use either separator; nothing written contains an absolute path.

## `[package.metadata.bazel]`

Bazel-only settings of one test or binary, which Cargo ignores:

```toml
[package.metadata.bazel.test.<name>]   # or .bin.<name>
workspace_sources = true   # data: //:workspace_sources; CARGO_MANIFEST_DIR relative to the runfiles root
data = ["//:Cargo.lock"]   # more data labels (a root file is exported); also sets the relative CARGO_MANIFEST_DIR
cpus = 4                   # tags: cpu:4 (the test pins that many workers)
cargo = true               # the toolchain's cargo as data, its path in CARGO
root_alias = "gazelle"     # (bin) an alias of this binary in the root package
```

An unknown key, or a name that is not a target of the package, is refused.

## Dependencies

None. A JSON, TOML or SHA-256 crate would bring a dozen crates into every consumer's build for one tool. The crate
carries a small JSON reader, the subset of TOML that `[lints]` and `bazel_gen.toml` use, and SHA-256 for the lock check.

## Tests (forge's `//build/bazel-gen`)

- `unit_test` (renames, build scripts and cross-package `#[path]` files: `src/rename_build_tests.rs`): the JSON and TOML readers, the configuration, the cfg evaluator, the `[lints]` reader, SHA-256, the rule
  writer, the stale comparison and the lock check, and, on small workspaces (`src/fixture_tests.rs`), the feature worlds,
  the tags, the refusals, Windows paths, git dependency labels, package data, each `[package.metadata.bazel]` key and
  inherited lints.
- `build/tests/test-bazel-gen-bazel.sh` (in `forge-bazel`): a scratch workspace takes the module, writes its BUILD files
  with `bazel run //:gazelle`, and `bazel test //:bazel_gen_stale_test` passes, fails naming the file after a
  `Cargo.toml` edit, and passes after regenerating; with a crate hub, it passes on a repinned lock and fails naming the
  lock in each mode (a manifest edit against `MODULE.bazel.lock`; a crate the `cargo-bazel-lock.json` has and
  `Cargo.lock` lacks).
