# euhedral-bazel-gen

Generates RustEE's Bazel build files from its Cargo manifests, so the two builds cannot drift apart
(`docs/design/bazel.md`). It is not published and no crate depends on it.

**Where it sits.** Layer 0: it depends on no workspace crate and none depends on it.

## Use

```text
bazel run //:gazelle                 # write every BUILD.bazel
bazel run //:gazelle -- -mode=diff   # write nothing; fail and name the stale files
cargo run -p euhedral-bazel-gen      # the same without Bazel (`-- -mode=diff` too)
```

Run it after adding, removing or renaming a crate, a target, a dependency or a feature, and commit the BUILD files. After
editing a `Cargo.toml` or `Cargo.lock`, also run `bazel mod deps --lockfile_mode=update` and commit `MODULE.bazel.lock`.
Never edit a generated file by hand, and add no BUILD file of your own inside the workspace: the stale test fails on
either.

## What it writes

It reads `cargo metadata --no-deps` and each member's `[lints]` (which the metadata does not carry), and writes one
`BUILD.bazel` per member and the root's. Each member's file has:

- `lints`, a `rust_lint_config` with the `[lints]` table and the cfg names Cargo declares (`feature` with the crate's
  features, `test`, `docsrs`, and the names of `unexpected_cfgs`'s `check-cfg`);
- `sources`, the manifest and target files, gathered at the root's `//:workspace_sources`;
- the library in each of the three feature worlds: `<crate>` (production: the crate's defaults, as a consumer gets it),
  `<crate>_testing` (`cargo test --workspace`'s features) and `<crate>_cuda_testing` (the testing world plus `cuda` on
  every member that has it). A world whose features and first-party dependency targets equal the previous world's is an
  `alias` of it;
- the tests in each test world that needs its own: `unit_test`, `doc_test`, `<test>_test` and `<bin>_bin_test`, with
  `_cuda` before `_test` in the cuda world (`unit_cuda_test`, `cuda_cuda_test`);
- the binaries in the first world with their required features, and the examples in the first test world with theirs.

**Tags.** A test whose graph links `euhedral-memory` with `cuda` opens the graphics card: it is tagged `gpu` and
`external`, so a plain `bazel test //...` skips it and `--config=gpu` runs it.

**Refusals.** The generator writes nothing and names the cause when:
- no member `euhedral-memory` declares `cuda` (the tags and the next check find the device backend by that name);
- the testing world resolves `euhedral-memory/cuda` (a dependency would make `cargo test --workspace` open the card);
- a member built alone would link another member with other features than the production world gives it (an empty
  `default` excepted, which turns on nothing);
- a manifest uses what it does not model: build scripts and build-dependencies, renamed dependencies, weak (`x?/f`) or
  third-party (`crate/f`) feature entries, first-party dependencies under `[target.*]`, target triples or unknown cfg
  names (`loom` is known, and false: Bazel never sets it), a `harness` key, a `[lib] name` other than the package's, a
  dotted key in a `[lints.<tool>]` table, target layouts other than Cargo's defaults, required features no world has, or two targets
  with one Bazel name.

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

None. The lock holds no JSON, TOML or SHA-256 crate, and adding `cargo_metadata`, `serde` or `toml` would bring a dozen
crates into every RustEE build for one tool. The crate carries a small JSON reader, the subset of TOML `[lints]` uses, and
SHA-256 for the lock check.

## Tests

- The unit tests cover the JSON reader, the cfg evaluator, the `[lints]` reader, SHA-256, the rule writer, the stale
  comparison and the lock check, and, on a small workspace (`src/fixture_tests.rs`), the feature worlds, the tags, the
  refusals and Windows paths.
- `tests/stale.rs` regenerates from the workspace and fails when a BUILD file differs, when a BUILD file on disk is
  one the generator does not write (outside what `.bazelignore` names; under Bazel it sees only the workspace sources),
  or when `MODULE.bazel.lock`'s record of a manifest is missing or out of date. It runs under Cargo and Bazel alike.
- `tests/toolchain_pin.rs` fails when the Rust version in `rust-toolchain.toml` differs from the ones in `mise.toml` and
  `MODULE.bazel`, or when the profile differs.
