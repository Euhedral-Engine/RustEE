# euhedral-bazel-gen

The tool that will generate RustEE's Bazel build files from its Cargo manifests, so the two builds cannot drift apart.
It is not published and no crate depends on it. Today it holds only the toolchain pin test.

**Where it sits.** Layer 0: it depends on no workspace crate and none depends on it.

## Tests

`tests/toolchain_pin.rs` fails when the Rust version in `rust-toolchain.toml` differs from the one in `mise.toml`, or
when the profile differs. Once `MODULE.bazel` exists, the same test will compare its pinned Rust version too.
