"""The test tags of the contracts (build/CONTRACTS.md, C7). Use these names; no repo invents its own for the same meaning."""

# Needs a GPU lease: excluded from the default `bazel test //...`, run with --config=gpu under the arbiter.
GPU = "gpu"

# Needs the whole machine (a benchmark, a timing test): Bazel's own tag, runs alone; --config=system under system-run --system, never cached.
EXCLUSIVE = "exclusive"

# Never part of //...: run by naming the target.
MANUAL = "manual"

# No Windows tag: the Windows job builds and tests all of //... (C7, C8).
ALL = [GPU, EXCLUSIVE, MANUAL]
