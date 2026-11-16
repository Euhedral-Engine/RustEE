#!/usr/bin/env bash
# The Linux steps of the `workspace` job in .github/workflows/ci.yml, for a runner that is not GitHub's: the scripts'
# own tests, formatting, Clippy with warnings as errors, the tests and doctests of the affected crates, the release
# stress tests, the Loom models, the replay test and the CUDA backends' build (the tests skip with no device).
# Miri is not run here.
#
# Usage: scripts/ci/linux.sh [BASE_REV]
# With a base revision only the crates a change since its merge base affects are built; without one, everything.
set -euo pipefail
cd "$(dirname "$0")/../.."

step() { echo "linux: $*" >&2; }
cargo() { mise exec -- cargo "$@"; }

out="$(mktemp)"
trap 'rm -f "$out"' EXIT
step "scripts/ci/test_affected.py, test_miri_build.py"
python3 scripts/ci/test_affected.py
python3 scripts/ci/test_miri_build.py

if [ -n "${1:-}" ]; then
  mise exec -- python3 scripts/ci/affected.py --base "$1" --head HEAD --merge-base --github-output "$out" > /dev/null
else
  mise exec -- python3 scripts/ci/affected.py --all --github-output "$out" > /dev/null
fi
output() { sed -n "s/^$1=//p" "$out" | tail -n 1; }
AFFECTED="$(output args)"
if [ "$(output any)" != true ]; then
  step "no crate is affected: nothing to build"
  exit 0
fi
step "affected: $(output reason)"

step "cargo fmt --all --check"
cargo fmt --all --check
step "cargo clippy $AFFECTED --all-targets -- -D warnings"
# shellcheck disable=SC2086
cargo clippy $AFFECTED --all-targets -- -D warnings
step "cargo test $AFFECTED"
# shellcheck disable=SC2086
cargo test $AFFECTED

if [ "$(output euhedral_core)" = true ]; then
  step "stress tests in release (euhedral-core)"
  cargo test --release -p euhedral-core --lib -- \
    flow_control::vertex generics:: ingest:: flow_control::hot_source utils::stop_watch flow_control::edge \
    control_plane::cache flow_control::views
fi
if [ "$(output euhedral_graph)" = true ]; then
  step "stress tests in release (euhedral-graph)"
  cargo test --release -p euhedral-graph --lib -- sequencer completions::
  cargo test --release -p euhedral-graph --test lake
fi
if [ "$(output euhedral_data_structures)" = true ]; then
  step "loom models (euhedral-data-structures)"
  RUSTFLAGS="--cfg loom" cargo test -p euhedral-data-structures --release --lib -- sync::claim_cell::loom queues::chunk::loom
fi
if [ "$(output euhedral_graph)" = true ]; then
  step "loom models (euhedral-graph)"
  RUSTFLAGS="--cfg loom" cargo test -p euhedral-graph --release --lib -- sequencer::loom stage_graph::loom take_all::loom lanes::loom
fi
if [ "$(output euhedral_gpu)" = true ]; then
  step "loom models (euhedral-gpu)"
  RUSTFLAGS="--cfg loom" cargo test -p euhedral-gpu --release --lib -- ring::loom instances::loom handles::loom
  step "replay as recorded, release (euhedral-gpu)"
  cargo test -p euhedral-gpu --release --lib --features fake -- graph_cache::shared_tests::a_hit_replayed_as_recorded
fi
if [ "$(output euhedral_memory)" = true ]; then
  step "CUDA backend (euhedral-memory)"
  cargo clippy -p euhedral-memory --all-targets --features cuda -- -D warnings
  cargo test -p euhedral-memory --features cuda
fi
if [ "$(output euhedral_gpu)" = true ]; then
  step "CUDA backend (euhedral-gpu)"
  cargo clippy -p euhedral-gpu --all-targets --features cuda -- -D warnings
  cargo test -p euhedral-gpu --features cuda
fi
step "ok"
