#!/usr/bin/env python3
"""Writes the GitLab child pipeline for the crates a change affects (the GitLab twin of .github/workflows/ci.yml).

Reads scripts/ci/affected.py's `key=value` output on stdin and writes JSON (valid YAML) to stdout: the workspace
checks (fmt, Clippy, tests, release stress tests, loom, CUDA build) and one Miri job per affected shard. Linux only.
"""
import json
import sys

out = dict(l.rstrip("\n").split("=", 1) for l in sys.stdin if "=" in l)
yes = lambda k: out.get(k) == "true"
HEAVY = {"ARBITER_ARGS": "--cpu shared:16 --mem 32G", "ARBITER_EST": "1800"}
pre = ['mise install rust >/dev/null', 'eval "$(mise env -s bash)"', 'rustc --version']
jobs = {}

if yes("any"):
    cmds = [f"cargo clippy {out['args']} --all-targets -- -D warnings", f"cargo test {out['args']}"]
    if yes("euhedral_core"):
        cmds += [f"cargo test --release -p euhedral-core --lib {f}" for f in (
            "flow_control::vertex", "generics::", "ingest::", "flow_control::hot_source", "utils::stop_watch",
            "flow_control::edge", "control_plane::cache", "flow_control::views")]
    if yes("euhedral_graph"):
        cmds += ["cargo test --release -p euhedral-graph --lib sequencer", "cargo test --release -p euhedral-graph --test lake",
                 "cargo test --release -p euhedral-graph --lib completions::"]
    if yes("euhedral_data_structures"):
        cmds += ["RUSTFLAGS='--cfg loom' cargo test -p euhedral-data-structures --release --lib sync::claim_cell::loom",
                 "RUSTFLAGS='--cfg loom' cargo test -p euhedral-data-structures --release --lib queues::chunk::loom"]
    if yes("euhedral_graph"):
        cmds += ["RUSTFLAGS='--cfg loom' cargo test -p euhedral-graph --release --lib sequencer::loom"]
    if yes("euhedral_memory"):
        cmds += ["cargo clippy -p euhedral-memory --all-targets --features cuda -- -D warnings"]
    if yes("euhedral_gpu"):
        cmds += ["cargo clippy -p euhedral-gpu --all-targets --features cuda -- -D warnings", "cargo test -p euhedral-gpu --features cuda"]
    jobs["workspace"] = {"stage": "test", "variables": HEAVY,
                         "script": pre + ["cargo fmt --all --check"] + cmds}

for i, shard in enumerate(json.loads(out.get("miri_matrix", '{"include":[]}'))["include"]):
    jobs[f"miri {i}: {shard['name']}"] = {
        "stage": "test", "variables": {**HEAVY, "MIRIFLAGS": "-Zmiri-many-seeds=0..16 -Zmiri-ignore-leaks"},
        "script": ['mise install rust >/dev/null', 'eval "$(mise env -s bash)"',
                   "rustup toolchain install nightly --profile minimal --component miri --component rust-src",
                   "cargo +nightly miri setup"] + list(shard["commands"])}

if not jobs:  # documentation-only change: nothing to run, but the pipeline must still pass
    jobs["nothing-affected"] = {"stage": "test", "script": [f"echo {out.get('reason', 'no crate affected')}"]}
json.dump({"stages": ["test"], "variables": {"GIT_DEPTH": "0"}, **jobs}, sys.stdout, indent=1)
