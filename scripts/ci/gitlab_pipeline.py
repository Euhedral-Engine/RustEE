#!/usr/bin/env python3
"""Writes the GitLab child pipeline for the crates a change affects (the Linux twin of .github/workflows/ci.yml).

Reads scripts/ci/affected.py's `key=value` output on stdin and writes JSON (valid YAML) to stdout: the `workspace`
job (fmt, Clippy, tests, release stress tests, loom, CUDA build), `miri-build` (Miri's sysroot and every selection
of tests, handed to the shards as an artifact) and one `miri` job per affected shard. Windows is not covered.
Keep the steps in step with ci.yml.
"""
import json
import shlex
import sys

out = dict(l.rstrip("\n").split("=", 1) for l in sys.stdin if "=" in l)
yes = lambda k: out.get(k) == "true"
# io_uring (euhedral-io) is blocked by docker's default seccomp profile.
DOCKER = {"DOCKER_ARGS": "--security-opt seccomp=unconfined"}
# What each kind of job asks the arbiter for: the cores and memory it uses, and an estimate of its run time in seconds
# (at most the runner's 600 s cap). A Miri test job asks for its own seeds' cores and its shard's memory (the matrix).
# Measured peaks: the workspace job (fmt, Clippy, every test, release stress, loom) about 4.5 GB and, with a warm build,
# 4 cores on average over 80 s; miri-build 1.3 GB and about 1 core over 15 s.
WORKSPACE = {"ARBITER_ARGS": "--threads 8 --mem 6G", "ARBITER_EST": "240", **DOCKER}
MIRI_BUILD = {"ARBITER_ARGS": "--threads 2 --mem 2G", "ARBITER_EST": "60", **DOCKER}
MIRI_TEST_EST = "240"
STABLE = ["mise install rust >/dev/null", 'eval "$(mise env -s bash)"', "rustc --version"]
MIRI_SETUP = [
    'toolchain=$(cat scripts/ci/miri-toolchain)',
    'rustup toolchain install "$toolchain" --profile minimal --component miri --component rust-src',
    'export RUSTUP_TOOLCHAIN="$toolchain"',
]
jobs = {}

if yes("any"):
    cmds = ["cargo fmt --all --check", f"cargo clippy {out['args']} --all-targets -- -D warnings", f"cargo test {out['args']}"]
    if yes("euhedral_core"):
        cmds += ["cargo test --release -p euhedral-core --lib -- flow_control::vertex generics:: ingest:: "
                 "flow_control::hot_source utils::stop_watch flow_control::edge control_plane::cache flow_control::views"]
    if yes("euhedral_graph"):
        cmds += ["cargo test --release -p euhedral-graph --lib -- sequencer completions::",
                 "cargo test --release -p euhedral-graph --test lake"]
    if yes("euhedral_data_structures"):
        cmds += ["RUSTFLAGS='--cfg loom' cargo test -p euhedral-data-structures --release --lib -- sync::claim_cell::loom queues::chunk::loom"]
    if yes("euhedral_graph"):
        cmds += ["RUSTFLAGS='--cfg loom' cargo test -p euhedral-graph --release --lib -- sequencer::loom stage_graph::loom take_all::loom lanes::loom"]
    if yes("euhedral_gpu"):
        cmds += ["RUSTFLAGS='--cfg loom' cargo test -p euhedral-gpu --release --lib -- ring::loom instances::loom handles::loom",
                 "cargo test -p euhedral-gpu --release --lib --features fake -- graph_cache::shared_tests::a_hit_replayed_as_recorded"]
    if yes("euhedral_memory"):
        cmds += ["cargo clippy -p euhedral-memory --all-targets --features cuda -- -D warnings",
                 "cargo test -p euhedral-memory --features cuda"]
    if yes("euhedral_gpu"):
        cmds += ["cargo clippy -p euhedral-gpu --all-targets --features cuda -- -D warnings", "cargo test -p euhedral-gpu --features cuda"]
    jobs["workspace"] = {"stage": "test", "variables": WORKSPACE, "script": STABLE + cmds}

if yes("miri"):
    matrix = json.loads(out["miri_matrix"])
    jobs["miri-build"] = {
        "stage": "build", "variables": MIRI_BUILD,
        "script": MIRI_SETUP + [
            "cargo miri setup",
            f"python3 scripts/ci/miri_build.py --matrix {shlex.quote(json.dumps(matrix, separators=(',', ':')))} > miri-build.sh",
            "bash -ex miri-build.sh",
            f"python3 scripts/ci/miri_build.py --check-names --matrix {shlex.quote(json.dumps(matrix, separators=(',', ':')))}",
            "git rev-parse HEAD > target/miri/built-from-commit",
            'tar --use-compress-program="zstd -T0 -3" -C / -cf miri-build.tar.zst "${CI_PROJECT_DIR#/}/target/miri" "${HOME#/}/.cache/miri"',
        ],
        "artifacts": {"paths": ["miri-build.tar.zst"], "expire_in": "1 day"},
    }
    for i, shard in enumerate(matrix["include"]):
        jobs[f"miri {i}: {shard['name']}"] = {
            "stage": "test", "needs": ["miri-build"],
            "variables": {"ARBITER_ARGS": f"--threads {shard['cpu']} --mem {shard['mem']}", "ARBITER_EST": MIRI_TEST_EST,
                          "MIRIFLAGS": shard["miriflags"], "COMMANDS": shard["commands"], **DOCKER},
            "script": MIRI_SETUP + [
                "git ls-files -z | xargs -0 touch -d 2000-01-01",
                'tar --use-compress-program="zstd -T0" -C / -xf miri-build.tar.zst',
                'built=$(cat target/miri/built-from-commit); test "$built" = "$(git rev-parse HEAD)" || { echo "build is of $built"; exit 1; }',
                # -e stops at the first failing command; -x logs each as it starts.
                'bash -ex -c "$COMMANDS"',
            ],
        }

if not jobs:  # a change that affects nothing: the pipeline still passes
    jobs["nothing-affected"] = {"stage": "test", "script": [f"echo {json.dumps(out.get('reason', 'no crate affected'))}"]}
# A child of a merge request pipeline is not created unless its own `workflow` lets it run: the parent's rules do not
# reach it (a merge request pipeline has no branch for the parent's `$CI_COMMIT_BRANCH` rule to match).
WORKFLOW = {"rules": [{"when": "always"}]}
json.dump({"workflow": WORKFLOW, "stages": ["build", "test"], "variables": {"GIT_DEPTH": "0"}, **jobs}, sys.stdout, indent=1)
