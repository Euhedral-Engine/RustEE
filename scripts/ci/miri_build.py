#!/usr/bin/env python3
"""The commands that build what the Miri shards run, each once.

A shard command is `cargo miri test <selection> [-- <test filters>]`. Its build depends on the selection only (the
packages, features and test targets), so shards that select the same build share one `cargo miri test --no-run
<selection>`. The `miri-build` job runs these commands once and hands its target directory and Miri sysroot to every
shard, which then only runs tests.

Usage:
  miri_build.py --matrix JSON    the commands of the matrix `affected.py` published (`{"include": [{"commands": ...}]}`)
  miri_build.py --all            the commands of every shard in miri-shards.json (the cache-warming job)
  miri_build.py --check-names --matrix JSON
                                 runs each command's selection with `-- --list` (after the build) and fails if a name
                                 the command gives (`--exact` names, `--skip` names, filters) matches no test
Prints one command per line.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = ["cargo", "miri", "test"]


def build_command(command):
    """`cargo miri test --no-run <selection>` for a shard command."""
    words = shlex.split(command)
    if words[:3] != PREFIX:
        raise ValueError(f"not a `cargo miri test` command: {command}")
    selection = words[3:]
    if "--" in selection:
        selection = selection[: selection.index("--")]
    return shlex.join(PREFIX + ["--no-run"] + selection)


def build_commands(commands):
    """The build commands of `commands`, in order, without repeats."""
    seen = []
    for command in commands:
        built = build_command(command)
        if built not in seen:
            seen.append(built)
    return seen


def list_command(command):
    """`cargo miri test <selection> -- --list`: the tests the selection has."""
    words = shlex.split(command)
    if words[:3] != PREFIX:
        raise ValueError(f"not a `cargo miri test` command: {command}")
    selection = words[3:]
    if "--" in selection:
        selection = selection[: selection.index("--")]
    return PREFIX + selection + ["--", "--list", "--format", "terse"]


def listed_tests(command):
    """The names of the tests `command`'s selection lists (every test binary of the selection)."""
    env = {k: v for k, v in os.environ.items() if k != "MIRIFLAGS"}
    out = subprocess.run(list_command(command), env=env, check=True, capture_output=True, text=True).stdout
    return {line[: -len(": test")] for line in out.splitlines() if line.endswith(": test")}


def name_problems(command, tests):
    """What in `command`'s test arguments matches nothing in `tests` (a stale name): a list of messages.

    With `--exact` a name must be a test; without it a filter, and a `--skip`, must be a part of some test's name.
    """
    words = shlex.split(command)
    args = words[words.index("--") + 1:] if "--" in words else []
    exact = "--exact" in args
    names, skipped, i = [], [], 0
    while i < len(args):
        if args[i] == "--skip":
            skipped.append(args[i + 1])
            i += 2
            continue
        if not args[i].startswith("-"):
            names.append(args[i])
        i += 1
    matches = (lambda n: n in tests) if exact else (lambda n: any(n in t for t in tests))
    return [f"{kind} {n!r} matches no test in: {command}"
            for kind, group in (("name", names), ("--skip", skipped)) for n in group if not matches(n)]


def check_names(commands, lister=listed_tests):
    """Every message `name_problems` has for `commands`; a selection is listed once."""
    listed, problems = {}, []
    for command in commands:
        key = build_command(command)
        if key not in listed:
            listed[key] = lister(command)
        problems += name_problems(command, listed[key])
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--matrix")
    group.add_argument("--all", action="store_true")
    ap.add_argument("--check-names", action="store_true")
    ap.add_argument("--miri-shards", default=os.path.join(HERE, "miri-shards.json"))
    args = ap.parse_args(argv)
    if args.all:
        with open(args.miri_shards) as f:
            commands = [c for shard in json.load(f) for c in shard["commands"]]
    else:
        commands = [c for entry in json.loads(args.matrix)["include"] for c in entry["commands"].split("\n")]
    if args.check_names:
        problems = check_names(commands)
        print("\n".join(problems))
        return 1 if problems else 0
    print("\n".join(build_commands(commands)))


if __name__ == "__main__":
    sys.exit(main())
