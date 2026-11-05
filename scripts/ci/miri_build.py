#!/usr/bin/env python3
"""The commands that build what the Miri shards run, each once.

A shard command is `cargo miri test <selection> [-- <test filters>]`. Its build depends on the selection only (the
packages, features and test targets), so shards that select the same build share one `cargo miri test --no-run
<selection>`. The `miri-build` job runs these commands once and hands its target directory and Miri sysroot to every
shard, which then only runs tests.

Usage:
  miri_build.py --matrix JSON    the commands of the matrix `affected.py` published (`{"include": [{"commands": ...}]}`)
  miri_build.py --all            the commands of every shard in miri-shards.json (the cache-warming job)
Prints one command per line.
"""

import argparse
import json
import os
import shlex
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--matrix")
    group.add_argument("--all", action="store_true")
    ap.add_argument("--miri-shards", default=os.path.join(HERE, "miri-shards.json"))
    args = ap.parse_args(argv)
    if args.all:
        with open(args.miri_shards) as f:
            commands = [c for shard in json.load(f) for c in shard["commands"]]
    else:
        commands = [c for entry in json.loads(args.matrix)["include"] for c in entry["commands"].split("\n")]
    print("\n".join(build_commands(commands)))


if __name__ == "__main__":
    sys.exit(main())
