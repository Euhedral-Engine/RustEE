#!/usr/bin/env python3
"""Which workspace crates does a change affect?

A crate is affected when one of its files changed, or when it depends (transitively, through a normal, dev or
build path dependency) on an affected crate. Layers below a change are not affected. The graph is read from
`cargo metadata --format-version 1 --no-deps`, never hard-coded.

Everything is affected when a root build file changed (see EVERYTHING), when a file belongs to no crate and is not
documentation (so an unknown file is never silently skipped), or when `--all` is given or the diff cannot be
computed. Nothing is affected when only documentation changed: `*.md`, `docs/**`, `LICENSE*`, `.gitignore`.
A crate's own `README.md` is the exception: the crates include it as their crate documentation, so its doctests
are part of the crate.

Usage:
  affected.py --base REV [--head REV] [--merge-base]   diff `REV..HEAD` (`REV...HEAD` with --merge-base)
  affected.py --files PATH...                          use these changed paths instead of a git diff
  affected.py --all                                    everything (workflow_dispatch)
  Add --github-output FILE to append the outputs below to $GITHUB_OUTPUT.

Outputs (stdout as `key=value`, and to --github-output):
  any          `true` if any crate is affected, `false` if nothing is
  everything   `true` if the whole workspace is affected
  args         `--workspace` when everything is affected, else `-p a -p b ...`; for cargo
  packages     `-p a -p b ...` always
  crates       the affected crate names, space separated
  <crate>      `true` or `false` per workspace crate, the name with `-` as `_` (euhedral_core)
  reason       one line saying why
"""

import argparse
import fnmatch
import json
import os
import subprocess
import sys

# A change to any of these affects every crate. Patterns match the repo-relative path.
EVERYTHING = [
    "Cargo.toml",
    "Cargo.lock",
    "mise.toml",
    "clippy.toml",
    ".clippy.toml",
    "rustfmt.toml",
    ".rustfmt.toml",
    "rust-toolchain",
    "rust-toolchain.toml",
    "build.rs",
    ".cargo/*",
    ".github/*",
    "scripts/ci/*",
]

# A change to only these affects nothing.
DOCS = ["*.md", "docs/*", "LICENSE*", ".gitignore"]


def matches(path, patterns):
    # fnmatch's `*` also crosses `/`, so `.github/*` covers the whole tree and `*.md` any depth.
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)


class Graph:
    """The workspace's crates: name -> directory (repo-relative, `` for the root) and name -> dependencies."""

    def __init__(self, dirs, deps):
        self.dirs = dirs
        self.deps = deps  # name -> set of workspace crates it depends on

    @classmethod
    def from_metadata(cls, metadata):
        root = os.path.realpath(metadata["workspace_root"])
        members = set(metadata["workspace_members"])
        packages = [p for p in metadata["packages"] if p["id"] in members]

        def rel(path):
            return os.path.relpath(os.path.realpath(path), root).replace(os.sep, "/")

        by_dir = {rel(os.path.dirname(p["manifest_path"])): p["name"] for p in packages}
        dirs = {name: d for d, name in by_dir.items()}
        deps = {}
        for p in packages:
            found = set()
            for dep in p["dependencies"]:
                if dep.get("path") is not None:
                    target = by_dir.get(rel(dep["path"]))
                    if target is not None and target != p["name"]:
                        found.add(target)
            deps[p["name"]] = found
        return cls(dirs, deps)

    @classmethod
    def load(cls, root):
        out = subprocess.run(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return cls.from_metadata(json.loads(out))

    def crate_of(self, path):
        """The crate whose directory holds `path`, the longest directory winning."""
        best = None
        for name, d in self.dirs.items():
            if d not in ("", ".") and path.startswith(d + "/"):
                if best is None or len(d) > len(self.dirs[best]):
                    best = name
        return best

    def with_dependents(self, seed):
        """`seed` and every crate that depends on one of them, transitively."""
        out = set(seed)
        grew = True
        while grew:
            grew = False
            for name, deps in self.deps.items():
                if name not in out and deps & out:
                    out.add(name)
                    grew = True
        return out


def affected(graph, files):
    """(set of affected crates, everything?, reason) for the changed repo-relative `files`."""
    seed = set()
    for f in files:
        crate = graph.crate_of(f)
        if crate is not None:
            if f.endswith(".md") and os.path.basename(f) != "README.md":
                continue
            seed.add(crate)
        elif matches(f, EVERYTHING):
            return set(graph.dirs), True, f"{f} is a root build file"
        elif matches(f, DOCS):
            continue
        else:
            return set(graph.dirs), True, f"{f} belongs to no crate and is not documentation"
    if not files:
        return set(), False, "no files changed"
    if not seed:
        return set(), False, "only documentation changed"
    out = graph.with_dependents(seed)
    return out, out == set(graph.dirs), "changed: " + " ".join(sorted(seed))


def outputs(graph, crates, everything, reason):
    names = sorted(crates)
    packages = " ".join(f"-p {n}" for n in names)
    out = {
        "any": str(bool(names)).lower(),
        "everything": str(everything).lower(),
        "args": "--workspace" if everything else packages,
        "packages": packages,
        "crates": " ".join(names),
    }
    for n in sorted(graph.dirs):
        out[n.replace("-", "_")] = str(n in crates).lower()
    out["reason"] = reason
    return out


def changed_files(root, base, head, three_dot):
    cmd = ["git", "diff", "--name-only", "--no-renames", f"{base}{'...' if three_dot else '..'}{head}"]
    out = subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    ap.add_argument("--base")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--merge-base", action="store_true")
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--github-output")
    args = ap.parse_args(argv)
    root = os.path.realpath(args.root)

    graph = Graph.load(root)
    if args.all:
        crates, everything, reason = set(graph.dirs), True, "everything was asked for"
    else:
        files = args.files
        if files is None:
            if not args.base:
                ap.error("give --base, --files or --all")
            try:
                files = changed_files(root, args.base, args.head, args.merge_base)
            except subprocess.CalledProcessError as e:
                print(f"cannot diff {args.base}..{args.head}: {e.stderr.strip()}", file=sys.stderr)
                files = None
        if files is None:
            crates, everything, reason = set(graph.dirs), True, "the diff could not be computed"
        else:
            crates, everything, reason = affected(graph, files)

    out = outputs(graph, crates, everything, reason)
    lines = [f"{k}={v}" for k, v in out.items()]
    print("\n".join(lines))
    if args.github_output:
        with open(args.github_output, "a") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
