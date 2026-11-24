#!/usr/bin/env python3
"""Which workspace crates does a change affect?

A crate is affected when one of its files changed, or when it depends (transitively, through a normal, dev or
build path dependency) on an affected crate. Layers below a change are not affected. The graph is read from
`cargo metadata --format-version 1 --no-deps`, never hard-coded.

Everything is affected when a root build file changed (see EVERYTHING), when a file belongs to no crate and is not
documentation or a known CI file (so an unknown file is never silently skipped), or when `--all` is given or the
diff cannot be computed. Nothing is affected when only documentation changed: `*.md`, `docs/**` (except BUILD files
under it, which are Bazel packages), `LICENSE*`, `.gitignore`. Two exceptions: a crate's own `README.md` (the crates
include it as their crate documentation, so its doctests are part of the crate) and any `.md` under a member's src/,
tests/, examples/ or benches/ (compile data there) count for their crate. A file that a member's BUILD.bazel names by
label (`//:docs/x.md` in a test's `data`) affects that member whatever it is, documentation included: the test reads it.

The root Bazel files (MODULE.bazel, MODULE.bazel.lock, .bazelrc, .bazelversion, .bazelignore and the root
BUILD.bazel) are root build files. A crate's own BUILD.bazel affects that crate; the generator crate's stale test
(BAZEL_GEN) is also affected by any member's Cargo.toml and by any file under a member's src/, tests/, examples/ or
benches/ or its build.rs.

The CI files select less than a root build file does:
  .github/workflows/ci.yml, .gitea/workflows/ci.yml   the workspace jobs run in full (they validate the workflow);
                                  Miri runs what the crate changes select, and the canary shard (the fastest
                                  single-threaded one, marked `"canary": true` in miri-shards.json), so Miri's wiring is
                                  exercised. Either file does it: they are twins, each run on its own forge
  .github/workflows/cache-warm.yml, .github/actionlint.yaml, scripts/ci/*.py   no crate (the `changes` job runs the
                                  script tests; the warm-up never runs for a pull request; actionlint's labels are read
                                  by the `workflows` job, which is not a crate's)
  scripts/ci/miri-shards.json     no crate; Miri runs the shards that were added or changed against the base
  scripts/ci/miri-toolchain       no crate; the pinned nightly changes Miri's results, so Miri runs every shard
Any other file under .github/, .gitea/ or scripts/ci/ is a root build file.

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
  miri         `true` if any Miri shard runs
  miri_matrix  a `{"include": [...]}` matrix of the Miri shards (scripts/ci/miri-shards.json) that run; each entry has
               the job's name, its commands (newline-joined), its seeds (how many), its seed range, the MIRIFLAGS it runs
               with, the cores (`cpu`) and memory (`mem`) it asks for, and why it runs. A shard with more seeds than
               SEEDS_PER_JOB becomes several jobs, each with a part of the range.
  miri_reasons the same reasons on one line (`name: why | name: why`)
  reason       one line saying why the workspace jobs run
"""

import argparse
import fnmatch
import json
import os
import re
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
    "MODULE.bazel",
    "MODULE.bazel.lock",
    "cargo-bazel-lock.json",
    ".bazelrc",
    ".bazelversion",
    ".bazelignore",
    "BUILD.bazel",
    "build.rs",
    ".cargo/*",
    ".github/*",
    ".gitea/*",
    "scripts/ci/*",
]

# The crate whose test reads every member's manifest and target files to see whether the generated BUILD files are stale
# (docs/design/bazel.md). It is affected by any member's Cargo.toml, by anything under a member's STALE_INPUT_DIRS or
# its build.rs: `git diff --name-only` cannot tell an added file from an edited one, and only an added or removed one can
# make a BUILD file stale, so every change there counts. A BUILD file anywhere in a member is its output: a hand edit to
# one makes it stale.
BAZEL_GEN = "euhedral-bazel-gen"
STALE_INPUT_DIRS = ("src/", "tests/", "examples/", "benches/")
STALE_INPUT_FILES = ("Cargo.toml", "build.rs")
BUILD_FILES = ("BUILD.bazel", "BUILD")

# CI files that select less (see the module docstring); they are looked at before EVERYTHING.
CI_WORKFLOWS = [".github/workflows/ci.yml", ".gitea/workflows/ci.yml"]
SHARDS_FILE = "scripts/ci/miri-shards.json"
TOOLCHAIN_FILE = "scripts/ci/miri-toolchain"
CI_NO_CRATES = [".github/workflows/cache-warm.yml", ".github/actionlint.yaml", "scripts/ci/*.py", SHARDS_FILE, TOOLCHAIN_FILE]

# A change to only these affects nothing.
DOCS = ["*.md", "docs/*", "LICENSE*", ".gitignore"]


def matches(path, patterns):
    # fnmatch's `*` also crosses `/`, so `.github/*` covers the whole tree and `*.md` any depth.
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)


class Graph:
    """The workspace's crates: name -> directory (repo-relative, `` for the root), name -> dependencies, and the files
    outside a crate that a crate's targets read (repo-relative path -> the crates whose BUILD file names it)."""

    def __init__(self, dirs, deps, data=None):
        self.dirs = dirs
        self.deps = deps  # name -> set of workspace crates it depends on
        self.data = data or {}

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
        graph = cls.from_metadata(json.loads(out))
        graph.data = data_inputs(root, graph.dirs)
        return graph

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


# Every quoted string of a BUILD file: a label (`//:docs/x.md`, `//pkg:file`), a package-relative path or a glob.
STRING = re.compile(r'"([^"]*)"')
GLOB_CHARS = re.compile(r"[*?\[{]")


def glob_dir(pattern):
    """The literal directory a glob pattern starts with: `docs/` of `docs/**`, "" (the package) of `**/*.md`."""
    literal = pattern[: GLOB_CHARS.search(pattern).start()]
    return literal[: literal.rfind("/") + 1].rstrip("/")
BUILD_FILES = ("BUILD.bazel", "BUILD")


def label_path(package, text):
    """The repo-relative path a label or a package-relative path in `package`'s BUILD file names, or None for a label
    of an external repository or one outside the repository. A target's name (`//:workspace_sources`) comes out as a
    path that no change ever matches, so it costs nothing."""
    if text.startswith("@"):
        return None
    if text.startswith("//"):
        pkg, _, target = text[2:].partition(":")
        path = os.path.join(pkg, target)  # `//pkg` alone is the package's own directory
    else:
        path = os.path.join(package, text[1:] if text.startswith(":") else text)
    path = os.path.normpath(path) if path else "."
    if path == "." or path == ".." or path.startswith("../") or os.path.isabs(path):
        return None
    return path


def data_inputs(root, dirs):
    """{path: crates} for every path a member's BUILD file names (a test's `data`, say), as a label or relative to the
    member's package. Whether the path exists is not asked: a change that deletes or renames the file still affects the
    crate that names it. A path may be a directory, whose every file then affects the crate."""
    out = {}
    for name, d in dirs.items():
        for build in BUILD_FILES:
            try:
                with open(os.path.join(root, d, build)) as f:
                    text = f.read()
            except FileNotFoundError:
                continue
            for word in STRING.findall(text):
                if not word.strip():
                    continue  # names nothing (an empty string is no path of the package)
                if GLOB_CHARS.search(word):
                    # A glob names files under its literal directory (the package's own when it starts with one).
                    path = os.path.normpath(os.path.join(d, glob_dir(word)))
                    path = None if path == "." or path.startswith("..") else path
                else:
                    path = label_path(d, word)
                if path is not None:
                    out.setdefault(path, set()).add(name)
    return out


def data_crates(graph, f):
    """The crates whose BUILD files name `f` or a directory above it."""
    found = set()
    for path, crates in graph.data.items():
        if f == path or f.startswith(path + "/"):
            found |= crates
    return found


def affected(graph, files, workflow_is_everything=True):
    """(set of affected crates, everything?, reason) for the changed repo-relative `files`.

    `ci.yml` is everything for the workspace jobs and, with `workflow_is_everything` false, nothing for Miri."""
    seed = set()
    for f in files:
        seed |= data_crates(graph, f)  # read by a crate's target, whatever else the file is
        if matches(f, CI_NO_CRATES):
            continue
        if f in CI_WORKFLOWS:
            if workflow_is_everything:
                return set(graph.dirs), True, f"{f} is the workflow itself: its jobs run in full"
            continue
        crate = graph.crate_of(f)
        if crate is not None:
            inside = f[len(graph.dirs[crate]) + 1:]
            if f.endswith(".md") and os.path.basename(f) != "README.md" and not inside.startswith(STALE_INPUT_DIRS):
                continue
            seed.add(crate)
            if BAZEL_GEN in graph.dirs and (
                inside in STALE_INPUT_FILES
                or inside.startswith(STALE_INPUT_DIRS)
                or os.path.basename(inside) in BUILD_FILES
            ):
                seed.add(BAZEL_GEN)
        elif matches(f, EVERYTHING):
            return set(graph.dirs), True, f"{f} is a root build file"
        elif os.path.basename(f) in BUILD_FILES:
            # Tested before DOCS: a BUILD file under docs/ is still a Bazel package outside every member.
            return set(graph.dirs), True, f"{f} is a BUILD file outside every member"
        elif matches(f, DOCS):
            continue
        else:
            return set(graph.dirs), True, f"{f} belongs to no crate and is not documentation"
    if not files:
        return set(), False, "no files changed"
    if not seed:
        return set(), False, "only documentation or CI files that select no crate changed"
    out = graph.with_dependents(seed)
    return out, out == set(graph.dirs), "changed: " + " ".join(sorted(seed))


def shard_problems(shards):
    """What is wrong with the shard list: a shard needs a unique name, a crate, `seeds` (a positive whole number) and
    commands that are `cargo miri test` runs (the toolchain is the job's)."""
    problems, names = [], set()
    for s in shards:
        name = s.get("name")
        if not name or name in names:
            problems.append(f"shard name missing or repeated: {name!r}")
        names.add(name)
        if not s.get("crate"):
            problems.append(f"{name}: no crate")
        seeds = s.get("seeds")
        if type(seeds) is not int or seeds < 1:
            problems.append(f"{name}: seeds must be a positive whole number, not {seeds!r}")
        if "mem" in s and not (isinstance(s["mem"], str) and s["mem"][:-1].isdigit() and s["mem"][-1] in "MG"):
            problems.append(f"{name}: mem is a size such as 2G, not {s['mem']!r}")
        if "canary" in s and s["canary"] is not True:
            problems.append(f"{name}: canary is true or absent")
        commands = s.get("commands")
        if not commands or not all(c.startswith("cargo miri test ") for c in commands):
            problems.append(f"{name}: commands must be `cargo miri test ...` runs")
    if sum(1 for s in shards if s.get("canary")) > 1:
        problems.append("more than one canary shard")
    return problems


def miri_selection(graph, files, shards, old_shards=()):
    """[(shard, why)] for the Miri shards that run for the changed `files`, in the shards' order.

    A shard runs because its crate is affected (the crate's files, or a root build file, or a crate it depends on), or
    because its entry in miri-shards.json was added or changed against `old_shards`, or because the pinned nightly
    changed, or as the canary when the workflow changed."""
    crates, everything, reason = affected(graph, files, workflow_is_everything=False)
    old = {s["name"]: s for s in old_shards}
    why = {s["name"]: [] for s in shards}
    for s in shards:
        if s["crate"] in crates:
            why[s["name"]].append(f"{s['crate']} is affected ({reason})")
        if TOOLCHAIN_FILE in files:
            why[s["name"]].append(f"{TOOLCHAIN_FILE} changed: the nightly changes Miri's results")
        if SHARDS_FILE in files and old.get(s["name"]) != s:
            why[s["name"]].append("the shard is new or changed in " + SHARDS_FILE)
    changed_workflows = [w for w in CI_WORKFLOWS if w in files]
    if changed_workflows:
        canary = [s for s in shards if s.get("canary")]
        if canary:
            names = " and ".join(changed_workflows)
            why[canary[0]["name"]].append(f"canary: {names} changed, so Miri's wiring is exercised")
    return [(s, "; ".join(why[s["name"]])) for s in shards if why[s["name"]]]


def miri_everything(shards, reason):
    """Every shard, for the same `reason`."""
    return [(s, reason) for s in shards]


# Miri runs the seeds of one job on one thread each (RAYON_NUM_THREADS does not limit it), so a job's seeds are the
# cores it uses: a shard with more seeds than this is split into jobs that each take a part of the range.
SEEDS_PER_JOB = 8
# The memory a Miri job asks for when its shard does not say (`"mem"`): the peak of a job measured at 0.5 GB and of
# eight seeds at 1 GB, with a margin.
MIRI_MEM = "2G"
# Leaks are ignored: the lock-free sources park frames on purpose. (Running the tests one at a time made no difference
# to the time: Miri interleaves a process's threads on one core either way.)
MIRIFLAGS = "-Zmiri-ignore-leaks"


def seed_ranges(seeds):
    """`[(from, to)]`: `seeds` seeds in as few jobs of at most SEEDS_PER_JOB as are needed, of equal size."""
    jobs = -(-seeds // SEEDS_PER_JOB)
    size = -(-seeds // jobs)
    return [(lo, min(lo + size, seeds)) for lo in range(0, seeds, size)]


def miri_matrix(selection):
    """The matrix for the Miri jobs: one entry per selected (shard, why) and seed range."""
    include = []
    for s, why in selection:
        ranges = seed_ranges(s["seeds"])
        for lo, hi in ranges:
            include.append({
                "name": s["name"] if len(ranges) == 1 else f"{s['name']} (seeds {lo}..{hi})",
                "seeds": hi - lo,
                "seed_range": f"{lo}..{hi}",
                "miriflags": f"-Zmiri-many-seeds={lo}..{hi} {MIRIFLAGS}",
                "cpu": hi - lo,
                "mem": s.get("mem", MIRI_MEM),
                "commands": "\n".join(s["commands"]),
                "reason": why,
            })
    return {"include": include}


def outputs(graph, crates, everything, reason, selection=()):
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
    matrix = miri_matrix(selection)
    out["miri"] = str(bool(matrix["include"])).lower()
    out["miri_matrix"] = json.dumps(matrix, separators=(",", ":"))
    out["miri_reasons"] = " | ".join(f"{e['name']}: {e['reason']}" for e in matrix["include"])
    out["reason"] = reason
    return out


def changed_files(root, base, head, three_dot):
    cmd = ["git", "diff", "--name-only", "--no-renames", f"{base}{'...' if three_dot else '..'}{head}"]
    out = subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line]


def shards_at(root, base, head, three_dot):
    """The shard list as the base of the diff has it (the merge base for `three_dot`); empty if it had none."""
    rev = base
    if three_dot:
        rev = subprocess.run(
            ["git", "merge-base", base, head], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
    shown = subprocess.run(["git", "show", f"{rev}:{SHARDS_FILE}"], cwd=root, capture_output=True, text=True)
    return json.loads(shown.stdout) if shown.returncode == 0 else []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    ap.add_argument("--base")
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--merge-base", action="store_true")
    ap.add_argument("--files", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--miri-shards", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "miri-shards.json"))
    ap.add_argument("--github-output")
    args = ap.parse_args(argv)
    root = os.path.realpath(args.root)

    with open(args.miri_shards) as f:
        shards = json.load(f)
    for problem in shard_problems(shards):
        ap.error(f"{args.miri_shards}: {problem}")

    graph = Graph.load(root)
    unknown = {s["crate"] for s in shards} - set(graph.dirs)
    if unknown:
        ap.error(f"{args.miri_shards} names crates that are not in the workspace: {sorted(unknown)}")
    files, old_shards = None, []
    if args.all:
        crates, everything, reason = set(graph.dirs), True, "everything was asked for"
        selection = miri_everything(shards, reason)
    else:
        files = args.files
        if files is None:
            if not args.base:
                ap.error("give --base, --files or --all")
            try:
                files = changed_files(root, args.base, args.head, args.merge_base)
                old_shards = shards_at(root, args.base, args.head, args.merge_base)
            except subprocess.CalledProcessError as e:
                print(f"cannot diff {args.base}..{args.head}: {e.stderr.strip()}", file=sys.stderr)
                files = None
        if files is None:
            crates, everything, reason = set(graph.dirs), True, "the diff could not be computed"
            selection = miri_everything(shards, reason)
        else:
            crates, everything, reason = affected(graph, files)
            selection = miri_selection(graph, files, shards, old_shards)
    out = outputs(graph, crates, everything, reason, selection)
    lines = [f"{k}={v}" for k, v in out.items()]
    print("\n".join(lines))
    for shard, why in selection:
        print(f"miri shard {shard['name']}: {why}", file=sys.stderr)
    if args.github_output:
        with open(args.github_output, "a") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
