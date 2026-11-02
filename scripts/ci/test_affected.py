#!/usr/bin/env python3
"""Tests for affected.py on a synthetic graph shaped like the workspace's layers. Run: python3 scripts/ci/test_affected.py"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import affected as A  # noqa: E402

NAMES = ["ds", "hash", "hw", "core", "graph", "io", "mem", "gpu", "bench"]
DEPS = {
    "ds": set(),
    "hash": set(),
    "hw": {"ds"},
    "core": {"ds", "hw", "hash"},
    "graph": {"core", "ds", "hw", "hash"},
    "io": {"core", "ds"},
    "mem": {"core", "ds", "hash", "graph"},
    "gpu": {"core", "graph", "mem", "ds", "hash", "hw"},
    "bench": {"core", "ds", "hw"},
}  # as the real workspace: benchmarks does not depend on graph, io, memory or gpu
G = A.Graph({n: f"euhedral-{n}" for n in NAMES}, DEPS)


def run(*files):
    crates, everything, _ = A.affected(G, list(files))
    return sorted(crates), everything


class Affected(unittest.TestCase):
    def test_docs_only_is_nothing(self):
        for f in ["README.md", "CHANGELOG.md", "docs/x/y.md", "LICENSE", "euhedral-core/GLOSSARY.md", ".gitignore"]:
            self.assertEqual(run(f), ([], False), f)

    def test_no_files_is_nothing(self):
        self.assertEqual(run(), ([], False))

    def test_core_change(self):
        self.assertEqual(
            run("euhedral-core/src/lib.rs"), (["bench", "core", "gpu", "graph", "io", "mem"], False)
        )

    def test_gpu_change(self):
        # Nothing depends on gpu (benchmarks sits above it by layer but does not depend on it).
        self.assertEqual(run("euhedral-gpu/src/x.rs"), (["gpu"], False))

    def test_leaf_and_layers_below_are_unrelated(self):
        self.assertEqual(run("euhedral-io/src/a.rs"), (["io"], False))
        self.assertEqual(run("euhedral-hash/src/a.rs")[0], sorted(set(NAMES) - {"ds", "hw"}))

    def test_lowest_layer_change_is_everything_above(self):
        crates, everything = run("euhedral-ds/src/a.rs")
        self.assertEqual(sorted(crates), sorted(set(NAMES) - {"hash"}))
        self.assertFalse(everything)

    def test_root_build_files_are_everything(self):
        for f in ["Cargo.lock", "Cargo.toml", "mise.toml", "clippy.toml", "rustfmt.toml", ".github/workflows/ci.yml", "scripts/ci/affected.py", "build.rs"]:
            crates, everything = run(f)
            self.assertTrue(everything, f)
            self.assertEqual(sorted(crates), sorted(NAMES), f)

    def test_unknown_file_is_everything(self):
        self.assertTrue(run("tools/new.sh")[1])

    def test_crate_readme_counts_but_other_md_does_not(self):
        self.assertEqual(run("euhedral-ds/README.md")[0], sorted(set(NAMES) - {"hash"}))
        self.assertEqual(run("euhedral-ds/GLOSSARY.md"), ([], False))

    def test_docs_plus_one_crate(self):
        self.assertEqual(run("README.md", "euhedral-gpu/src/a.rs"), (["gpu"], False))

    def test_nested_directories_map_to_the_longest_prefix(self):
        g = A.Graph({"a": "a", "b": "a/b"}, {"a": set(), "b": set()})
        self.assertEqual(g.crate_of("a/b/src/x.rs"), "b")
        self.assertEqual(g.crate_of("a/src/x.rs"), "a")

    def test_outputs(self):
        crates, everything, reason = A.affected(G, ["euhedral-gpu/src/a.rs"])
        out = A.outputs(G, crates, everything, reason)
        self.assertEqual(out["any"], "true")
        self.assertEqual(out["args"], "-p gpu")
        self.assertEqual(out["gpu"], "true")
        self.assertEqual(out["core"], "false")
        crates, everything, reason = A.affected(G, ["Cargo.lock"])
        out = A.outputs(G, crates, everything, reason)
        self.assertEqual(out["args"], "--workspace")
        self.assertEqual(out["core"], "true")
        crates, everything, reason = A.affected(G, ["README.md"])
        self.assertEqual(A.outputs(G, crates, everything, reason)["any"], "false")


class Miri(unittest.TestCase):
    SHARDS = [
        {"name": "a", "crate": "core", "commands": ["c1", "c2"]},
        {"name": "b", "crate": "graph", "commands": ["c3"]},
        {"name": "c", "crate": "ds", "commands": ["c4"]},
    ]

    def matrix(self, *files):
        crates, everything, reason = A.affected(G, list(files))
        return A.outputs(G, crates, everything, reason, self.SHARDS)

    def test_only_affected_crates_have_shards(self):
        out = self.matrix("euhedral-graph/src/a.rs")
        self.assertEqual(json.loads(out["miri_matrix"]), {"include": [{"name": "b", "commands": "c3"}]})
        self.assertEqual(out["miri"], "true")

    def test_a_lower_layer_brings_the_shards_above(self):
        names = [s["name"] for s in json.loads(self.matrix("euhedral-ds/src/a.rs")["miri_matrix"])["include"]]
        self.assertEqual(names, ["a", "b", "c"])

    def test_no_shard_when_none_is_affected(self):
        for f in ["README.md", "euhedral-io/src/a.rs"]:
            out = self.matrix(f)
            self.assertEqual((out["miri"], json.loads(out["miri_matrix"])), ("false", {"include": []}), f)

    def test_commands_are_newline_joined(self):
        self.assertEqual(json.loads(self.matrix("euhedral-core/src/a.rs")["miri_matrix"])["include"][0]["commands"], "c1\nc2")

    def test_the_real_shards_name_workspace_crates_only(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "miri-shards.json")) as f:
            shards = json.load(f)
        self.assertTrue(shards)
        for s in shards:
            self.assertTrue(s["name"] and s["commands"], s)
            self.assertTrue(all(c.startswith("cargo +nightly miri test ") for c in s["commands"]), s)
            self.assertTrue(all(f"-p {s['crate']} " in c for c in s["commands"]), s)


class Metadata(unittest.TestCase):
    def test_reads_path_dependencies_and_ignores_the_rest(self):
        md = {
            "workspace_root": "/w",
            "workspace_members": ["a", "b"],
            "packages": [
                {"id": "a", "name": "a", "manifest_path": "/w/a/Cargo.toml", "dependencies": []},
                {
                    "id": "b",
                    "name": "b",
                    "manifest_path": "/w/b/Cargo.toml",
                    "dependencies": [
                        {"name": "a", "path": "/w/a"},
                        {"name": "serde", "path": None},
                        {"name": "b", "path": "/w/b"},
                    ],
                },
                {"id": "x", "name": "x", "manifest_path": "/w/x/Cargo.toml", "dependencies": []},
            ],
        }
        g = A.Graph.from_metadata(md)
        self.assertEqual(g.deps, {"a": set(), "b": {"a"}})
        self.assertEqual(g.dirs, {"a": "a", "b": "b"})


if __name__ == "__main__":
    unittest.main()
