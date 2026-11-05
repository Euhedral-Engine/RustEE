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
        for f in [
            "Cargo.lock", "Cargo.toml", "mise.toml", "clippy.toml", ".clippy.toml", "rustfmt.toml", ".rustfmt.toml",
            "rust-toolchain", "rust-toolchain.toml", ".cargo/config.toml", "build.rs",
        ]:
            crates, everything = run(f)
            self.assertTrue(everything, f)
            self.assertEqual(sorted(crates), sorted(NAMES), f)

    def test_the_workflow_runs_the_workspace_jobs_in_full(self):
        self.assertEqual(run(".github/workflows/ci.yml"), (sorted(NAMES), True))

    def test_ci_files_that_select_no_crate(self):
        for f in [
            ".github/workflows/cache-warm.yml", "scripts/ci/affected.py", "scripts/ci/test_affected.py",
            "scripts/ci/miri_build.py", "scripts/ci/miri-shards.json", "scripts/ci/miri-toolchain",
        ]:
            self.assertEqual(run(f), ([], False), f)

    def test_other_ci_files_are_still_root_build_files(self):
        for f in [".github/CODEOWNERS", ".github/workflows/other.yml", "scripts/ci/run.sh"]:
            self.assertEqual(run(f), (sorted(NAMES), True), f)

    def test_a_ci_file_beside_a_crate_change_adds_nothing(self):
        self.assertEqual(run("scripts/ci/miri-toolchain", "euhedral-gpu/src/a.rs"), (["gpu"], False))

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
        {"name": "a", "crate": "core", "seeds": 16, "commands": ["c1", "c2"]},
        {"name": "b", "crate": "graph", "seeds": 4, "commands": ["c3"]},
        {"name": "c", "crate": "ds", "seeds": 1, "canary": True, "commands": ["c4"]},
    ]

    def select(self, *files, old=None):
        old = self.SHARDS if old is None else old
        return A.miri_selection(G, list(files), self.SHARDS, old)

    def names(self, *files, **kw):
        return [s["name"] for s, _ in self.select(*files, **kw)]

    def matrix(self, *files):
        crates, everything, reason = A.affected(G, list(files))
        return A.outputs(G, crates, everything, reason, self.select(*files))

    def test_only_affected_crates_have_shards(self):
        out = self.matrix("euhedral-graph/src/a.rs")
        include = json.loads(out["miri_matrix"])["include"]
        self.assertEqual([(e["name"], e["seeds"], e["commands"]) for e in include], [("b", 4, "c3")])
        self.assertIn("graph is affected", include[0]["reason"])
        self.assertEqual(out["miri"], "true")
        self.assertTrue(out["miri_reasons"].startswith("b: graph is affected"))

    def test_a_lower_layer_brings_the_shards_above(self):
        self.assertEqual(self.names("euhedral-ds/src/a.rs"), ["a", "b", "c"])

    def test_no_shard_when_none_is_affected(self):
        for f in ["README.md", "euhedral-io/src/a.rs", ".github/workflows/cache-warm.yml", "scripts/ci/affected.py"]:
            out = self.matrix(f)
            self.assertEqual((out["miri"], json.loads(out["miri_matrix"])["include"]), ("false", []), f)

    def test_root_build_files_run_every_shard(self):
        for f in ["Cargo.lock", "mise.toml", ".cargo/config.toml", ".github/CODEOWNERS", "scripts/ci/run.sh"]:
            self.assertEqual(self.names(f), ["a", "b", "c"], f)

    def test_the_pinned_nightly_runs_every_shard_and_nothing_else_runs(self):
        self.assertEqual(self.names("scripts/ci/miri-toolchain"), ["a", "b", "c"])
        crates, everything, reason = A.affected(G, ["scripts/ci/miri-toolchain"])
        self.assertEqual((crates, everything), (set(), False))
        self.assertIn("nightly", self.select("scripts/ci/miri-toolchain")[0][1])

    def test_the_workflow_runs_the_canary_and_what_the_crates_select(self):
        self.assertEqual(self.names(".github/workflows/ci.yml"), ["c"])
        self.assertIn("canary", self.select(".github/workflows/ci.yml")[0][1])
        self.assertEqual(self.names(".github/workflows/ci.yml", "euhedral-core/src/a.rs"), ["a", "b", "c"])
        self.assertEqual(self.names(".github/workflows/ci.yml", "euhedral-graph/src/a.rs"), ["b", "c"])

    def test_the_workflow_without_a_canary_runs_no_shard(self):
        shards = [{k: v for k, v in s.items() if k != "canary"} for s in self.SHARDS]
        self.assertEqual(A.miri_selection(G, [".github/workflows/ci.yml"], shards, shards), [])

    def test_the_shards_file_runs_the_shards_added_or_changed(self):
        shards_file = "scripts/ci/miri-shards.json"
        self.assertEqual(self.names(shards_file), [])
        changed = [dict(self.SHARDS[0], seeds=4), self.SHARDS[1], self.SHARDS[2]]
        self.assertEqual(self.names(shards_file, old=changed), ["a"])
        self.assertIn("new or changed", self.select(shards_file, old=changed)[0][1])
        self.assertEqual(self.names(shards_file, old=self.SHARDS[:2]), ["c"])
        self.assertEqual(self.names(shards_file, old=[]), ["a", "b", "c"])

    def test_the_shards_file_adds_to_what_the_crates_select(self):
        changed = [self.SHARDS[0], dict(self.SHARDS[1], seeds=16), self.SHARDS[2]]
        names = self.names("scripts/ci/miri-shards.json", "euhedral-core/src/a.rs", old=changed)
        self.assertEqual(names, ["a", "b"])
        why = dict((s["name"], w) for s, w in self.select("scripts/ci/miri-shards.json", "euhedral-gpu/src/x.rs", old=changed))
        self.assertEqual(list(why), ["b"])
        self.assertIn("new or changed", why["b"])

    def test_every_reason_is_given(self):
        for f in [["euhedral-core/src/a.rs"], ["Cargo.lock"], ["scripts/ci/miri-toolchain"], [".github/workflows/ci.yml"]]:
            self.assertTrue(all(why for _, why in self.select(*f)), f)

    def test_commands_are_newline_joined(self):
        self.assertEqual(json.loads(self.matrix("euhedral-core/src/a.rs")["miri_matrix"])["include"][0]["commands"], "c1\nc2")

    def test_all_asks_for_every_shard(self):
        self.assertEqual([s["name"] for s, _ in A.miri_everything(self.SHARDS, "on demand")], ["a", "b", "c"])

    def test_the_real_shards_name_workspace_crates_only(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "miri-shards.json")) as f:
            shards = json.load(f)
        self.assertTrue(shards)
        for s in shards:
            self.assertTrue(s["name"] and s["commands"], s)
            self.assertTrue(all(c.startswith("cargo miri test ") for c in s["commands"]), s)
            self.assertTrue(all(f"-p {s['crate']} " in c for c in s["commands"]), s)
        self.assertEqual(A.shard_problems(shards), [])
        self.assertEqual([s["name"] for s in shards if s.get("canary")], ["hardware-utils: CPU sets"])

    def test_a_shard_needs_its_seeds_a_name_and_miri_commands(self):
        good = {"name": "a", "crate": "core", "seeds": 4, "commands": ["cargo miri test -p core --lib"]}
        self.assertEqual(A.shard_problems([good]), [])
        for bad in [
            {**good, "seeds": 0},
            {**good, "seeds": "4"},
            {**good, "seeds": True},
            {k: v for k, v in good.items() if k != "seeds"},
            {**good, "commands": ["cargo +nightly miri test -p core --lib"]},
            {**good, "commands": []},
            {**good, "crate": ""},
            {**good, "canary": False},
        ]:
            self.assertTrue(A.shard_problems([bad]), bad)
        self.assertTrue(A.shard_problems([good, good]), "a name is used once")
        self.assertTrue(A.shard_problems([{**good, "canary": True}, {**good, "name": "b", "canary": True}]), "one canary")


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
