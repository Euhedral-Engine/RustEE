#!/usr/bin/env python3
"""Tests for affected.py on a synthetic graph shaped like the workspace's layers. Run: python3 scripts/ci/test_affected.py"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import affected as A  # noqa: E402

NAMES = ["ds", "hash", "hw", "core", "graph", "io", "mem", "net", "gpu", "bench", "bazelgen"]
DEPS = {
    "ds": set(),
    "hash": set(),
    "hw": {"ds"},
    "core": {"ds", "hw", "hash"},
    "graph": {"core", "ds", "hw", "hash"},
    "io": {"core", "ds"},
    "mem": {"core", "ds", "hash", "graph"},
    "net": set(),
    "gpu": {"core", "graph", "mem", "ds", "hash", "hw"},
    "bench": {"core", "ds", "hw"},
    "bazelgen": set(),
}  # as the real workspace: benchmarks does not depend on graph, io, memory or gpu
DIRS = {n: f"euhedral-{n}" for n in NAMES} | {"bazelgen": "tools/bazel-gen"}
G = A.Graph(DIRS, DEPS)
REAL_BAZEL_GEN = A.BAZEL_GEN
A.BAZEL_GEN = "bazelgen"  # the synthetic graph names the generator crate this way


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
            run("euhedral-core/src/lib.rs"), (["bazelgen", "bench", "core", "gpu", "graph", "io", "mem"], False)
        )

    def test_gpu_change(self):
        # Nothing depends on gpu (benchmarks sits above it by layer but does not depend on it).
        self.assertEqual(run("euhedral-gpu/src/x.rs"), (["bazelgen", "gpu"], False))

    def test_leaf_and_layers_below_are_unrelated(self):
        self.assertEqual(run("euhedral-io/src/a.rs"), (["bazelgen", "io"], False))
        # The net crate has no dependency yet: nothing below it is affected by it, and it by nothing below.
        self.assertEqual(run("euhedral-net/src/a.rs"), (["bazelgen", "net"], False))
        self.assertEqual(run("euhedral-hash/src/a.rs")[0], sorted(set(NAMES) - {"ds", "hw", "net"}))

    def test_lowest_layer_change_is_everything_above(self):
        crates, everything = run("euhedral-ds/src/a.rs")
        self.assertEqual(sorted(crates), sorted(set(NAMES) - {"hash", "net"}))
        self.assertFalse(everything)

    def test_root_build_files_are_everything(self):
        for f in [
            "Cargo.lock", "Cargo.toml", "mise.toml", "clippy.toml", ".clippy.toml", "rustfmt.toml", ".rustfmt.toml",
            "rust-toolchain", "rust-toolchain.toml", ".cargo/config.toml", "build.rs",
            "MODULE.bazel", "MODULE.bazel.lock", ".bazelrc", ".bazelversion", ".bazelignore", "BUILD.bazel",
        ]:
            crates, everything = run(f)
            self.assertTrue(everything, f)
            self.assertEqual(sorted(crates), sorted(NAMES), f)
            self.assertIn("root build file", A.affected(G, [f])[2], f)  # not merely a file that belongs to no crate

    def test_a_build_file_under_docs_is_not_documentation(self):
        for f in ["docs/x/BUILD.bazel", "docs/BUILD"]:
            crates, everything = run(f)
            self.assertTrue(everything, f)
            self.assertEqual(sorted(crates), sorted(NAMES), f)

    def test_a_crates_build_file_affects_that_crate_and_the_stale_check(self):
        # A hand edit to a generated file makes it stale: the generator crate's stale test must run.
        self.assertEqual(run("euhedral-io/BUILD.bazel"), (["bazelgen", "io"], False))
        self.assertEqual(run("euhedral-gpu/BUILD.bazel"), (["bazelgen", "gpu"], False))
        self.assertEqual(run("euhedral-gpu/kernels/BUILD.bazel"), (["bazelgen", "gpu"], False))

    def test_the_inputs_of_the_stale_check_affect_the_generator_crate(self):
        for f in [
            "euhedral-gpu/Cargo.toml", "euhedral-gpu/src/a/b.rs", "euhedral-gpu/tests/t.rs", "euhedral-gpu/examples/e.rs",
            "euhedral-gpu/benches/b.rs", "euhedral-gpu/build.rs",
        ]:
            self.assertEqual(run(f), (["bazelgen", "gpu"], False), f)
        # Below a crate others depend on, the dependents come as before, with the generator crate added.
        self.assertEqual(run("euhedral-ds/Cargo.toml")[0], sorted(set(NAMES) - {"hash", "net"}))
        self.assertEqual(run("euhedral-io/src/x.rs"), (["bazelgen", "io"], False))

    def test_other_files_of_a_crate_do_not_affect_the_generator_crate(self):
        for f in ["euhedral-gpu/README.md", "euhedral-gpu/kernels/k.cu", "euhedral-gpu/Cargo.lock"]:
            self.assertNotIn("bazelgen", run(f)[0], f)

    def test_the_workflow_runs_the_workspace_jobs_in_full(self):
        for f in [".github/workflows/ci.yml", ".gitea/workflows/ci.yml"]:
            self.assertEqual(run(f), (sorted(NAMES), True), f)

    def test_the_gitea_directory_is_a_root_build_file_apart_from_the_workflow(self):
        for f in [".gitea/CODEOWNERS", ".gitea/workflows/other.yml", ".gitea/rollover.toml"]:
            self.assertEqual(run(f), (sorted(NAMES), True), f)

    def test_ci_files_that_select_no_crate(self):
        for f in [
            ".github/workflows/cache-warm.yml", ".github/actionlint.yaml", "scripts/ci/affected.py",
            "scripts/ci/test_affected.py", "scripts/ci/miri_build.py", "scripts/ci/miri-shards.json", "scripts/ci/miri-toolchain",
        ]:
            self.assertEqual(run(f), ([], False), f)

    def test_other_ci_files_are_still_root_build_files(self):
        for f in [".github/CODEOWNERS", ".github/workflows/other.yml", "scripts/ci/run.sh"]:
            self.assertEqual(run(f), (sorted(NAMES), True), f)

    def test_a_ci_file_beside_a_crate_change_adds_nothing(self):
        self.assertEqual(run("scripts/ci/miri-toolchain", "euhedral-gpu/src/a.rs"), (["bazelgen", "gpu"], False))

    def test_unknown_file_is_everything(self):
        self.assertTrue(run("tools/new.sh")[1])

    def test_crate_readme_counts_but_other_md_does_not(self):
        self.assertEqual(run("euhedral-ds/README.md")[0], sorted(set(NAMES) - {"hash", "net", "bazelgen"}))
        self.assertEqual(run("euhedral-ds/GLOSSARY.md"), ([], False))

    def test_docs_plus_one_crate(self):
        self.assertEqual(run("README.md", "euhedral-gpu/src/a.rs"), (["bazelgen", "gpu"], False))

    def test_nested_directories_map_to_the_longest_prefix(self):
        g = A.Graph({"a": "a", "b": "a/b"}, {"a": set(), "b": set()})
        self.assertEqual(g.crate_of("a/b/src/x.rs"), "b")
        self.assertEqual(g.crate_of("a/src/x.rs"), "a")

    def test_outputs(self):
        crates, everything, reason = A.affected(G, ["euhedral-gpu/src/a.rs"])
        out = A.outputs(G, crates, everything, reason)
        self.assertEqual(out["any"], "true")
        self.assertEqual(out["args"], "-p bazelgen -p gpu")
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

    def test_the_gitea_workflow_runs_the_canary_too(self):
        self.assertEqual(self.names(".gitea/workflows/ci.yml"), ["c"])
        why = self.select(".gitea/workflows/ci.yml")[0][1]
        self.assertIn("canary", why)
        self.assertIn(".gitea/workflows/ci.yml", why)
        self.assertEqual(self.names(".gitea/workflows/ci.yml", "euhedral-graph/src/a.rs"), ["b", "c"])

    def test_both_workflows_changed_name_both_in_the_canary_reason(self):
        why = self.select(".github/workflows/ci.yml", ".gitea/workflows/ci.yml")
        self.assertEqual([s["name"] for s, _ in why], ["c"])
        self.assertIn(".github/workflows/ci.yml and .gitea/workflows/ci.yml", why[0][1])

    def test_the_actionlint_labels_run_no_shard(self):
        self.assertEqual(self.names(".github/actionlint.yaml"), [])

    def test_the_gitea_directory_runs_every_shard_but_the_workflow(self):
        for f in [".gitea/CODEOWNERS", ".gitea/workflows/other.yml"]:
            self.assertEqual(self.names(f), ["a", "b", "c"], f)

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

    def test_the_generator_crate_is_in_the_real_workspace(self):
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        self.assertIn(REAL_BAZEL_GEN, A.Graph.load(root).dirs)

    def test_seeds_are_split_into_jobs_that_cover_the_range_once(self):
        self.assertEqual(A.seed_ranges(1), [(0, 1)])
        self.assertEqual(A.seed_ranges(4), [(0, 4)])
        self.assertEqual(A.seed_ranges(8), [(0, 8)])
        self.assertEqual(A.seed_ranges(16), [(0, 8), (8, 16)])
        self.assertEqual(A.seed_ranges(20), [(0, 7), (7, 14), (14, 20)])
        for n in range(1, 70):
            ranges = A.seed_ranges(n)
            self.assertEqual([i for lo, hi in ranges for i in range(lo, hi)], list(range(n)), n)
            self.assertTrue(all(0 < hi - lo <= A.SEEDS_PER_JOB for lo, hi in ranges), n)

    def test_a_shard_with_many_seeds_is_several_jobs_that_ask_for_their_own_cores(self):
        include = json.loads(self.matrix("euhedral-core/src/a.rs")["miri_matrix"])["include"]
        jobs = [e for e in include if e["name"].startswith("a")]
        self.assertEqual([(e["name"], e["seeds"], e["cpu"], e["seed_range"]) for e in jobs],
                         [("a (seeds 0..8)", 8, 8, "0..8"), ("a (seeds 8..16)", 8, 8, "8..16")])
        self.assertEqual({e["commands"] for e in jobs}, {"c1\nc2"})
        self.assertIn("-Zmiri-many-seeds=8..16 ", jobs[1]["miriflags"])
        self.assertIn("-Zmiri-ignore-leaks", jobs[1]["miriflags"])
        self.assertEqual(jobs[0]["mem"], A.MIRI_MEM)

    def test_a_shard_can_name_its_memory(self):
        shards = [{**self.SHARDS[1], "mem": "3G", "commands": ["cargo miri test -p graph --lib"]}]
        [entry] = A.miri_matrix([(shards[0], "why")])["include"]
        self.assertEqual((entry["mem"], entry["cpu"], entry["seed_range"]), ("3G", 4, "0..4"))
        self.assertTrue(A.shard_problems([{**shards[0], "mem": "lots"}]))
        self.assertEqual(A.shard_problems(shards), [])

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
