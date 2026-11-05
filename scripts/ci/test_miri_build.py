#!/usr/bin/env python3
"""Tests for miri_build.py: run with `python3 scripts/ci/test_miri_build.py`."""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import miri_build as M  # noqa: E402


class BuildCommands(unittest.TestCase):
    def test_a_build_is_the_selection_without_the_test_filters(self):
        self.assertEqual(
            M.build_command("cargo miri test -p euhedral-gpu --features fake --lib -- ring::tests --skip a::b"),
            "cargo miri test --no-run -p euhedral-gpu --features fake --lib",
        )
        self.assertEqual(
            M.build_command("cargo miri test -p euhedral-gpu --test fake_shared_lane"),
            "cargo miri test --no-run -p euhedral-gpu --test fake_shared_lane",
        )

    def test_shards_that_select_the_same_build_share_one_command(self):
        commands = [
            "cargo miri test -p x --lib -- a",
            "cargo miri test -p x --lib -- --exact b",
            "cargo miri test -p y --lib -- a",
            "cargo miri test -p x --lib -- c",
        ]
        self.assertEqual(
            M.build_commands(commands),
            ["cargo miri test --no-run -p x --lib", "cargo miri test --no-run -p y --lib"],
        )

    def test_a_command_that_is_not_a_miri_test_is_refused(self):
        with self.assertRaises(ValueError):
            M.build_command("cargo test -p x --lib")
        with self.assertRaises(ValueError):
            M.build_command("cargo +nightly miri test -p x --lib")

    def test_the_matrix_of_changes_gives_the_builds_of_its_shards(self):
        matrix = {"include": [{"name": "a", "commands": "cargo miri test -p x --lib -- a\ncargo miri test -p y --lib"}]}
        self.assertEqual(
            M.build_commands([c for e in matrix["include"] for c in e["commands"].split("\n")]),
            ["cargo miri test --no-run -p x --lib", "cargo miri test --no-run -p y --lib"],
        )

    def test_every_real_shard_command_has_a_build(self):
        with open(os.path.join(M.HERE, "miri-shards.json")) as f:
            shards = json.load(f)
        built = M.build_commands([c for s in shards for c in s["commands"]])
        self.assertTrue(built)
        self.assertTrue(all(b.startswith("cargo miri test --no-run -p ") and " -- " not in b for b in built), built)


if __name__ == "__main__":
    unittest.main()
