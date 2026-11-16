#!/usr/bin/env python3
"""Tests for gitlab_pipeline.py: run with `python3 scripts/ci/test_gitlab_pipeline.py`."""

import json
import os
import subprocess
import sys
import unittest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gitlab_pipeline.py")


def pipeline(affected):
    out = subprocess.run([sys.executable, SCRIPT], input=affected, capture_output=True, text=True, check=True).stdout
    return json.loads(out)


class ChildPipeline(unittest.TestCase):
    def test_the_child_runs_in_a_merge_request_pipeline(self):
        # Without a `workflow` of its own the child of a merge request pipeline is refused ("downstream pipeline could
        # not be created"), so every merge request pipeline would fail at its trigger.
        for affected in ("any=false\nreason=docs only\n", "any=true\nargs=-p euhedral-hashing\nmiri=false\n"):
            self.assertEqual(pipeline(affected)["workflow"], {"rules": [{"when": "always"}]})

    def test_a_change_that_affects_nothing_still_passes(self):
        jobs = pipeline("any=false\nreason=docs only\n")
        self.assertIn("nothing-affected", jobs)
        self.assertNotIn("workspace", jobs)

    def test_an_affected_crate_gets_the_workspace_job(self):
        jobs = pipeline("any=true\nargs=-p euhedral-hashing\nmiri=false\n")
        self.assertIn("workspace", jobs)
        self.assertNotIn("nothing-affected", jobs)
        self.assertIn("cargo test -p euhedral-hashing", jobs["workspace"]["script"])


if __name__ == "__main__":
    unittest.main()
