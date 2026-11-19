#!/usr/bin/env python3
"""Tests for gitlab_pipeline.py: run with `python3 scripts/ci/test_gitlab_pipeline.py`."""

import json
import os
import shlex
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

    def test_the_device_tests_run_in_their_own_job_on_a_shared_grant_of_the_card(self):
        jobs = pipeline("any=true\nargs=-p euhedral-gpu\nmiri=false\neuhedral_gpu=true\neuhedral_memory=true\n")
        device = jobs["device"]
        self.assertRegex(device["variables"]["ARBITER_ARGS"], r"--gpu shared:\d+\b")
        self.assertIn("--device /dev/nvidia0", device["variables"]["DOCKER_ARGS"])
        script = "\n".join(device["script"])
        for crate in ("euhedral-gpu", "euhedral-memory"):
            self.assertIn(f"cargo test -p {crate} --features cuda -- --nocapture", script)
            self.assertIn(f"device_check.py device-{crate}.log {crate}/tests", script)
        self.assertIn("set -o pipefail", device["script"])
        self.assertNotIn("--features cuda -- --nocapture", "\n".join(jobs["workspace"]["script"]))
        self.assertNotIn("cargo test -p euhedral-gpu --features cuda", "\n".join(jobs["workspace"]["script"]))

    def test_a_change_to_no_device_crate_has_no_device_job(self):
        self.assertNotIn("device", pipeline("any=true\nargs=-p euhedral-hashing\nmiri=false\n"))

    def test_a_matrix_with_an_apostrophe_reaches_the_build_script_whole(self):
        # A shard's name carries the reason it runs, which can quote ("Miri's wiring"): the matrix is one shell word.
        matrix = {"include": [{"name": "canary: Miri's wiring (changed)", "seeds": 1, "cpu": 1, "mem": "1G",
                                "miriflags": "-Zmiri-many-seeds=0..1", "commands": "cargo miri test -p x"}]}
        jobs = pipeline("any=false\nmiri=true\nmiri_matrix=" + json.dumps(matrix) + "\n")
        line = next(l for l in jobs["miri-build"]["script"] if "miri_build.py" in l)
        words = shlex.split(line)
        self.assertEqual(json.loads(words[words.index("--matrix") + 1]), matrix)

    def test_every_kind_of_job_asks_for_what_it_uses(self):
        matrix = {"include": [
            {"name": "a (seeds 0..8)", "seeds": 8, "cpu": 8, "mem": "2G", "miriflags": "-Zmiri-many-seeds=0..8", "commands": "cargo miri test -p x"},
            {"name": "b", "seeds": 1, "cpu": 1, "mem": "1G", "miriflags": "-Zmiri-many-seeds=0..1", "commands": "cargo miri test -p x"},
        ]}
        jobs = pipeline("any=true\nargs=-p x\nmiri=true\nmiri_matrix=" + json.dumps(matrix) + "\n")
        jobs = {name: job for name, job in jobs.items() if name not in ("workflow", "stages", "variables")}
        args = {name: job["variables"]["ARBITER_ARGS"] for name, job in jobs.items() if "variables" in job}
        self.assertEqual(args["miri 0: a (seeds 0..8)"], "--threads 8 --mem 2G")
        self.assertEqual(args["miri 1: b"], "--threads 1 --mem 1G")
        self.assertEqual(jobs["miri 0: a (seeds 0..8)"]["variables"]["MIRIFLAGS"], "-Zmiri-many-seeds=0..8")
        for name, job in jobs.items():
            if "ARBITER_EST" in job.get("variables", {}):
                self.assertLessEqual(int(job["variables"]["ARBITER_EST"]), 600, name)
        for name in ("miri 0: a (seeds 0..8)", "miri 1: b"):
            self.assertLessEqual(int(jobs[name]["variables"]["ARBITER_EST"]), 600, "a test job fits the runner's cap")


if __name__ == "__main__":
    unittest.main()
