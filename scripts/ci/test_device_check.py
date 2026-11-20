#!/usr/bin/env python3
"""Tests for device_check.py: run with `python3 scripts/ci/test_device_check.py`."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import device_check


def tests_dir(*names):
    d = tempfile.mkdtemp()
    for n in names + ("planner",):
        open(os.path.join(d, n + ".rs"), "w").close()
    return d


RAN = "     Running tests/cuda.rs (target/debug/deps/cuda-1)\ntest a ... ok\n"


class DeviceCheck(unittest.TestCase):
    def test_a_run_of_every_binary_with_no_skip_passes(self):
        problems, hardware = device_check.check(RAN, tests_dir("cuda"))
        self.assertEqual((problems, hardware), ([], []))

    def test_a_skip_for_want_of_a_card_fails(self):
        log = RAN + "DEVICE-TEST-SKIPPED: no CUDA device (x)\nDEVICE-TEST-SKIPPED: no CUDA device (x)\n"
        problems, _ = device_check.check(log, tests_dir("cuda"))
        self.assertEqual(problems, ["2 device test(s) skipped: no CUDA device (x)"])

    def test_a_hardware_skip_is_listed_and_accepted(self):
        log = RAN + "DEVICE-TEST-SKIPPED-HARDWARE: compute capability below 9.0: skipped\n"
        problems, hardware = device_check.check(log, tests_dir("cuda"))
        self.assertEqual((problems, hardware), ([], ["compute capability below 9.0: skipped"]))

    def test_a_binary_that_did_not_run_fails(self):
        problems, _ = device_check.check(RAN, tests_dir("cuda", "cuda_graph"))
        self.assertEqual(problems, ["the device test binary tests/cuda_graph.rs did not run"])

    def test_a_log_with_no_device_test_fails(self):
        problems, _ = device_check.check("", tests_dir("cuda"))
        self.assertEqual(len(problems), 1)


if __name__ == "__main__":
    unittest.main()
