#!/usr/bin/env python3
"""Reads the output of a crate's device tests (`cargo test -p CRATE --features cuda -- --nocapture`) and fails when they
did not run on the card.

A device test that finds no driver, card or kernel compiler returns early and passes, so cargo's "ok" proves nothing.
The test support prints `DEVICE-TEST-SKIPPED: reason` for such a skip and `DEVICE-TEST-SKIPPED-HARDWARE: reason` for a
skip because this card or driver lacks the feature under test. This script fails on the first kind, and on a run that
did not run every device test binary of the crate (every `tests/cuda*.rs`), and lists the second kind.

Usage: device_check.py LOG TESTS_DIR
"""
import glob
import os
import re
import sys

SKIP = "DEVICE-TEST-SKIPPED:"
HARDWARE = "DEVICE-TEST-SKIPPED-HARDWARE:"


def check(log, tests_dir):
    """Returns (problems, hardware skips) for the log text of one crate's device tests."""
    problems = []
    skips = [l.split(SKIP, 1)[1].strip() for l in log.splitlines() if SKIP in l]
    hardware = sorted({l.split(HARDWARE, 1)[1].strip() for l in log.splitlines() if HARDWARE in l})
    for reason in sorted(set(skips)):
        problems.append(f"{skips.count(reason)} device test(s) skipped: {reason}")
    ran = set(re.findall(r"Running tests/(\S+?)\.rs", log))
    expected = {os.path.basename(p)[:-3] for p in glob.glob(os.path.join(tests_dir, "cuda*.rs"))}
    for name in sorted(expected - ran):
        problems.append(f"the device test binary tests/{name}.rs did not run")
    if not expected:
        problems.append(f"no device test binary (tests/cuda*.rs) in {tests_dir}")
    return problems, hardware


def main(argv):
    log_path, tests_dir = argv[1], argv[2]
    with open(log_path, errors="replace") as f:
        problems, hardware = check(f.read(), tests_dir)
    for reason in hardware:
        print(f"device_check: skipped for the card's hardware (accepted): {reason}")
    for problem in problems:
        print(f"device_check: FAILED: {problem}")
    if problems:
        return 1
    print("device_check: every device test binary ran and none skipped for want of a driver, card or compiler")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
