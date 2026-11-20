#!/usr/bin/env python3
"""Pins .gitea/workflows/ci.yml and .github/workflows/ci.yml to each other, and the Gitea one to what its jobs promise.
Run: python3 scripts/ci/test_workflow_twins.py (needs PyYAML).

The two files are twins: for the jobs both have, the steps that are not forge-only are the same, field for field and in
order. A forge-only step has a name that ends in `(gitea)` or `(github)`; a step that `uses` an action may differ, since
the actions differ (checkout, caches, artifacts). The `ci-ok` needs list may differ. Everything else a job promises
(which jobs a change runs, the leases, the device job, the one place a secret is read) is pinned for the Gitea file below.
"""

import json
import os
import re
import sys
import unittest

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, HERE)
import affected  # noqa: E402


def load(path):
    with open(os.path.join(ROOT, path)) as f:
        doc = yaml.safe_load(f)
    doc["on"] = doc.pop(True, doc.get("on"))  # YAML reads a bare `on` as the boolean
    return doc


GITEA = load(".gitea/workflows/ci.yml")
GITHUB = load(".github/workflows/ci.yml")
TWIN_JOBS = ["changes", "workspace", "miri-build", "miri", "ci-ok"]
TWIN_FIELDS = ["name", "if", "run", "env", "id", "shell", "working-directory"]
DOCKER_LABELS = {"arbiter-docker", "arbiter-docker-unconfined", "arbiter-gpu"}
CHANGES_OUTPUT = re.compile(r"needs\.changes\.outputs\.(\w+)")


def steps(doc, job):
    return doc["jobs"][job]["steps"]


def forge_only(step):
    return step.get("name", "").endswith(("(gitea)", "(github)"))


def shared(doc, job):
    """The steps of `job` both files must have, as the fields that are compared."""
    return [
        {k: s[k] for k in TWIN_FIELDS if k in s}
        for s in steps(doc, job)
        if "uses" not in s and not forge_only(s)
    ]


def runs(doc, job):
    return [s["run"] for s in steps(doc, job) if "run" in s]


class Twins(unittest.TestCase):
    def test_the_name_and_the_gate_job_are_fixed(self):
        # The required check is `ci / ci-ok (pull_request)`: renaming either silently unrequires it.
        for doc in (GITEA, GITHUB):
            self.assertEqual(doc["name"], "ci")
            self.assertIn("ci-ok", doc["jobs"])

    def test_the_shared_jobs_have_the_same_steps_in_the_same_order(self):
        for job in TWIN_JOBS:
            self.assertEqual(shared(GITEA, job), shared(GITHUB, job), job)

    def test_both_files_have_the_shared_jobs(self):
        for job in TWIN_JOBS:
            self.assertIn(job, GITEA["jobs"])
            self.assertIn(job, GITHUB["jobs"])

    def test_the_jobs_that_exist_in_one_file_only(self):
        self.assertEqual(set(GITEA["jobs"]) - set(GITHUB["jobs"]), {"workflows", "device", "windows"})
        self.assertEqual(set(GITHUB["jobs"]) - set(GITEA["jobs"]), set())

    def test_a_forge_only_step_is_in_its_own_forge(self):
        for doc, other in ((GITEA, "(github)"), (GITHUB, "(gitea)")):
            for job, body in doc["jobs"].items():
                for s in body["steps"]:
                    self.assertFalse(s.get("name", "").endswith(other), f"{job}: {s.get('name')}")

    def test_the_github_only_steps_never_reach_gitea(self):
        # Dropping the preinstalled stable toolchain would uninstall it from the rustup directory the runners share.
        text = json.dumps(GITEA)
        for banned in ("Drop the preinstalled stable", "rustup toolchain uninstall", "rust-cache", "builtin:"):
            self.assertNotIn(banned, text)

    def test_the_changes_outputs_are_the_same_set_and_every_use_is_declared(self):
        self.assertEqual(set(GITEA["jobs"]["changes"]["outputs"]), set(GITHUB["jobs"]["changes"]["outputs"]))
        for doc in (GITEA, GITHUB):
            declared = set(doc["jobs"]["changes"]["outputs"])
            used = set(CHANGES_OUTPUT.findall(json.dumps(doc["jobs"])))
            self.assertLessEqual(used, declared)

    def test_every_crate_has_an_output(self):
        # Every workspace crate the script names has the output a job could read.
        crates = {
            "euhedral_" + n for n in
            ["data_structures", "hashing", "hardware_utils", "core", "graph", "io", "memory", "net", "gpu", "benchmarks"]
        }
        for doc in (GITEA, GITHUB):
            self.assertTrue(crates <= set(doc["jobs"]["changes"]["outputs"]))

    def test_the_script_tests_run_in_both_files(self):
        for doc in (GITEA, GITHUB):
            text = "\n".join(runs(doc, "changes"))
            for test in ["test_affected", "test_miri_build", "test_workflow_twins", "test_device_check"]:
                self.assertIn(f"scripts/ci/{test}.py", text)

    def test_a_push_diffs_against_the_commit_before_it(self):
        # `before` is the commit main was at, so this is the base; no merge base (main only moves forward). An all-zero
        # `before` (a new branch) and a manual run with no base run everything.
        for doc in (GITEA, GITHUB):
            step = next(s for s in steps(doc, "changes") if s.get("id") == "affected")
            self.assertEqual(step["env"]["BEFORE"], "${{ github.event.before }}")
            run = step["run"]
            self.assertIn('push)', run)
            self.assertIn('--base "$BEFORE" --head HEAD --github-output', run)
            self.assertNotIn('--base "$BEFORE" --head HEAD --merge-base', run)
            self.assertIn('--base "origin/$BASE_REF" --head HEAD --merge-base', run)
            self.assertIn("--all", run)

    def test_the_ratchet_runs_when_any_crate_changes(self):
        for doc in (GITEA, GITHUB):
            ratchet = [s for s in steps(doc, "workspace") if "hot_fix_ratchet" in s.get("run", "")]
            self.assertEqual(len(ratchet), 1)
            self.assertEqual(ratchet[0]["if"], "needs.changes.outputs.euhedral_core != 'true'")

    def test_the_matrix_reaches_the_build_through_the_environment(self):
        # A shard's commands can hold an apostrophe: interpolated into a script it would end a quote.
        for doc in (GITEA, GITHUB):
            self.assertEqual(doc["jobs"]["miri-build"]["env"]["MATRIX"], "${{ needs.changes.outputs.miri_matrix }}")
            for job, body in doc["jobs"].items():
                for s in body["steps"]:
                    self.assertNotIn("miri_matrix", s.get("run", ""), job)

    def test_the_miri_matrix_is_dynamic_and_guarded(self):
        miri = GITEA["jobs"]["miri"]
        self.assertEqual(miri["strategy"]["matrix"], "${{ fromJSON(needs.changes.outputs.miri_matrix) }}")
        self.assertIs(miri["strategy"]["fail-fast"], False)
        # An empty matrix fails the job, so the job must be skipped before the matrix expands.
        self.assertEqual(miri["if"], "needs.changes.outputs.miri == 'true'")
        self.assertEqual(miri["needs"], ["changes", "miri-build"])


class Gitea(unittest.TestCase):
    def test_it_runs_for_pull_requests_and_pushes_to_main(self):
        on = GITEA["on"]
        self.assertIn("pull_request", on)
        self.assertEqual(on["push"], {"branches": ["main"]})
        self.assertIn("workflow_dispatch", on)

    def test_a_pull_request_run_is_cancelled_by_a_newer_push_and_main_is_not(self):
        self.assertEqual(GITEA["concurrency"]["cancel-in-progress"], "${{ github.event_name == 'pull_request' }}")

    def test_a_change_that_affects_nothing_still_passes(self):
        # Every job but the first and the gate is gated on what `changes` found (or is the workflow lint).
        for job, body in GITEA["jobs"].items():
            if job in ("changes", "ci-ok"):
                continue
            if job == "workflows":
                self.assertEqual(body["if"], "github.event_name == 'pull_request'")
            else:
                self.assertIn("needs.changes.outputs", body["if"], job)
                self.assertIn("changes", body["needs"], job)
        ok = GITEA["jobs"]["ci-ok"]
        self.assertEqual(ok["if"], "always()")
        self.assertEqual(set(ok["needs"]), set(GITEA["jobs"]) - {"ci-ok"})

    def test_ci_ok_fails_on_failure_or_cancelled_only(self):
        step = steps(GITEA, "ci-ok")[0]
        self.assertEqual(step["env"]["RESULTS"], "${{ join(needs.*.result, ' ') }}")
        self.assertIn('*" failure "*|*" cancelled "*) exit 1', step["run"])
        self.assertNotIn("skipped", step["run"])

    def test_an_affected_crate_gets_the_workspace_job(self):
        self.assertEqual(GITEA["jobs"]["workspace"]["if"], "needs.changes.outputs.any == 'true'")
        self.assertEqual(GITEA["jobs"]["workspace"]["needs"], "changes")

    def test_the_device_job_runs_on_a_shared_grant_of_the_card(self):
        device = GITEA["jobs"]["device"]
        self.assertEqual(device["runs-on"], "arbiter-gpu")
        self.assertRegex(device["env"]["ARBITER_ARGS"], r"--gpu shared:\d+")
        self.assertNotIn("device", GITHUB["jobs"])
        for crate in ("euhedral-memory", "euhedral-gpu"):
            key = crate.replace("-", "_")
            step = next(s for s in device["steps"] if f"-p {crate} --features cuda" in s.get("run", ""))
            self.assertEqual(step["if"], f"needs.changes.outputs.{key} == 'true'")
            self.assertIn("set -o pipefail", step["run"])
            self.assertIn(f"cargo test -p {crate} --features cuda -- --nocapture 2>&1 | tee device-{crate}.log", step["run"])
            self.assertIn(f"device_check.py device-{crate}.log {crate}/tests", step["run"])
        # The workspace job never runs the device tests: it has no card, and a skip there would pass.
        for run in runs(GITEA, "workspace"):
            for line in run.splitlines():
                self.assertFalse("--features cuda" in line and "--nocapture" in line, line)
        nvrtc = next(s for s in device["steps"] if s.get("name") == "NVRTC (gitea)")
        self.assertIn("EUHEDRAL_NVRTC_DIR", nvrtc["run"])
        self.assertEqual(device["env"]["EUHEDRAL_NVRTC_DIR"], "${{ vars.EUHEDRAL_NVRTC_DIR }}")

    def test_no_device_crate_no_device_job(self):
        condition = GITEA["jobs"]["device"]["if"]
        self.assertEqual(sorted(CHANGES_OUTPUT.findall(condition)), ["euhedral_gpu", "euhedral_memory"])
        self.assertNotIn("&&", condition)

    def test_every_docker_job_asks_for_what_it_uses(self):
        for job, body in GITEA["jobs"].items():
            if body["runs-on"] not in DOCKER_LABELS:
                self.fail(f"{job} runs on {body['runs-on']}, not one of the project's labels")
            if body.get("env", {}).get("ARBITER") == "off":
                continue
            env = body["env"]
            self.assertRegex(env["ARBITER_ARGS"], r"^--threads \d+ --mem \d+[MG]( --gpu shared:\d+)?$", job)
            self.assertLessEqual(int(env["ARBITER_EST"]), 600, job)

    def test_the_labels_that_carry_the_seccomp_and_the_card(self):
        # io_uring needs seccomp unconfined: the jobs that build and run the tests are on the unconfined label.
        expected = {
            "changes": "arbiter-docker", "workflows": "arbiter-docker", "ci-ok": "arbiter-docker",
            "windows": "arbiter-docker", "workspace": "arbiter-docker-unconfined",
            "miri-build": "arbiter-docker-unconfined", "miri": "arbiter-docker-unconfined", "device": "arbiter-gpu",
        }
        self.assertEqual({j: b["runs-on"] for j, b in GITEA["jobs"].items()}, expected)

    def test_the_shard_lease_covers_the_largest_shard(self):
        # The lease is the job's, set before the matrix expands, so it cannot follow a shard: it is the largest one's.
        with open(os.path.join(HERE, "miri-shards.json")) as f:
            shards = json.load(f)

        def gigabytes(size):
            return int(size[:-1]) / (1024 if size[-1] == "M" else 1)

        most = max((s.get("mem", affected.MIRI_MEM) for s in shards), key=gigabytes)
        args = GITEA["jobs"]["miri"]["env"]["ARBITER_ARGS"]
        self.assertEqual(args, f"--threads {affected.SEEDS_PER_JOB} --mem {most}")
        self.assertNotIn("matrix.", args)
        self.assertNotIn("matrix.", GITEA["jobs"]["miri"]["env"]["ARBITER_EST"])

    def test_a_docker_job_has_no_timeout_of_its_own(self):
        # The runner holds a docker job to its lease; only the proxy job, which waits outside the arbiter, has one.
        for job, body in GITEA["jobs"].items():
            if job == "windows":
                self.assertEqual(body["timeout-minutes"], 100)
            else:
                self.assertNotIn("timeout-minutes", body, job)

    def test_the_windows_job_waits_outside_the_arbiter(self):
        windows = GITEA["jobs"]["windows"]
        self.assertEqual(windows["env"]["ARBITER"], "off")
        self.assertEqual(windows["runs-on"], "arbiter-docker")
        self.assertEqual(
            windows["if"], "github.event_name == 'pull_request' && needs.changes.outputs.any == 'true'"
        )
        proxy = next(s for s in windows["steps"] if "github-actions-proxy" in s.get("uses", ""))
        self.assertTrue(proxy["uses"].endswith("/euhedral/ci/github-actions-proxy@v4"))
        self.assertEqual(proxy["with"]["inputs"], "only_os=windows-2025\nbase_ref=${{ github.base_ref }}\n")
        self.assertEqual(proxy["with"]["workflow"], "ci.yml")

    def test_the_apps_key_is_read_only_by_the_windows_job(self):
        # Replaces the plan job's check that no GitHub App variable is in a job that runs a pull request's own code.
        for doc, name in ((GITEA, "gitea"), (GITHUB, "github")):
            top = {k: v for k, v in doc.items() if k != "jobs"}
            self.assertNotIn("secrets.", json.dumps(top), name)
            for job, body in doc["jobs"].items():
                if name == "gitea" and job == "windows":
                    self.assertIn("secrets.GH_APP_PRIVATE_KEY_B64", json.dumps(body))
                else:
                    self.assertNotIn("secrets.", json.dumps(body), f"{name}: {job}")

    def test_every_action_is_one_the_forge_can_run(self):
        allowed = {
            "actions/checkout@v4", "actions/upload-artifact@v3", "actions/download-artifact@v3",
            "http://localhost:8929/euhedral/ci/github-actions-proxy@v4",
        }
        used = {s["uses"] for body in GITEA["jobs"].values() for s in body["steps"] if "uses" in s}
        self.assertLessEqual(used, allowed)

    def test_the_toolchain_steps_export_mise_to_the_later_steps(self):
        for job in ("workspace", "device"):
            step = next(s for s in steps(GITEA, job) if s.get("name") == "Install the toolchain pinned in mise.toml (gitea)")
            for needle in ("mise install rust", "GITHUB_ENV", "GITHUB_PATH", "RUSTUP_TOOLCHAIN", "CARGO_HOME"):
                self.assertIn(needle, step["run"], job)
        # And both copies are the same text.
        a, b = (next(s for s in steps(GITEA, j) if "mise.toml (gitea)" in s.get("name", "")) for j in ("workspace", "device"))
        self.assertEqual(a["run"], b["run"])


if __name__ == "__main__":
    unittest.main()
