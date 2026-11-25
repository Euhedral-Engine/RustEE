#!/usr/bin/env python3
"""Pins .gitea/workflows/ci.yml and .github/workflows/ci.yml to each other, and the Gitea one to what its jobs promise.
Run: python3 scripts/ci/test_workflow_twins.py (needs PyYAML).

The two files are twins: for the jobs both have, the steps that are not forge-only are the same, key for key and in
order. A forge-only step has a name that ends in `(gitea)` or `(github)`; a step that `uses` an action may differ, since
the actions differ (checkout, caches, artifacts). The `ci-ok` needs list may differ. Everything else a job promises
(which jobs a change runs, the leases, the device job, the one place a secret is read) is pinned for the Gitea file below.
The Bazel jobs are pinned in `Bazel`: `bazel test //... --config=ci` and forge's C7 in both files, on the bazel label with no
token on Gitea, forge's repo check fetched from forge's main, all of it in `ci-ok`, and every Bazel input running them.
Gitea's `windows` job is the one job that calls a reusable workflow (`CALLS`): euhedral/ci's Windows workflow at the tag v5,
which checks out with the system git and runs no `actions/checkout`; it gets the GitHub App's three secrets by name, and on
GitHub its dispatch runs the workspace job's Windows leg and `bazel-windows`, the Bazel test on windows_x86_64.
"""

import json
import os
import re
import subprocess
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
TWIN_JOBS = ["changes", "workspace", "bazel", "miri-build", "miri", "ci-ok"]
DOCKER_LABELS = {"arbiter-docker", "arbiter-docker-unconfined", "arbiter-docker-bazel", "arbiter-gpu"}
SECRETS = re.compile(r"\bsecrets\b")  # `secrets.X`, `toJSON(secrets)` and `secrets[...]` alike
CHANGES_OUTPUT = re.compile(r"needs\.changes\.outputs\.(\w+)")
# The Gitea jobs that call a reusable workflow, and the workflow each calls: every other job has its own steps.
CALLS = {"windows": "euhedral/ci/.gitea/workflows/bazel-windows.yml@refs/tags/v5"}
APP_SECRETS = ["GH_APP_ID", "GH_APP_INSTALLATION_ID", "GH_APP_PRIVATE_KEY_B64"]
BAZEL_WINDOWS = "bazel test //... --config=ci --platforms=@euhedral_build//platforms:windows_x86_64"
SHORT_PATHS = """{
  echo "startup --output_base=C:/b"
  echo "common --experimental_platform_in_output_dir"
  echo "common --experimental_override_platform_cpu_name=@euhedral_build//platforms:windows_x86_64=w"
  echo "common --compilation_mode=dbg"
} > user.bazelrc
"""


def steps(doc, job):
    return doc["jobs"][job]["steps"]


def local_jobs(doc):
    """The jobs with steps of their own: all of them but the calls of a reusable workflow `CALLS` pins."""
    return {job: body for job, body in doc["jobs"].items() if job not in CALLS or doc is not GITEA}


def forge_only(step):
    return step.get("name", "").endswith(("(gitea)", "(github)"))


def shared(doc, job):
    """The steps of `job` both files must have, with every key."""
    return [
        dict(s)  # every key: a step that differs in any (`if`, `continue-on-error`, `with`, ...) is not a twin
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
        self.assertEqual(set(GITEA["jobs"]) - set(GITHUB["jobs"]), {"workflows", "check-repo", "device", "windows"})
        self.assertEqual(set(GITHUB["jobs"]) - set(GITEA["jobs"]), {"bazel-windows"})

    def test_a_forge_only_step_is_in_its_own_forge(self):
        for doc, other in ((GITEA, "(github)"), (GITHUB, "(gitea)")):
            for job, body in local_jobs(doc).items():
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
            for job, body in local_jobs(doc).items():
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

    def test_a_pull_request_run_is_cancelled_by_a_newer_push_and_a_push_to_main_is_not(self):
        self.assertEqual(GITEA["concurrency"]["cancel-in-progress"], "${{ github.event_name == 'pull_request' }}")
        # A push to main has a group of its own (its commit): a newer queued run must not replace a pending one.
        self.assertEqual(
            GITEA["concurrency"]["group"],
            "ci-${{ github.event_name == 'pull_request' && github.ref || github.sha }}",
        )

    def test_no_job_or_step_may_fail_without_failing_the_run(self):
        self.assertNotIn("continue-on-error", json.dumps(GITEA))

    def test_every_job_has_exactly_these_keys_and_conditions(self):
        keys = {
            "changes": {"runs-on", "env", "outputs", "steps"},
            "workflows": {"if", "runs-on", "env", "steps"},
            "workspace": {"needs", "if", "runs-on", "defaults", "env", "steps"},
            "check-repo": {"needs", "if", "runs-on", "defaults", "env", "steps"},
            "bazel": {"needs", "if", "runs-on", "defaults", "env", "steps"},
            "device": {"needs", "if", "runs-on", "defaults", "env", "steps"},
            "miri-build": {"needs", "if", "runs-on", "env", "steps"},
            "miri": {"name", "needs", "if", "strategy", "runs-on", "env", "steps"},
            "windows": {"needs", "if", "uses", "with", "secrets"},
            "ci-ok": {"needs", "if", "runs-on", "env", "steps"},
        }
        self.assertEqual({j: set(b) for j, b in GITEA["jobs"].items()}, keys)
        conditions = {
            "workflows": "github.event_name == 'pull_request'",
            "workspace": "needs.changes.outputs.any == 'true'",
            "check-repo": "needs.changes.outputs.any == 'true'",
            "bazel": "needs.changes.outputs.any == 'true'",
            "device": "needs.changes.outputs.euhedral_memory == 'true' || needs.changes.outputs.euhedral_gpu == 'true'",
            "miri-build": "needs.changes.outputs.miri == 'true'",
            "miri": "needs.changes.outputs.miri == 'true'",
            "windows": "github.event_name == 'pull_request' && needs.changes.outputs.any == 'true'",
            "ci-ok": "always()",
        }
        self.assertEqual({j: b["if"] for j, b in GITEA["jobs"].items() if "if" in b}, conditions)

    def test_ci_ok_fails_when_changes_published_nothing(self):
        # If `changes` passed but its outputs did not arrive, every later job is skipped: not a pass.
        for doc in (GITEA, GITHUB):
            self.assertIn("changes", doc["jobs"]["ci-ok"]["needs"])
            step = steps(doc, "ci-ok")[0]
            self.assertEqual(step["env"]["ANY"], "${{ needs.changes.outputs.any }}")
            self.assertRegex(step["run"], r'case "\$ANY" in\s+true\|false\) ;;\s+\*\) [^\n]*exit 1 ;;')

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
        for job, body in local_jobs(GITEA).items():
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
            "workspace": "arbiter-docker-unconfined",
            "miri-build": "arbiter-docker-unconfined", "miri": "arbiter-docker-unconfined", "device": "arbiter-gpu",
            "check-repo": "arbiter-docker", "bazel": "arbiter-docker-bazel",
        }
        self.assertEqual({j: b["runs-on"] for j, b in local_jobs(GITEA).items()}, expected)

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
        # The runner holds a docker job to its lease; the proxy job, which waits outside it, is the called workflow's.
        for job, body in local_jobs(GITEA).items():
            self.assertNotIn("timeout-minutes", body, job)

    def test_only_the_windows_job_calls_a_workflow_and_it_is_the_shared_one_at_v5(self):
        # The tag by its full ref: a branch named v5 would win a plain name.
        self.assertEqual({j: b["uses"] for j, b in GITEA["jobs"].items() if "uses" in b}, CALLS)
        for job, body in local_jobs(GITEA).items():
            self.assertIn("steps", body, job)
            self.assertIn("runs-on", body, job)

    def test_the_windows_job_runs_no_checkout_and_reads_no_forge_variable(self):
        # The called workflow checks out with the system git and takes the forge's address from the run: nothing of the
        # job runs `actions/checkout` (it starts `node` from the image's PATH) or reads FORGE_URL.
        windows = GITEA["jobs"]["windows"]
        self.assertNotIn("steps", windows)
        text = json.dumps(windows)
        for banned in ("actions/checkout", "vars.FORGE_URL", "FORGE_URL", "refs/tags/v4", "github-actions-proxy"):
            self.assertNotIn(banned, text)

    def test_the_windows_job_dispatches_ci_yml_for_pull_requests(self):
        windows = GITEA["jobs"]["windows"]
        self.assertEqual(windows["needs"], "changes")
        self.assertEqual(
            windows["if"], "github.event_name == 'pull_request' && needs.changes.outputs.any == 'true'"
        )
        # The called workflow sets `only_os=windows-2025` and `base_ref`; the branch it pushes carries the pull request's
        # workflow files, which a push may change only with `workflows:write`.
        self.assertEqual(windows["with"], {
            "repo": "Euhedral-Engine/RustEE",
            "workflow": "ci.yml",
            "permissions": "contents:write,actions:write,workflows:write",
        })
        self.assertIn("windows", GITEA["jobs"]["ci-ok"]["needs"])

    def test_the_github_dispatch_runs_cargo_and_bazel_on_windows(self):
        # `only_os=windows-2025` keeps the workspace job's Windows leg and runs the Bazel test on the Windows platform.
        matrix = GITHUB["jobs"]["workspace"]["strategy"]["matrix"]["os"]
        self.assertIn("inputs.only_os", matrix)
        job = GITHUB["jobs"]["bazel-windows"]
        self.assertEqual(job["runs-on"], "windows-2025")
        self.assertEqual(job["needs"], "changes")
        self.assertEqual(job["if"], "needs.changes.outputs.any == 'true' && inputs.only_os != 'ubuntu-latest'")
        self.assertEqual(job["defaults"], {"run": {"shell": "bash"}})
        # The output paths are short so that link.exe can open the doc tests' inputs (it takes no path of 260 characters).
        self.assertEqual(steps(GITHUB, "bazel-windows")[1:], [
            step(GITHUB, "bazel", "Bazel is the version in .bazelversion (github)"),
            {"name": "Short output paths (github)", "run": SHORT_PATHS},
            {"name": "Bazel test on the Windows platform", "env": {"MSYS_NO_PATHCONV": "1"}, "run": BAZEL_WINDOWS},
            {"name": "Stop the Bazel server", "if": "always()", "run": "bazel shutdown"},
        ])
        self.assertEqual(steps(GITHUB, "bazel-windows")[0], {"uses": "actions/checkout@v4"})
        self.assertIn("bazel-windows", GITHUB["jobs"]["ci-ok"]["needs"])

    def test_no_host_or_port_is_written_in_either_workflow(self):
        for path in (".gitea/workflows/ci.yml", ".github/workflows/ci.yml"):
            with open(os.path.join(ROOT, path)) as f:
                text = f.read()
            self.assertNotIn("localhost", text, path)
            # No URL at all: a host, with or without a port, is the forge's to say (an allowed one is listed here).
            allowed = []
            self.assertEqual([u for u in re.findall(r"[\w+.-]*://\S*", text) if u not in allowed], [], path)
            self.assertNotRegex(text, r":(8929|18929|2224|12224|3002|13002)\b", path)

    def test_the_clone_directory_is_no_tracked_path(self):
        out = subprocess.run(["git", "ls-files", ".ci"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        self.assertEqual(out.strip(), "")
        # affected.py reads the diff of tracked files only, so an untracked clone is never a changed file.

    def test_the_apps_key_is_read_only_by_the_windows_job(self):
        # Replaces the plan job's check that no GitHub App variable is in a job that runs a pull request's own code.
        for doc, name in ((GITEA, "gitea"), (GITHUB, "github")):
            top = {k: v for k, v in doc.items() if k != "jobs"}
            self.assertIsNone(SECRETS.search(json.dumps(top)), name)
            for job, body in doc["jobs"].items():
                if name == "gitea" and job == "windows":
                    # By name, each its own: `secrets: inherit` would give the called workflow every secret.
                    self.assertEqual(body["secrets"], {k: f"${{{{ secrets.{k} }}}}" for k in APP_SECRETS})
                    rest = {k: v for k, v in body.items() if k != "secrets"}
                    self.assertIsNone(SECRETS.search(json.dumps(rest)))
                else:
                    self.assertIsNone(SECRETS.search(json.dumps(body)), f"{name}: {job}")

    def test_every_action_is_one_the_forge_can_run(self):
        allowed = {
            "actions/checkout@v4", "actions/upload-artifact@v3", "actions/download-artifact@v3",
        }
        used = {s["uses"] for body in local_jobs(GITEA).values() for s in body["steps"] if "uses" in s}
        self.assertLessEqual(used, allowed)

    def test_the_toolchain_steps_export_mise_to_the_later_steps(self):
        for job in ("workspace", "device"):
            step = next(s for s in steps(GITEA, job) if s.get("name") == "Install the toolchain pinned in mise.toml (gitea)")
            for needle in ("mise install rust", "GITHUB_ENV", "GITHUB_PATH", "RUSTUP_TOOLCHAIN", "CARGO_HOME"):
                self.assertIn(needle, step["run"], job)
        # And both copies are the same text.
        a, b = (next(s for s in steps(GITEA, j) if "mise.toml (gitea)" in s.get("name", "")) for j in ("workspace", "device"))
        self.assertEqual(a["run"], b["run"])


# forge's C7 (build/scripts/check-repo.sh), copied: the bazel label has no token to fetch the script, so the `bazel` job runs
# this query itself and the `check-repo` job runs the rest with --skip-c7, after checking that forge's script still builds
# this query.
FORGE_C7 = 'attr(tags, "(\\[|, )gpu(,|\\])", //...) except attr(tags, "(\\[|, )external(,|\\])", //...)'
FORGE_CHECK = '"$RUNNER_TEMP/forge/build/scripts/check-repo.sh"'
SYSTEM_PATH = {"PATH": "/usr/bin:/bin"}  # the image's own tools: /cache, first on its PATH, is writable by earlier jobs
C7_RUN = (
    f"untagged=$(bazel query '{FORGE_C7}')\n"  # its own line, so a failed query fails the step (bash -e)
    'if [ -n "$untagged" ]; then\n'
    "  printf 'tagged gpu but not external:\\n%s\\n' \"$untagged\"\n"
    "  exit 1\n"
    "fi\n"
)
# The last four steps of both files' `bazel` job, exactly: a step-level `if`, an extra key or an edited line is a change.
BAZEL_STEPS = [
    {
        "name": "The crate lock is current",
        "run": 'kept=$(mktemp -d)\n'
        'cp cargo-bazel-lock.json MODULE.bazel.lock Cargo.lock "$kept"\n'
        'CARGO_BAZEL_REPIN=1 bazel mod deps --lockfile_mode=update\n'
        'for lock in cargo-bazel-lock.json MODULE.bazel.lock Cargo.lock; do\n'
        '  cmp "$kept/$lock" "$lock" || { echo "$lock is stale: run CARGO_BAZEL_REPIN=1 bazel mod deps '
        '--lockfile_mode=update and commit it"; exit 1; }\n'
        'done\n',
    },
    {"name": "The BUILD files are generated", "run": "bazel run //:gazelle --config=ci -- -mode=diff"},
    {"name": "Bazel test", "run": "bazel test //... --config=ci"},
    {"name": "Every gpu target is also external (C7)", "run": C7_RUN},
    # Whatever failed: the caches are collected only in an idle server, and a running one holds the lease.
    {"name": "Stop the Bazel server", "if": "always()", "run": "bazel shutdown"},
]


def step(doc, job, name):
    return next(s for s in steps(doc, job) if s.get("name") == name)


def render_forge_c7(lines):
    """The query that forge's two C7 lines (the tag pattern and the query) build, as bash would."""
    tag_line, query_line = lines
    pattern = re.fullmatch(r"""tag\(\) \{ printf '(.*)' "\$1"; \}""", tag_line).group(1).replace("\\\\", "\\")
    query = re.fullmatch(r'query "(.*)"', query_line).group(1).replace('\\"', '"')
    return re.sub(r"\$\(tag (\w+)\)", lambda m: pattern.replace("%s", m.group(1)), query)


class Bazel(unittest.TestCase):
    def test_both_files_run_exactly_the_bazel_tests_c7_and_the_shutdown(self):
        gitea = [{"name": "Copy the runner's checkout (gitea)", "run": "cp -a /checkout/. ."}] + BAZEL_STEPS
        github = [
            {"uses": "actions/checkout@v4"},
            {
                "name": "Bazel is the version in .bazelversion (github)",
                "run": 'have=$(bazel --version)\n'
                'test "$have" = "bazel $(cat .bazelversion)" || '
                '{ echo "bazel is \'$have\', .bazelversion says $(cat .bazelversion)"; exit 1; }\n',
            },
        ] + BAZEL_STEPS
        self.assertEqual(steps(GITEA, "bazel"), gitea)
        self.assertEqual(steps(GITHUB, "bazel"), github)
        for doc, name in ((GITEA, "gitea"), (GITHUB, "github")):
            self.assertEqual(doc["jobs"]["bazel"]["defaults"], {"run": {"shell": "bash"}}, name)
            self.assertIn("bazel", doc["jobs"]["ci-ok"]["needs"], name)

    def test_the_gitea_bazel_job_is_on_the_bazel_label_and_reads_no_token(self):
        job = GITEA["jobs"]["bazel"]
        self.assertEqual(job["runs-on"], "arbiter-docker-bazel")
        # The runner gives the job no token, so actions/checkout cannot work: the job copies the runner's checkout.
        self.assertEqual([s for s in job["steps"] if "uses" in s], [])
        text = json.dumps(job)
        self.assertIsNone(SECRETS.search(text))
        for token in ("github.token", "GITEA_TOKEN", "GITHUB_TOKEN", "vars."):
            self.assertNotIn(token, text)
        # The container rc's --local_resources=memory=5734 is 70 % of an 8G lease; no command passes another.
        self.assertEqual(job["env"]["ARBITER_ARGS"], "--threads 8 --mem 8G")
        self.assertNotIn("--local_resources", text)

    def test_bazel_in_ci_is_held_to_the_leases_threads(self):
        with open(os.path.join(ROOT, ".bazelrc")) as f:
            lines = [line.split() for line in f if line.strip() and not line.startswith("#")]
        cpu = [flag for line in lines if line[0] in ("build:ci", "test:ci", "common:ci") for flag in line[1:]
               if flag.startswith("--local_resources=cpu=")]
        threads = re.fullmatch(r"--threads (\d+) --mem \d+G", GITEA["jobs"]["bazel"]["env"]["ARBITER_ARGS"]).group(1)
        self.assertEqual(cpu, [f"--local_resources=cpu={threads}"])

    def test_the_github_bazel_job_does_not_run_on_a_one_os_run(self):
        job = GITHUB["jobs"]["bazel"]
        self.assertEqual(job["runs-on"], "ubuntu-latest")
        self.assertEqual(job["if"], "needs.changes.outputs.any == 'true' && (!inputs.only_os || inputs.only_os == 'all')")

    def test_the_repo_check_is_forges_main_and_fails_when_it_cannot_be_fetched(self):
        job = GITEA["jobs"]["check-repo"]
        self.assertEqual(job["runs-on"], "arbiter-docker")
        self.assertEqual(job["defaults"], {"run": {"shell": "bash"}})
        self.assertIn("check-repo", GITEA["jobs"]["ci-ok"]["needs"])
        self.assertNotIn("check-repo", GITHUB["jobs"])
        names = ["Fetch forge's main (gitea)", "forge's C7 is the query the bazel job runs (gitea)",
                 "forge's repo check, C7 aside (gitea)"]
        # The token is not left in the clone's .git/config.
        self.assertEqual(job["steps"][0], {"uses": "actions/checkout@v4", "with": {"persist-credentials": False}})
        self.assertEqual([s.get("name") for s in job["steps"][1:]], names)
        fetch, c7, check = job["steps"][1:]
        for s in (fetch, c7, check):
            self.assertEqual(set(s), {"name", "env", "run"}, s["name"])  # no `if`, no `continue-on-error`
        self.assertEqual(fetch["env"], {"FORGE_URL": "${{ vars.FORGE_URL }}", "FORGE_TOKEN": "${{ github.token }}"} | SYSTEM_PATH)
        self.assertEqual(c7["env"], SYSTEM_PATH)
        self.assertEqual(check["env"], SYSTEM_PATH)  # the token is the fetch's alone
        self.assertEqual(check["run"], f"bash {FORGE_CHECK} --skip-c7 .")
        run = fetch["run"]
        self.assertIn('"$FORGE_URL/euhedral/forge.git" refs/heads/main\n', run)  # main by its full ref, nothing else
        self.assertIn('git -C "$RUNNER_TEMP/forge" checkout -q --detach FETCH_HEAD', run)
        self.assertLess(run.index('rm -rf "$RUNNER_TEMP/forge"'), run.index('git init -q "$RUNNER_TEMP/forge"'))
        self.assertIn(f"test -f {FORGE_CHECK} ||", run)
        # The token is a header of this one command, sent to the forge only.
        self.assertIn('GIT_CONFIG_KEY_0="http.$FORGE_URL/.extraHeader"', run)
        self.assertIn("GIT_CONFIG_KEY_1=http.followRedirects GIT_CONFIG_VALUE_1=false", run)
        self.assertNotIn("git config", run)
        self.assertEqual(run.count("$FORGE_TOKEN"), 1)  # only into the header, never echoed

    def test_forges_c7_lines_build_the_query_the_bazel_job_runs(self):
        lines = [
            """tag() { printf '(\\\\[|, )%s(,|\\\\])' "$1"; }""",
            '''query "attr(tags, \\"$(tag gpu)\\", //...) except attr(tags, \\"$(tag external)\\", //...)"''',
        ]
        c7 = step(GITEA, "check-repo", "forge's C7 is the query the bazel job runs (gitea)")["run"]
        self.assertEqual(
            c7,
            "while IFS= read -r line; do\n"
            f'  grep -qF -- "$line" {FORGE_CHECK} || '
            '{ printf "forge\'s C7 no longer has: %s\\n" "$line"; exit 1; }\n'
            "done <<'C7'\n" + "\n".join(lines) + "\nC7\n",
        )
        self.assertEqual(render_forge_c7(lines), FORGE_C7)

    def test_no_job_runs_a_check_repo_script_of_the_checkout(self):
        # The code under test must not supply its own checker: every check-repo.sh named is the one fetched from forge.
        for doc, name in ((GITEA, "gitea"), (GITHUB, "github")):
            for job, body in local_jobs(doc).items():
                for s in body["steps"]:
                    run = s.get("run", "")
                    self.assertEqual(run.count("check-repo.sh"), run.count(FORGE_CHECK), f"{name}: {job}")

    def test_every_bazel_input_runs_the_bazel_jobs(self):
        # The jobs run when `any` is set: every Rust, Cargo, BUILD, MODULE, .bazel* and generator file sets it, and so
        # does a document that a crate's target reads (its BUILD file names it in `data`).
        audit = "docs/specs/audit.md"
        graph = affected.Graph(
            {"euhedral-core": "euhedral-core"},
            {"euhedral-core": set()},
            {audit: {"euhedral-core"}},
        )
        inputs = [
            "euhedral-core/src/lib.rs", "euhedral-core/tests/new_test.rs", "euhedral-core/examples/e.rs",
            "euhedral-core/benches/b.rs", "euhedral-core/build.rs", "euhedral-core/Cargo.toml", "euhedral-core/README.md",
            "euhedral-core/src/notes.md", "euhedral-core/BUILD.bazel", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml",
            "mise.toml", "BUILD.bazel", "MODULE.bazel", "MODULE.bazel.lock", "cargo-bazel-lock.json", ".bazelrc", ".bazelversion", ".bazelignore",
            ".gitattributes", "build/module/MODULE.bazel", "build/module/platforms/BUILD.bazel", "docs/x/BUILD.bazel",
            "bazel_gen.toml", "build/module/bazel-gen/src/lib.rs", ".gitea/workflows/ci.yml", audit,
        ]
        for f in inputs:
            crates, everything, reason = affected.affected(graph, [f])
            self.assertEqual(affected.outputs(graph, crates, everything, reason)["any"], "true", f)
        for doc in (GITEA, GITHUB):
            self.assertTrue(doc["jobs"]["bazel"]["if"].startswith("needs.changes.outputs.any == 'true'"))
        self.assertEqual(GITEA["jobs"]["check-repo"]["if"], "needs.changes.outputs.any == 'true'")


if __name__ == "__main__":
    unittest.main()
