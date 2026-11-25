import base64
import hashlib
import io
import json
import os
import tarfile
import tempfile
import unittest

import check_vendored_module as cvm

VERSION = "0.0.0-20260101000000"
OLD_LINE = b"# a made-up published line"
cvm.EXCEPTIONS = {"platforms/BUILD.bazel": (hashlib.sha256(OLD_LINE).hexdigest(), b"# a made-up vendored line")}
FILES = {
    "MODULE.bazel": b'module(name = "euhedral_build")\n',
    "platforms/BUILD.bazel": b"a\n" + OLD_LINE + b"\nb\n",
}


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def make_world(tmp, files=FILES):
    registry, root = os.path.join(tmp, "registry"), os.path.join(tmp, "repo")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo("euhedral_build/" + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    blob = buf.getvalue()
    vdir = os.path.join(registry, "modules", "euhedral_build", VERSION)
    write(os.path.join(vdir, "source.tar.gz"), blob)
    integrity = "sha256-" + base64.b64encode(hashlib.sha256(blob).digest()).decode()
    write(os.path.join(vdir, "source.json"),
          json.dumps({"integrity": integrity, "strip_prefix": "euhedral_build"}).encode())
    write(os.path.join(root, "MODULE.bazel"),
          b'bazel_dep(name = "euhedral_build", version = "%s")\n' % VERSION.encode())
    for name, data in files.items():
        if name == "platforms/BUILD.bazel":
            data = data.replace(OLD_LINE, cvm.EXCEPTIONS[name][1])
        write(os.path.join(root, "build", "module", name), data)
    return registry, root


class CheckVendoredModule(unittest.TestCase):
    def run_check(self, mutate=None):
        with tempfile.TemporaryDirectory() as tmp:
            registry, root = make_world(tmp)
            if mutate:
                mutate(root, registry)
            return cvm.main(["x", registry, root])

    def test_a_matching_copy_passes(self):
        self.assertEqual(self.run_check(), 0)

    def test_one_byte_of_drift_fails(self):
        def drift(root, registry):
            path = os.path.join(root, "build", "module", "MODULE.bazel")
            with open(path, "ab") as f:
                f.write(b" ")
        self.assertEqual(self.run_check(drift), 1)

    def test_a_tampered_archive_fails_on_integrity(self):
        def tamper(root, registry):
            path = os.path.join(registry, "modules", "euhedral_build", VERSION, "source.tar.gz")
            with open(path, "ab") as f:
                f.write(b"\0")
        with self.assertRaises(SystemExit):
            self.run_check(tamper)

    def test_a_version_the_registry_lacks_fails(self):
        def other(root, registry):
            path = os.path.join(root, "MODULE.bazel")
            with open(path, "w") as f:
                f.write('bazel_dep(name = "euhedral_build", version = "0.0.0-20250101000000")\n')
        with self.assertRaises(SystemExit):
            self.run_check(other)


if __name__ == "__main__":
    unittest.main()
