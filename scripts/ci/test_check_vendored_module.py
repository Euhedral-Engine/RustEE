import base64
import contextlib
import hashlib
import io
import json
import os
import shutil
import tarfile
import tempfile
import unittest

import check_vendored_module as cvm

VERSION = "0.0.0-20260101000000"
FILES = {
    "MODULE.bazel": b'module(name = "euhedral_build")\n',
    "platforms/BUILD.bazel": b"a\nb\n",
}


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def make_world(tmp, extra_members=()):
    """A registry directory holding FILES at VERSION (plus `extra_members`, TarInfo/bytes pairs) and a repo whose
    build/module is the same tree."""
    registry, root = os.path.join(tmp, "registry"), os.path.join(tmp, "repo")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in FILES.items():
            info = tarfile.TarInfo("euhedral_build/" + name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
        for info, data in extra_members:
            tar.addfile(info, io.BytesIO(data) if data is not None else None)
    blob = buf.getvalue()
    vdir = os.path.join(registry, "modules", "euhedral_build", VERSION)
    write(os.path.join(vdir, "source.tar.gz"), blob)
    integrity = "sha256-" + base64.b64encode(hashlib.sha256(blob).digest()).decode()
    write(os.path.join(vdir, "source.json"),
          json.dumps({"integrity": integrity, "strip_prefix": "euhedral_build"}).encode())
    write(os.path.join(root, "MODULE.bazel"),
          b'bazel_dep(name = "euhedral_build", version = "%s")\n' % VERSION.encode())
    for name, data in FILES.items():
        write(os.path.join(root, "build", "module", name), data)
    return registry, root


class CheckVendoredModule(unittest.TestCase):
    def check(self, mutate=None, extra_members=()):
        """(exit code, what it printed or said): the check against a fresh world after `mutate(root, registry)`."""
        with tempfile.TemporaryDirectory() as tmp:
            registry, root = make_world(tmp, extra_members)
            if mutate:
                mutate(root, registry)
            out = io.StringIO()
            try:
                with contextlib.redirect_stdout(out):
                    code = cvm.main(["x", registry, root])
            except SystemExit as e:
                return 1, str(e.code)
            return code, out.getvalue()

    def module(self, root, *parts):
        return os.path.join(root, "build", "module", *parts)

    def test_a_matching_copy_passes(self):
        self.assertEqual(self.check()[0], 0)

    def test_one_byte_of_drift_fails(self):
        def drift(root, registry):
            with open(self.module(root, "MODULE.bazel"), "ab") as f:
                f.write(b" ")
        code, said = self.check(drift)
        self.assertEqual(code, 1)
        self.assertIn("differs: MODULE.bazel", said)

    def test_an_added_file_fails(self):
        code, said = self.check(lambda root, registry: write(self.module(root, "new.bzl"), b""))
        self.assertEqual(code, 1)
        self.assertIn("only in build/module: new.bzl", said)

    def test_a_missing_file_fails(self):
        code, said = self.check(lambda root, registry: os.remove(self.module(root, "platforms", "BUILD.bazel")))
        self.assertEqual(code, 1)
        self.assertIn("only in the published module: platforms/BUILD.bazel", said)

    def test_an_added_executable_bit_fails(self):
        code, said = self.check(lambda root, registry: os.chmod(self.module(root, "MODULE.bazel"), 0o755))
        self.assertEqual(code, 1)
        self.assertIn("executable bit differs: MODULE.bazel", said)

    def test_a_published_symlink_fails(self):
        link = tarfile.TarInfo("euhedral_build/link")
        link.type, link.linkname = tarfile.SYMTYPE, "MODULE.bazel"
        code, said = self.check(extra_members=[(link, None)])
        self.assertEqual(code, 1)
        self.assertIn("not a regular file or a directory", said)

    def test_a_symlinked_file_under_build_module_fails(self):
        code, said = self.check(lambda root, registry: os.symlink("MODULE.bazel", self.module(root, "link")))
        self.assertEqual(code, 1)
        self.assertIn("link is a symlink", said)

    def test_a_symlinked_directory_under_build_module_fails(self):
        def link_dir(root, registry):
            os.symlink("platforms", self.module(root, "other"))
        code, said = self.check(link_dir)
        self.assertEqual(code, 1)
        self.assertIn("other is a symlink", said)

    def test_a_tampered_archive_fails_on_integrity(self):
        def tamper(root, registry):
            with open(os.path.join(registry, "modules", "euhedral_build", VERSION, "source.tar.gz"), "ab") as f:
                f.write(b"\0")
        code, said = self.check(tamper)
        self.assertEqual(code, 1)
        self.assertIn("source.json says", said)

    def test_a_version_the_registry_lacks_fails(self):
        def other(root, registry):
            write(os.path.join(root, "MODULE.bazel"), b'bazel_dep(name = "euhedral_build", version = "0.0.0-20250101000000")\n')
        code, said = self.check(other)
        self.assertEqual(code, 1)
        self.assertIn("0.0.0-20250101000000 is not in the registry", said)

    def test_a_missing_registry_fails(self):
        code, said = self.check(lambda root, registry: shutil.rmtree(registry))
        self.assertEqual(code, 1)
        self.assertIn("is not in the registry", said)


if __name__ == "__main__":
    unittest.main()
