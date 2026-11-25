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
    write(os.path.join(vdir, "MODULE.bazel"), FILES["MODULE.bazel"])
    write(os.path.join(registry, "modules", "euhedral_build", "metadata.json"),
          json.dumps({"versions": [VERSION], "yanked_versions": {}}).encode())
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

    def registry_file(self, registry, *parts):
        return os.path.join(registry, "modules", "euhedral_build", *parts)

    def edit_source(self, **changes):
        def edit(root, registry):
            path = self.registry_file(registry, VERSION, "source.json")
            with open(path, encoding="utf-8") as f:
                source = json.load(f)
            source.update(changes)
            write(path, json.dumps(source).encode())
        return edit

    def test_patches_in_source_json_fail(self):
        code, said = self.check(self.edit_source(patches={"a.patch": "sha256-x"}, patch_strip=1))
        self.assertEqual(code, 1)
        self.assertIn("patch_strip, patches", said)

    def test_an_overlay_in_source_json_fails(self):
        code, said = self.check(self.edit_source(overlay={"BUILD": "sha256-x"}))
        self.assertEqual(code, 1)
        self.assertIn("overlay", said)

    def test_an_unknown_key_in_source_json_fails(self):
        code, said = self.check(self.edit_source(archive_type="zip"))
        self.assertEqual(code, 1)
        self.assertIn("archive_type", said)

    def test_a_different_strip_prefix_fails(self):
        code, said = self.check(self.edit_source(strip_prefix="other"))
        self.assertEqual(code, 1)
        self.assertIn("strip_prefix", said)

    def test_a_registry_module_with_an_added_dependency_fails(self):
        def add(root, registry):
            with open(self.registry_file(registry, VERSION, "MODULE.bazel"), "ab") as f:
                f.write(b'bazel_dep(name = "extra", version = "1.0")\n')
        code, said = self.check(add)
        self.assertEqual(code, 1)
        self.assertIn("beyond the module() version", said)

    def test_a_registry_module_that_differs_in_another_module_argument_fails(self):
        def rename(root, registry):
            write(self.registry_file(registry, VERSION, "MODULE.bazel"),
                  b'module(name = "euhedral_build", version = "1", compatibility_level = 2)\n')
            write(self.module(root, "MODULE.bazel"), b'module(name = "euhedral_build", version = "2")\n')
        code, said = self.check(rename)
        self.assertEqual(code, 1)
        self.assertIn("beyond the module() version", said)

    def test_a_registry_module_differing_only_in_the_module_version_passes(self):
        def versions(root, registry):
            # build/module's own MODULE.bazel is the archive's too, so only the registry's copy and the tree move together.
            body = b'# note\nmodule(name = "euhedral_build", version = "%s")\n\nbazel_dep(name = "platforms", version = "1.1.0")\n'
            write(self.module(root, "MODULE.bazel"), body % b"0.1.0")
            write(self.registry_file(registry, VERSION, "MODULE.bazel"), body % b"0.0.0-x+abc")
            archive = self.registry_file(registry, VERSION, "source.tar.gz")
            with tarfile.open(archive) as tar:
                members = [(m, tar.extractfile(m).read() if m.isfile() else None) for m in tar]
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                for m, data in members:
                    if m.name == "euhedral_build/MODULE.bazel":
                        data, m.size = body % b"0.1.0", len(body % b"0.1.0")
                    tar.addfile(m, io.BytesIO(data) if data is not None else None)
            blob = buf.getvalue()
            write(archive, blob)
            integrity = "sha256-" + base64.b64encode(hashlib.sha256(blob).digest()).decode()
            write(self.registry_file(registry, VERSION, "source.json"),
                  json.dumps({"integrity": integrity, "strip_prefix": "euhedral_build"}).encode())
        self.assertEqual(self.check(versions)[0], 0)

    def test_a_missing_registry_module_fails(self):
        code, said = self.check(lambda root, registry: os.remove(self.registry_file(registry, VERSION, "MODULE.bazel")))
        self.assertEqual(code, 1)
        self.assertIn("has no MODULE.bazel", said)

    def test_a_yanked_version_fails(self):
        def yank(root, registry):
            write(self.registry_file(registry, "metadata.json"),
                  json.dumps({"versions": [VERSION], "yanked_versions": {VERSION: "broken build"}}).encode())
        code, said = self.check(yank)
        self.assertEqual(code, 1)
        self.assertIn("is yanked: broken build", said)

    def test_a_version_missing_from_metadata_fails(self):
        def drop(root, registry):
            write(self.registry_file(registry, "metadata.json"), json.dumps({"versions": [], "yanked_versions": {}}).encode())
        code, said = self.check(drop)
        self.assertEqual(code, 1)
        self.assertIn("not in metadata.json's versions", said)


if __name__ == "__main__":
    unittest.main()
