#!/usr/bin/env python3
"""Fail unless build/module is the published euhedral_build module the root MODULE.bazel names.

Reads the version from the bazel_dep of MODULE.bazel, finds that version's archive in the registry directory,
checks the archive against the integrity in its source.json, and compares the archive's tree with build/module
byte for byte. The one allowed difference is a comment in platforms/BUILD.bazel (EXCEPTIONS).
Usage: check_vendored_module.py [registry dir (default /registry)] [repo root (default: this repo)]
"""
import base64
import hashlib
import io
import json
import os
import re
import sys
import tarfile

MODULE = "euhedral_build"
# path -> (line in the published file, line in the vendored copy). The published comment names a program that uses
# the build, which RustEE's files never do.
EXCEPTIONS = {
    "platforms/BUILD.bazel": (
        b"# The Quest headset (MySphereVR, phase 6).",
        b"# The Quest headset (a later repo's phase).",
    ),
}


def declared_version(root):
    with open(os.path.join(root, "MODULE.bazel"), encoding="utf-8") as f:
        text = f.read()
    found = re.findall(r'bazel_dep\(\s*name\s*=\s*"%s"\s*,\s*version\s*=\s*"([^"]+)"' % MODULE, text)
    if len(found) != 1:
        raise SystemExit("MODULE.bazel must have exactly one bazel_dep on %s with a version" % MODULE)
    return found[0]


def published_tree(registry, version):
    """The files of the published archive, {relative path: bytes}, after checking its integrity."""
    vdir = os.path.join(registry, "modules", MODULE, version)
    if not os.path.isdir(vdir):
        raise SystemExit("%s %s is not in the registry at %s" % (MODULE, version, registry))
    with open(os.path.join(vdir, "source.json"), encoding="utf-8") as f:
        source = json.load(f)
    with open(os.path.join(vdir, "source.tar.gz"), "rb") as f:
        blob = f.read()
    algo, _, want = source["integrity"].partition("-")
    if algo != "sha256":
        raise SystemExit("integrity is not sha256: %s" % source["integrity"])
    have = base64.b64encode(hashlib.sha256(blob).digest()).decode()
    if have != want:
        raise SystemExit("the archive's sha256 is %s, source.json says %s" % (have, want))
    prefix = source["strip_prefix"].rstrip("/") + "/"
    files = {}
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            if not name.startswith(prefix) or ".." in name.split("/"):
                raise SystemExit("unexpected archive entry %s" % name)
            files[name[len(prefix):]] = tar.extractfile(member).read()
    return files


def vendored_tree(root):
    base = os.path.join(root, "build", "module")
    files = {}
    for here, _, names in os.walk(base):
        for name in names:
            path = os.path.join(here, name)
            with open(path, "rb") as f:
                files[os.path.relpath(path, base).replace(os.sep, "/")] = f.read()
    return files


def differences(published, vendored):
    published = dict(published)
    for path, (old, new) in EXCEPTIONS.items():
        data = published.get(path)
        if data is not None:
            if old not in data.splitlines():
                raise SystemExit("%s: the published file lacks the line the exception replaces" % path)
            published[path] = data.replace(old, new)
    out = ["only in the published module: %s" % p for p in sorted(set(published) - set(vendored))]
    out += ["only in build/module: %s" % p for p in sorted(set(vendored) - set(published))]
    out += ["differs: %s" % p for p in sorted(set(published) & set(vendored)) if published[p] != vendored[p]]
    return out


def main(argv):
    registry = argv[1] if len(argv) > 1 else "/registry"
    root = argv[2] if len(argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    version = declared_version(root)
    diff = differences(published_tree(registry, version), vendored_tree(root))
    if diff:
        print("build/module is not the published %s %s:" % (MODULE, version))
        print("\n".join("  " + d for d in diff))
        return 1
    print("build/module is the published %s %s" % (MODULE, version))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
