#!/usr/bin/env python3
"""Fail unless build/module is the published euhedral_build module the root MODULE.bazel names.

Reads the version from the bazel_dep of MODULE.bazel, finds that version's archive in the registry directory,
checks the archive against the integrity in its source.json, and compares the archive's tree with build/module
byte for byte (and the executable bit of each file). Symlinks and other special files fail.
It also refuses a source.json that carries anything Bazel would apply besides the archive (patches, overlay, ...),
compares the registry's MODULE.bazel (the file Bazel resolves dependencies from) with build/module/MODULE.bazel
(only the version of the module() call may differ), and fails when metadata.json yanks the version or omits it.
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
STRIP_PREFIX = "euhedral_build"
SOURCE_KEYS = {"url", "integrity", "strip_prefix"}


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
    extra = sorted(set(source) - SOURCE_KEYS)
    if extra:
        raise SystemExit("source.json has %s, which Bazel applies and this check does not compare (only %s are allowed)"
                         % (", ".join(extra), ", ".join(sorted(SOURCE_KEYS))))
    for key in ("integrity", "strip_prefix"):
        if not isinstance(source.get(key), str):
            raise SystemExit("source.json has no string %s" % key)
    if source["strip_prefix"] != STRIP_PREFIX:
        raise SystemExit("source.json strip_prefix is %r, the comparison assumes %r" % (source["strip_prefix"], STRIP_PREFIX))
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
            if member.isdir():
                continue
            name = member.name
            if not member.isfile():
                raise SystemExit("the archive entry %s is not a regular file or a directory" % name)
            if not name.startswith(prefix) or ".." in name.split("/"):
                raise SystemExit("unexpected archive entry %s" % name)
            files[name[len(prefix):]] = (tar.extractfile(member).read(), bool(member.mode & 0o111))
    return files


def without_module_version(text, where):
    """`text` with the version string of its one module(...) call replaced by a fixed one."""
    calls = list(re.finditer(r"^module\(", text, re.M))
    if len(calls) != 1:
        raise SystemExit("%s must have exactly one module( call at the start of a line" % where)
    start = calls[0].end()
    depth, quote, i = 1, False, start
    while i < len(text) and depth:
        c = text[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == '"':
                quote = False
        elif c == '"':
            quote = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        i += 1
    if depth:
        raise SystemExit("%s: the module( call is not closed" % where)
    args = text[start:i]
    if len(re.findall(r"\bversion\s*=", args)) > 1:
        raise SystemExit("%s: the module( call has more than one version" % where)
    args = re.sub(r'\bversion\s*=\s*"[^"]*"', 'version = ""', args)
    return text[:start] + args + text[i:]


def registry_module_differences(registry, version, root):
    path = os.path.join(registry, "modules", MODULE, version, "MODULE.bazel")
    if not os.path.isfile(path):
        raise SystemExit("the registry has no MODULE.bazel for %s %s" % (MODULE, version))
    # newline="" keeps \r and \r\n as they are: the files must match byte for byte outside the version.
    with open(path, encoding="utf-8", newline="") as f:
        published = f.read()
    with open(os.path.join(root, "build", "module", "MODULE.bazel"), encoding="utf-8", newline="") as f:
        vendored = f.read()
    if without_module_version(published, "the registry's MODULE.bazel") != without_module_version(vendored, "build/module/MODULE.bazel"):
        return ["the registry's MODULE.bazel differs from build/module/MODULE.bazel beyond the module() version"]
    return []


def check_metadata(registry, version):
    path = os.path.join(registry, "modules", MODULE, "metadata.json")
    if not os.path.isfile(path):
        raise SystemExit("the registry has no metadata.json for %s" % MODULE)
    try:
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
    except ValueError as e:
        raise SystemExit("metadata.json for %s is not valid JSON: %s" % (MODULE, e))
    if not isinstance(meta, dict):
        raise SystemExit("metadata.json for %s is not a JSON object" % MODULE)
    versions = meta.get("versions")
    if not isinstance(versions, list) or not all(isinstance(v, str) for v in versions):
        raise SystemExit("metadata.json's versions is not a list of strings")
    yanked = meta.get("yanked_versions", {})
    if not isinstance(yanked, dict):
        raise SystemExit("metadata.json's yanked_versions is not an object")
    if version in yanked:
        raise SystemExit("%s %s is yanked: %s" % (MODULE, version, yanked[version]))
    if version not in versions:
        raise SystemExit("%s %s is not in metadata.json's versions" % (MODULE, version))


def vendored_tree(root):
    base = os.path.join(root, "build", "module")
    if os.path.islink(base):
        raise SystemExit("build/module is a symlink")
    files = {}
    for here, dirs, names in os.walk(base):
        for name in dirs + names:
            if os.path.islink(os.path.join(here, name)):
                raise SystemExit("%s is a symlink" % os.path.relpath(os.path.join(here, name), root))
        for name in names:
            path = os.path.join(here, name)
            with open(path, "rb") as f:
                files[os.path.relpath(path, base).replace(os.sep, "/")] = (f.read(), bool(os.stat(path).st_mode & 0o111))
    return files


def differences(published, vendored):
    out = ["only in the published module: %s" % p for p in sorted(set(published) - set(vendored))]
    out += ["only in build/module: %s" % p for p in sorted(set(vendored) - set(published))]
    both = sorted(set(published) & set(vendored))
    out += ["differs: %s" % p for p in both if published[p][0] != vendored[p][0]]
    out += ["executable bit differs: %s" % p for p in both if published[p][1] != vendored[p][1]]
    return out


def main(argv):
    registry = argv[1] if len(argv) > 1 else "/registry"
    root = argv[2] if len(argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    version = declared_version(root)
    published = published_tree(registry, version)
    check_metadata(registry, version)
    diff = differences(published, vendored_tree(root))
    diff += registry_module_differences(registry, version, root)
    if diff:
        print("build/module is not the published %s %s:" % (MODULE, version))
        print("\n".join("  " + d for d in diff))
        return 1
    print("build/module is the published %s %s" % (MODULE, version))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
