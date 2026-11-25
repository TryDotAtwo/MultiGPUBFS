"""Reuse checksummed native artifacts from an immutable Release or Actions build."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import urllib.request
import zipfile


def reuse_release(tag):
    base = "https://github.com/TryDotAtwo/MultiGPUBFS/releases/download/" + tag + "/"
    receipt = json.loads(urllib.request.urlopen(base + "manifest.json", timeout=120).read())
    work = Path("/tmp/mgbfs-portable")
    (work / "target/release").mkdir(parents=True)
    (work / "native-build").mkdir()
    wheel = work / "accepted.whl"
    urllib.request.urlretrieve(base + receipt["wheel"], wheel)
    assert hashlib.sha256(wheel.read_bytes()).hexdigest() == receipt["sha256"]
    with zipfile.ZipFile(wheel) as archive:
        names = {
            name[name.index("multigpubfs/_native/"):]: name
            for name in archive.namelist()
            if "multigpubfs/_native/" in name and not name.endswith("/")
        }
        manifest = json.loads(archive.read(names["multigpubfs/_native/manifest.json"]))
        for name, target in (
            ("bin/mgbfs", work / "target/release/mgbfs"),
            ("lib/libmgbfs_cuda.so", work / "native-build/libmgbfs_cuda.so"),
        ):
            data = archive.read(names["multigpubfs/_native/" + name])
            assert hashlib.sha256(data).hexdigest() == manifest["sha256"][name]
            target.write_bytes(data)
    (work / "target/release/mgbfs").chmod(0o755)
    (work / "native-manifest.json").write_text(json.dumps(manifest))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def changed_python_modules(git, source, target):
    changes = git("diff", "--no-ext-diff", "--no-renames", "--name-status", "-z", source, target, "--")
    entries = iter(changes.split(b"\0")[:-1])
    modules = []
    for status, raw_path in zip(entries, entries):
        path = PurePosixPath(os.fsdecode(raw_path))
        if (len(path.parts) > 1 and path.parts[0] in {"docs", "tests"}) or path.as_posix() in {
            ".github/workflows/package-native.yml",
            "scripts/reuse_portable_native.py",
        }:
            continue
        if (
            status == b"M"
            and path.parent == PurePosixPath("multigpubfs")
            and path.suffix == ".py"
            and path.name not in {"architecture_support.py", "artifact_metadata.py"}
        ):
            modules.append(path)
            continue
        raise ValueError(f"Cannot reuse the portable wheel with a source change to {path}")
    return modules


def native_records(path, prefix):
    # Wheel unpack verifies these hashes and wheel pack generates a fresh RECORD.
    # Comparing them covers the executable, runtimes and licenses without another
    # pass decompressing the full native payload.
    with zipfile.ZipFile(path) as archive:
        record, = (name for name in archive.namelist() if name.endswith(".dist-info/RECORD"))
        rows = csv.reader(archive.read(record).decode("utf-8").splitlines())
        native = {
            name.removeprefix(prefix): (digest, size, archive.getinfo(name).external_attr >> 16)
            for name, digest, size in rows
            if name.startswith(prefix) and name != prefix + "manifest.json"
        }
    executable = native.get("bin/mgbfs")
    if executable is None or not executable[2] & 0o111:
        raise ValueError("The portable wheel has no executable native CLI")
    if any(not digest or not size for digest, size, _ in native.values()):
        raise ValueError("The portable wheel has native files without checksummed records")
    return native


def repack_artifact(artifact_dir, source, output):
    repo = Path(__file__).resolve().parents[1]

    def git(*args):
        return subprocess.check_output(("git", *args), cwd=repo)

    source = git("rev-parse", "--verify", "--end-of-options", source + "^{commit}").decode().strip()
    target = git("rev-parse", "HEAD").decode().strip()
    modules = changed_python_modules(git, source, target)

    receipt = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    name = receipt["wheel"]
    if receipt["schema"] != 1 or Path(name).name != name or not name.endswith(".whl"):
        raise ValueError("Invalid Actions wheel receipt")
    original = artifact_dir / name
    if original.stat().st_size != receipt["size"] or sha256_file(original) != receipt["sha256"]:
        raise ValueError("Actions wheel does not match the receipt size and SHA-256")
    if receipt["manifest"].get("package_source_commit") != source:
        raise ValueError("Actions wheel package source does not match --source")

    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError(f"Wheel output directory must be empty: {output}")
    with tempfile.TemporaryDirectory(prefix="mgbfs-wheel-", dir=output.parent) as scratch:
        unpacked = Path(scratch) / "unpacked"
        packed = Path(scratch) / "packed"
        packed.mkdir()
        subprocess.run([sys.executable, "-m", "wheel", "unpack", "-d", str(unpacked), str(original)], check=True)
        tree, = unpacked.iterdir()
        package_info, = (
            candidate for candidate in tree.rglob("_package_info.json") if candidate.parent.name == "multigpubfs"
        )
        package = package_info.parent
        metadata = json.loads(package_info.read_text(encoding="utf-8"))
        manifest_path = package / "_native" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest != receipt["manifest"] or metadata.get("source_commit") != source:
            raise ValueError("The embedded package metadata does not match the Actions wheel receipt")

        for path in modules:
            destination = package / path.name
            if not destination.is_file() or destination.read_bytes() != git("show", f"{source}:{path}"):
                raise ValueError(f"Wheel module differs from the recorded source: {path}")
            destination.write_bytes(git("show", f"{target}:{path}"))

        metadata["source_commit"] = target
        manifest["package_source_commit"] = target
        write_json(package_info, metadata)
        write_json(manifest_path, manifest)

        subprocess.run([sys.executable, "-m", "wheel", "pack", "-d", str(packed), str(tree)], check=True)
        wheel, = packed.glob("*.whl")
        prefix = manifest_path.parent.relative_to(tree).as_posix() + "/"
        if native_records(original, prefix) != native_records(wheel, prefix):
            raise ValueError("Native payload or file modes changed during wheel repack")
        receipt.update(wheel=wheel.name, size=wheel.stat().st_size, sha256=sha256_file(wheel), manifest=manifest)
        write_json(packed / "manifest.json", receipt)
        (packed / "SHA256SUMS").write_text(f"{receipt['sha256']}  {wheel.name}\n", encoding="utf-8")
        for artifact in packed.iterdir():
            artifact.replace(output / artifact.name)


def main():
    if len(sys.argv) == 2 and not sys.argv[1].startswith("-"):
        reuse_release(sys.argv[1])
        return
    parser = argparse.ArgumentParser(description=__doc__, epilog="Pass a single release tag to extract Release binaries.")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Git commit used to build the Actions wheel.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repack_artifact(args.artifact_dir, args.source, args.output)


if __name__ == "__main__":
    main()
