"""Immutable, validated NCCL build bundles for cloud workers."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def valid_name(name):
    p = PurePosixPath(name)
    return (not p.is_absolute() and ".." not in p.parts
            and "\\" not in name and str(p) == name
            and (name.startswith("include/") or name.startswith("lib/libnccl.so.")))


def export_bundle(root, output, recipe):
    root, output = Path(root).resolve(), Path(output)
    libraries = [p for p in (root / "lib").glob("libnccl.so.*")
                 if p.is_file() and not p.is_symlink()]
    if len(libraries) != 1:
        raise ValueError("Expected one canonical NCCL shared library")
    paths = sorted(p for p in (root / "include").rglob("*") if p.is_file()) + libraries
    names = [p.relative_to(root).as_posix() for p in paths]
    if not {"include/nccl.h", "include/nccl_device.h"}.issubset(names):
        raise ValueError("Required headers missing")
    for path, name in zip(paths, names):
        if not valid_name(name) or not path.resolve().is_relative_to(root):
            raise ValueError("Unsafe source path")
    if output.exists():
        raise ValueError("Output already exists")
    output.mkdir(parents=True)
    archive = output / "nccl-build.tar.gz"
    files = {}
    with tarfile.open(archive, "w:gz") as tf:
        for path, name in zip(paths, names):
            info = tf.gettarinfo(str(path), arcname=name)
            if not info.isreg():
                raise ValueError("Non-regular source member")
            with path.open("rb") as stream:
                tf.addfile(info, stream)
            files[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    manifest = {"version": 1, "recipe": recipe,
                "archive_sha256": digest(archive), "files": files}
    (output / "nccl-cache.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest


def restore_bundle(manifest_path, destination, recipe, archive_sha256):
    manifest_path, destination = Path(manifest_path), Path(destination)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archive = manifest_path.parent / "nccl-build.tar.gz"
    if (manifest.get("version") != 1 or manifest.get("recipe") != recipe
            or manifest.get("archive_sha256") != archive_sha256
            or digest(archive) != archive_sha256):
        raise ValueError("NCCL cache identity mismatch")
    files = manifest.get("files", {})
    libraries = [name for name in files if name.startswith("lib/libnccl.so.")]
    if (len(libraries) != 1 or not {"include/nccl.h", "include/nccl_device.h"}.issubset(files)
            or any(not valid_name(name) for name in files)):
        raise ValueError("Invalid NCCL member directory")
    if destination.exists():
        raise ValueError("Destination already exists")
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        if (len(members) != len(files) or {m.name for m in members} != set(files)
                or any(not m.isreg() or not valid_name(m.name) for m in members)):
            raise ValueError("Unsafe or inconsistent archive members")
        # Validate every byte before creating the installation directory.
        for member in members:
            meta = files[member.name]
            h = hashlib.sha256()
            with tf.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            if member.size != meta["bytes"] or h.hexdigest() != meta["sha256"]:
                raise ValueError("NCCL member checksum mismatch")
        destination.mkdir(parents=True)
        for member in members:
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as stream, target.open("xb") as out:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    out.write(block)
        library = Path(libraries[0]).name
        for alias in ("libnccl.so", "libnccl.so.2"):
            (destination / "lib" / alias).symlink_to(library)
    return manifest
