import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nccl_dependency_cache import export_bundle, restore_bundle

RECIPE = {"upstream": "pinned", "architecture": "sm75", "patches": ["a"], "nvcc": "12.9"}
FILES = {"include/nccl.h": b"header", "include/nccl_device.h": b"device",
         "lib/libnccl.so.2.29.7": b"tested-library"}

def fixture(root, extra=None):
    contents = dict(FILES)
    if extra:
        contents.update(extra)
    archive = root / "nccl-build.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for name, body in contents.items():
            member = tarfile.TarInfo(name)
            member.size = len(body)
            tf.addfile(member, io.BytesIO(body))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    manifest = root / "nccl-cache.json"
    manifest.write_text(json.dumps({"version": 1, "recipe": RECIPE,
        "archive_sha256": digest, "files": {name: {"bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest()} for name, body in contents.items()}}))
    return manifest, digest

class NcclDependencyCacheTests(unittest.TestCase):
    def test_export_retains_headers_library_and_recipe(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            build = root / "build"
            for name, body in FILES.items():
                path = build / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(body)
            out = root / "out"
            export_bundle(build, out, RECIPE)
            self.assertTrue((out / "nccl-cache.json").is_file())
            manifest = json.loads((out / "nccl-cache.json").read_text())
            self.assertEqual(manifest["recipe"], RECIPE)
            self.assertEqual(set(manifest["files"]), set(FILES))

    def test_restore_preserves_bytes_and_link_names(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest, digest = fixture(root)
            dest = root / "restored"
            restore_bundle(manifest, dest, RECIPE, digest)
            for name, body in FILES.items():
                self.assertEqual((dest / name).read_bytes(), body)
            self.assertEqual((dest / "lib/libnccl.so").read_bytes(), b"tested-library")
            self.assertEqual((dest / "lib/libnccl.so.2").read_bytes(), b"tested-library")

    def test_wrong_recipe_is_fatal_before_installation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest, digest = fixture(root)
            dest = root / "restored"
            with self.assertRaises(ValueError):
                restore_bundle(manifest, dest, dict(RECIPE, architecture="sm80"), digest)
            self.assertFalse(dest.exists())

    def test_changed_archive_is_fatal_before_installation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest, digest = fixture(root)
            with (root / "nccl-build.tar.gz").open("ab") as stream:
                stream.write(b"corruption")
            dest = root / "restored"
            with self.assertRaises(ValueError):
                restore_bundle(manifest, dest, RECIPE, digest)
            self.assertFalse(dest.exists())

    def test_untrusted_digest_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest, digest = fixture(root)
            with self.assertRaises(ValueError):
                restore_bundle(manifest, root / "restored", RECIPE, "0" * 64)

    def test_path_escape_is_fatal_before_installation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            manifest, digest = fixture(root, {"../escape": b"bad"})
            dest = root / "restored"
            with self.assertRaises(ValueError):
                restore_bundle(manifest, dest, RECIPE, digest)
            self.assertFalse(dest.exists())
            self.assertFalse((root.parent / "escape").exists())

if __name__ == "__main__":
    unittest.main()
