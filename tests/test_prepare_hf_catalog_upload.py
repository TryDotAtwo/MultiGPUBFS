import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PrepareCatalogUpload(unittest.TestCase):
    def test_maps_verified_package_to_append_only_catalog_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / "package"
            for table in ("states", "layers", "runs"):
                (package / table).mkdir(parents=True)
                (package / table / f"s8-k1.{table}.parquet").write_bytes(table.encode())
            manifest = {
                "schema": "MGBFS_HF_DATASET_V1",
                "run_id": "s8-k1",
                "files": [
                    {"path": f"{table}/s8-k1.{table}.parquet"}
                    for table in ("states", "layers", "runs")
                ],
            }
            (package / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (package / "verification.json").write_text(
                json.dumps({"status": "PASS", "unique_states": 40320}), encoding="utf-8"
            )
            output = root / "upload"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/prepare_hf_catalog_upload.py"),
                 str(package), str(output)],
                check=True,
            )
            self.assertTrue((output / "states/s8-k1.states.parquet").is_file())
            self.assertTrue((output / "manifests/s8-k1.json").is_file())
            self.assertTrue((output / "verification/s8-k1.json").is_file())
            self.assertFalse((output / "manifest.json").exists())

    def test_rejects_unverified_or_path_escaping_packages(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / "package"
            package.mkdir()
            (package / "manifest.json").write_text(
                json.dumps({"schema": "MGBFS_HF_DATASET_V1", "run_id": "r1", "files": [{"path": "../x"}]}),
                encoding="utf-8",
            )
            (package / "verification.json").write_text('{"status":"PASS"}', encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/prepare_hf_catalog_upload.py"),
                 str(package), str(root / "out")], capture_output=True, text=True
            )
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
