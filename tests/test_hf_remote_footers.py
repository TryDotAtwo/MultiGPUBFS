import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class RemoteFooterAudit(unittest.TestCase):
    def test_explicit_anonymous_mode_never_uses_cached_environment_token(self):
        from scripts import audit_hf_parquet_footers as audit

        self.assertTrue(hasattr(audit, "resolve_token"), "anonymous auth selector missing")
        resolve_token = audit.resolve_token

        with patch.dict("os.environ", {"HF_TOKEN": "private-token"}):
            self.assertIs(resolve_token(anonymous=True), False)
            self.assertEqual(resolve_token(anonymous=False), "private-token")
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "HF_TOKEN_MISSING"):
                resolve_token(anonymous=False)

    def test_real_parquet_footers_require_complete_rank_parts_and_matching_totals(self):
        try:
            from scripts.audit_hf_parquet_footers import audit_manifest
        except ImportError as error:
            self.fail(f"footer auditor missing: {error}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = []
            for rank, part, rows in ((0, 0, 2), (0, 1, 1), (1, 0, 2), (1, 1, 1)):
                path = f"states/test-rank-{rank:05d}-part-{part:08d}.parquet"
                target = root / path
                target.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.table({"state": [b"x"] * rows,
                                         "rank": pa.array([rank] * rows, type=pa.uint32())}),
                               target)
                files.append({"path": path, "rows": rows})

            manifest = {"world_size": 2, "total_unique_states": 6,
                        "layer_counts": [1, 3, 2], "files": files}
            opener = lambda path: open(root / path, "rb")
            result = audit_manifest(manifest, opener, workers=1, max_rows=2)
            self.assertEqual(result["file_count"], 4)
            self.assertEqual(result["rows_by_rank"], {"0": 3, "1": 3})
            self.assertEqual(result["rows_total"], 6)
            self.assertTrue(result["per_file_rows_checked"])
            legacy = dict(manifest, files=[{"path": item["path"]} for item in files])
            self.assertFalse(audit_manifest(legacy, opener, workers=1,
                                            max_rows=2)["per_file_rows_checked"])

            # The aggregate stays six when two parts swap their row counts.
            pq.write_table(pa.table({"state": [b"x"],
                                     "rank": pa.array([0], type=pa.uint32())}),
                           root / files[0]["path"])
            pq.write_table(pa.table({"state": [b"x", b"x"],
                                     "rank": pa.array([0, 0], type=pa.uint32())}),
                           root / files[1]["path"])
            with self.assertRaisesRegex(ValueError, "FOOTER_FILE_ROWS"):
                audit_manifest(manifest, opener, workers=1, max_rows=2)
            pq.write_table(pa.table({"state": [b"x", b"x"],
                                     "rank": pa.array([0, 0], type=pa.uint32())}),
                           root / files[0]["path"])
            pq.write_table(pa.table({"state": [b"x"],
                                     "rank": pa.array([0], type=pa.uint32())}),
                           root / files[1]["path"])

            missing = dict(manifest, files=files[:3])
            with self.assertRaisesRegex(ValueError, "FOOTER_ROW_TOTAL"):
                audit_manifest(missing, opener, workers=1, max_rows=2)

            wrong_parts = [dict(item) for item in files]
            wrong_parts[1]["path"] = wrong_parts[1]["path"].replace("part-00000001", "part-00000002")
            (root / wrong_parts[1]["path"]).write_bytes((root / files[1]["path"]).read_bytes())
            with self.assertRaisesRegex(ValueError, "FOOTER_PART_SEQUENCE"):
                audit_manifest(dict(manifest, files=wrong_parts), opener, workers=1, max_rows=2)

            pq.write_table(pa.table({"state": ["x"], "rank": pa.array([0], type=pa.uint32())}),
                           root / files[1]["path"])
            with self.assertRaisesRegex(ValueError, "FOOTER_SCHEMA"):
                audit_manifest(manifest, opener, workers=1, max_rows=2)


if __name__ == "__main__":
    unittest.main()
