import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class RemoteFooterAudit(unittest.TestCase):
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
                files.append({"path": path})

            manifest = {"world_size": 2, "total_unique_states": 6,
                        "layer_counts": [1, 3, 2], "files": files}
            opener = lambda path: open(root / path, "rb")
            result = audit_manifest(manifest, opener, workers=1, max_rows=2)
            self.assertEqual(result["file_count"], 4)
            self.assertEqual(result["rows_by_rank"], {"0": 3, "1": 3})
            self.assertEqual(result["rows_total"], 6)

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
