import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.promote_hf_stream import combine_rank_commits


def commit(rank, counts=(1, 2)):
    run_id = "s3-run"
    branch = "mgbfs-s3-run"
    return {
        "schema": "MGBFS_HF_STREAM_COMMIT_V1",
        "status": "COMPLETE",
        "run_id": run_id,
        "group_id": "s3",
        "config_digest": "01" * 32,
        "branch": branch,
        "rank": rank,
        "state_bytes": 9,
        "total_unique_states": sum(counts),
        "max_depth": len(counts) - 1,
        "layer_counts": list(counts),
        "archive_chain_sha256": f"{rank + 2:02x}" * 32,
        "files": [{
            "path": f"pending/{branch}/states/rank-{rank:05d}-part-00000000.parquet",
            "bytes": 123,
            "sha256": f"{rank + 4:02x}" * 32,
        }],
    }


class PromoteStream(unittest.TestCase):
    def test_combines_rank_layers_and_builds_disjoint_final_paths(self):
        combined = combine_rank_commits([commit(1, (0, 2)), commit(0, (1, 0))], expected_world=2)
        self.assertEqual(combined["layer_counts"], [1, 2])
        self.assertEqual(combined["total_unique_states"], 3)
        self.assertEqual([item["path"] for item in combined["files"]], [
            "states/s3-run-rank-00000-part-00000000.parquet",
            "states/s3-run-rank-00001-part-00000000.parquet",
        ])

    def test_rejects_missing_rank_config_mismatch_and_duplicate_destination(self):
        with self.assertRaisesRegex(ValueError, "RANK_SET"):
            combine_rank_commits([commit(0)], expected_world=2)
        broken = commit(1)
        broken["config_digest"] = "02" * 32
        with self.assertRaisesRegex(ValueError, "STREAM_CONFIG"):
            combine_rank_commits([commit(0), broken], expected_world=2)
        duplicate = commit(1)
        duplicate["files"][0]["path"] = commit(0)["files"][0]["path"]
        with self.assertRaisesRegex(ValueError, "STREAM_FILE"):
            combine_rank_commits([commit(0), duplicate], expected_world=2)


if __name__ == "__main__":
    unittest.main()
