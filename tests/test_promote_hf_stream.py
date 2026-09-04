import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.promote_hf_stream import combine_rank_commits, promote


def commit(rank, counts=(1, 2), branch="mgbfs-s3-run"):
    run_id = "s3-run"
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

    def test_promotion_is_one_atomic_commit_with_server_side_copies(self):
        class Copy:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class Add:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class Api:
            def __init__(self):
                self.calls = []

            def create_commit(self, **kwargs):
                self.calls.append(kwargs)
                return object()

        api = Api()
        combined, _ = promote(
            api, "TryDotAtwo/results", [commit(0), commit(1)], 2,
            copy_cls=Copy, add_cls=Add,
        )
        self.assertEqual(len(api.calls), 1)
        call = api.calls[0]
        copies = [item for item in call["operations"] if isinstance(item, Copy)]
        adds = [item for item in call["operations"] if isinstance(item, Add)]
        self.assertEqual(len(copies), 2)
        self.assertEqual(len(adds), 3)
        self.assertTrue(all(
            item.kwargs["src_revision"] == "mgbfs-s3-run" for item in copies
        ))
        self.assertEqual(
            {item.kwargs["path_in_repo"] for item in adds},
            {"layers/s3-run.parquet", "runs/s3-run.json", "verification/s3-run.json"},
        )
        self.assertEqual(combined["total_unique_states"], 6)

    def test_promotion_accepts_one_staging_branch_per_rank(self):
        class Copy:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class Add:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class Api:
            def create_commit(self, **kwargs):
                self.operations = kwargs["operations"]
                return object()

        api = Api()
        combined, _ = promote(
            api, "TryDotAtwo/results",
            [commit(0, branch="s3-rank-0"), commit(1, branch="s3-rank-1")], 2,
            copy_cls=Copy, add_cls=Add,
        )
        copies = [item for item in api.operations if isinstance(item, Copy)]
        self.assertEqual([item.kwargs["src_revision"] for item in copies], [
            "s3-rank-0", "s3-rank-1",
        ])
        self.assertEqual(combined["branches"], ["s3-rank-0", "s3-rank-1"])


if __name__ == "__main__":
    unittest.main()
