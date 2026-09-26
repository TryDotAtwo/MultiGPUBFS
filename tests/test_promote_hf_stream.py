import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.promote_hf_stream import combine_rank_commits, promote, promote_verified


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
            "rows": sum(counts),
            "sha256": f"{rank + 4:02x}" * 32,
        }],
    }


class PromoteStream(unittest.TestCase):
    def test_reference_rejects_wrong_layers_before_any_hub_access(self):
        class NoNetwork:
            def repo_info(self, **kwargs):
                raise AssertionError('Hub accessed before reference validation')

        records = [commit(0, (1, 1, 1)), commit(1, (0, 1, 2))]
        # Six states in both cases, but the depth histogram is wrong.
        with self.assertRaisesRegex(ValueError, 'LAYER_MISMATCH'):
            promote_verified(NoNetwork(), 'unused/results', records, 2,
                             reference={'n': 3, 'layers': [1, 3, 2]})

    def test_boolean_archive_count_is_not_a_state_count(self):
        record = commit(0, (True, 2))
        with self.assertRaisesRegex(ValueError, 'STREAM_LAYERS'):
            combine_rank_commits([record], 1)

    def test_positive_rank_requires_complete_parquet_row_inventory(self):
        for files in (None, [], [dict(commit(0)["files"][0], rows=2)]):
            record = commit(0)
            if files is None:
                record.pop("files")
            else:
                record["files"] = files
            with self.assertRaisesRegex(ValueError, "STREAM_FILE_ROWS"):
                combine_rank_commits([record], 1)

    def test_staged_state_path_cannot_escape_rank_inventory(self):
        for path in (
            "pending/mgbfs-s3-run/states/../rank-00000-part-00000000.parquet",
            "pending/mgbfs-s3-run/states/rank-00001-part-00000000.parquet",
        ):
            record = commit(0)
            record["files"][0]["path"] = path
            with self.assertRaisesRegex(ValueError, "STREAM_FILE"):
                combine_rank_commits([record], 1)

    def test_eight_rank_publication_requires_complete_disjoint_inventory(self):
        records = [commit(r, (int(r == 0), r), branch=f'fixture-rank-{r}')
                   for r in reversed(range(8))]
        combined = combine_rank_commits(records, expected_world=8)
        self.assertEqual(combined['layer_counts'], [1, 28])
        self.assertEqual(combined['total_unique_states'], 29)
        self.assertEqual(len({x['path'] for x in combined['files']}), 8)
        self.assertEqual(combined['branches'], [f'fixture-rank-{r}' for r in range(8)])
        for broken in (records[:-1], records[:-1] + [records[0]]):
            with self.assertRaisesRegex(ValueError, 'RANK_SET'):
                combine_rank_commits(broken, expected_world=8)
        records[0]['status'] = 'INCOMPLETE'
        with self.assertRaisesRegex(ValueError, 'STREAM_SCHEMA'):
            combine_rank_commits(records, expected_world=8)

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

            def repo_info(self, **kwargs):
                return SimpleNamespace(sha="a" * 40)

            def get_paths_info(self, **kwargs):
                return [SimpleNamespace(path=path, size=123,
                    lfs=SimpleNamespace(sha256=("04" if "rank-00000" in path else "05") * 32))
                    for path in kwargs["paths"]]

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
            item.kwargs["src_revision"] == "a" * 40 for item in copies
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
            def repo_info(self, **kwargs):
                return SimpleNamespace(sha=("a" if kwargs["revision"] == "s3-rank-0" else "b") * 40)

            def get_paths_info(self, **kwargs):
                return [SimpleNamespace(path=path, size=123,
                    lfs=SimpleNamespace(sha256=("04" if "rank-00000" in path else "05") * 32))
                    for path in kwargs["paths"]]

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
            "a" * 40, "b" * 40,
        ])
        self.assertEqual(combined["branches"], ["s3-rank-0", "s3-rank-1"])

    def test_promotion_pins_and_verifies_source_before_copy(self):
        from types import SimpleNamespace

        class Copy:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class Add:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class Api:
            def __init__(self, valid=True):
                self.valid = valid
                self.operations = None

            def repo_info(self, **kwargs):
                self.assert_branch = kwargs["revision"] == "mgbfs-s3-run"
                return SimpleNamespace(sha="a" * 40)

            def get_paths_info(self, **kwargs):
                self.assert_revision = kwargs["revision"] == "a" * 40
                return [SimpleNamespace(path=path, size=123,
                    lfs=SimpleNamespace(sha256=("04" if self.valid else "ff") * 32))
                    for path in kwargs["paths"]]

            def create_commit(self, **kwargs):
                self.operations = kwargs["operations"]
                return object()

        good = Api()
        combined, _ = promote(good, "TryDotAtwo/results", [commit(0)], 1,
                              copy_cls=Copy, add_cls=Add)
        self.assertTrue(good.assert_branch and good.assert_revision)
        self.assertEqual(combined["files"][0]["source_revision"], "a" * 40)
        self.assertEqual(good.operations[0].kwargs["src_revision"], "a" * 40)
        bad = Api(valid=False)
        with self.assertRaisesRegex(ValueError, "STREAM_SOURCE"):
            promote(bad, "TryDotAtwo/results", [commit(0)], 1,
                    copy_cls=Copy, add_cls=Add)
        self.assertIsNone(bad.operations)


if __name__ == "__main__":
    unittest.main()
