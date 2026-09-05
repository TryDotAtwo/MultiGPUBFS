import hashlib
import sys
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import promote_hf_stream as module


class ReconcilePublication(unittest.TestCase):
    def test_layer_metadata_accepts_equivalent_parquet_encoding_not_wrong_counts(self):
        combined = {"run_id": "run", "group_id": "s3", "layer_counts": [1, 2],
                    "config_digest": "00", "rank_archive_chains": [], "files": []}
        table = pa.table({"run_id": pa.array(["run", "run"], pa.string()),
                          "group_id": pa.array(["s3", "s3"], pa.string()),
                          "depth": pa.array([0, 1], pa.uint32()),
                          "unique_states": pa.array([1, 2], pa.uint64())})
        out = pa.BufferOutputStream()
        pq.write_table(table, out, compression="NONE")
        alternate = out.getvalue().to_pybytes()
        payloads = module._metadata_payloads(combined, layer_bytes=alternate)
        self.assertEqual(payloads["layers/run.parquet"], alternate)
        wrong = table.set_column(3, "unique_states", pa.array([1, 3], pa.uint64()))
        out = pa.BufferOutputStream(); pq.write_table(wrong, out)
        with self.assertRaisesRegex(ValueError, "PUBLICATION_CONFLICT"):
            module._metadata_payloads(combined, layer_bytes=out.getvalue().to_pybytes())

    def test_write_is_guarded_by_the_revision_checked_for_absence(self):
        api = SimpleNamespace(repo_info=lambda **kw: SimpleNamespace(sha="checked-parent"))
        with patch.object(module, "combine_rank_commits", return_value={}), \
             patch.object(module, "reconcile_publication", return_value=None) as check, \
             patch.object(module, "promote", return_value=({}, "receipt")) as write:
            module.promote_verified(api, "owner/repo", [], 2)
        self.assertEqual(check.call_args.kwargs["revision"], "checked-parent")
        self.assertEqual(write.call_args.kwargs["parent_commit"], "checked-parent")

    def test_existing_verified_run_never_writes(self):
        api = SimpleNamespace(repo_info=lambda **kw: SimpleNamespace(sha="existing"))
        receipt = SimpleNamespace(oid="existing")
        with patch.object(module, "combine_rank_commits", return_value={}), \
             patch.object(module, "reconcile_publication", return_value=receipt), \
             patch.object(module, "promote") as write:
            self.assertEqual(module.promote_verified(api, "owner/repo", [], 2), ({}, receipt))
        write.assert_not_called()

    def test_verified_objects_return_immutable_receipt_without_payload_download(self):
        class Api:
            def repo_info(self, **kwargs):
                return SimpleNamespace(sha="immutable-sha")

            def get_paths_info(self, **kwargs):
                self_revision = kwargs["revision"]
                assert self_revision == "immutable-sha"
                records = {
                    "runs/run.json": SimpleNamespace(path="runs/run.json", size=2,
                        blob_id=hashlib.sha1(b"blob 2\0{}").hexdigest(), lfs=None),
                    "states/a.parquet": SimpleNamespace(path="states/a.parquet", size=123,
                        blob_id="pointer", lfs=SimpleNamespace(sha256="ab" * 32)),
                }
                return [records[p] for p in kwargs["paths"] if p in records]

        combined = {"run_id": "run", "files": [
            {"path": "states/a.parquet", "bytes": 123, "sha256": "ab" * 32}]}
        with patch.object(module, "_metadata_payloads", return_value={"runs/run.json": b"{}"}):
            receipt = module.reconcile_publication(Api(), "owner/repo", combined)
        self.assertEqual(receipt.oid, "immutable-sha")
        self.assertEqual(receipt.commit_url,
                         "https://huggingface.co/datasets/owner/repo/commit/immutable-sha")

    def test_missing_manifest_is_not_success(self):
        api = SimpleNamespace(repo_info=lambda **kw: SimpleNamespace(sha="rev"),
                              get_paths_info=lambda **kw: [])
        self.assertIsNone(module.reconcile_publication(api, "owner/repo", {"run_id": "run"}))

    def test_missing_or_corrupt_state_object_is_never_reconciled(self):
        manifest = SimpleNamespace(path="runs/run.json", size=2,
            blob_id=hashlib.sha1(b"blob 2\0{}").hexdigest(), lfs=None)
        combined = {"run_id": "run", "files": [
            {"path": "states/a", "bytes": 123, "sha256": "ab" * 32}]}
        for state in [None, SimpleNamespace(path="states/a", size=123,
                    blob_id="pointer", lfs=SimpleNamespace(sha256="cd" * 32))]:
            def paths(**kwargs):
                return [x for x in [manifest, state]
                        if x is not None and x.path in kwargs["paths"]]
            api = SimpleNamespace(repo_info=lambda **kw: SimpleNamespace(sha="rev"),
                                  get_paths_info=paths)
            with self.subTest(state=state), patch.object(module, "_metadata_payloads",
                    return_value={"runs/run.json": b"{}"}):
                with self.assertRaisesRegex(ValueError, "PUBLICATION_CONFLICT"):
                    module.reconcile_publication(api, "owner/repo", combined)

    def test_existing_wrong_metadata_is_fatal(self):
        api = SimpleNamespace(repo_info=lambda **kw: SimpleNamespace(sha="rev"),
            get_paths_info=lambda **kw: [SimpleNamespace(path="runs/run.json", size=2,
                                                        blob_id="wrong", lfs=None)])
        with patch.object(module, "_metadata_payloads", return_value={"runs/run.json": b"{}"}):
            with self.assertRaisesRegex(ValueError, "PUBLICATION_CONFLICT"):
                module.reconcile_publication(api, "owner/repo", {"run_id": "run", "files": []})

    def test_timeout_after_commit_reconciles_without_second_write(self):
        complete = {"run_id": "run"}
        receipt = SimpleNamespace(oid="committed")
        with patch.object(module, "combine_rank_commits", return_value=complete), \
             patch.object(module, "reconcile_publication", side_effect=[None, receipt]), \
             patch.object(module, "promote", side_effect=TimeoutError("504")) as write:
            api = SimpleNamespace(repo_info=lambda **kw: SimpleNamespace(sha="before"))
            result = module.promote_verified(api, "owner/repo", [], 2)
        self.assertEqual(result, (complete, receipt))
        self.assertEqual(write.call_count, 1)

    def test_timeout_without_commit_preserves_failure(self):
        with patch.object(module, "combine_rank_commits", return_value={}), \
             patch.object(module, "reconcile_publication", return_value=None), \
             patch.object(module, "promote", side_effect=TimeoutError("504")):
            with self.assertRaisesRegex(TimeoutError, "504"):
                api = SimpleNamespace(repo_info=lambda **kw: SimpleNamespace(sha="before"))
                module.promote_verified(api, "owner/repo", [], 2)


if __name__ == "__main__":
    unittest.main()
