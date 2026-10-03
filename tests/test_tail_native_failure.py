import sys
import unittest
import tempfile
import json
import signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_tail_bfs import native_failure,finish_run,cancellation_signals,native_completion,release_rank_spools


class FailureTests(unittest.TestCase):
    def test_rank_spools_wait_for_readers_and_preserve_final_snapshot(self):
        class Reader:
            alive=True
            def is_alive(self):return self.alive
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);spool=root/'spool-0';spool.mkdir()
            part=spool/'layer-000003.bin';part.write_bytes(b'pending rank')
            saved=root/'saved';saved.mkdir()
            final=saved/'layer-000002.bin';final.write_bytes(b'completed snapshot')
            metadata=spool/'receipt.json';metadata.write_text('{}')
            reader=Reader()
            self.assertFalse(release_rank_spools(root,1,[reader]))
            self.assertTrue(part.exists())
            reader.alive=False
            self.assertTrue(release_rank_spools(root,1,[reader]))
            self.assertFalse(part.exists())
            self.assertEqual(final.read_bytes(),b'completed snapshot')
            self.assertTrue(metadata.exists())
            self.assertTrue(release_rank_spools(root,1,[reader]))

    def test_completed_prefix_is_never_graph_completion(self):
        rank=dict(status='INCOMPLETE',stop_reason='calibration layer limit',
                  calibration_layers=34,last_completed_layer=33)
        self.assertEqual(native_completion([rank,rank],'34',34),
                         (False,'calibration layer limit'))
        for limit,depth in [(None,34),('35',34),('34',33)]:
            with self.assertRaises(ValueError):native_completion([rank,rank],limit,depth)
        with self.assertRaises(ValueError):native_completion([],None,0)
        with self.assertRaises(ValueError):native_completion([rank,{'status':'COMPLETE'}],'34',34)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            finish_run(root,root/'manifest.json','calibration layer limit',None,
                       'fixture','fixture',None,complete=False)
            self.assertEqual(json.loads((root/'run-summary.json').read_text())['status'],'INCOMPLETE')

    def test_signals_request_cancellation_and_restore_handlers(self):
        previous=signal.getsignal(signal.SIGTERM)
        with cancellation_signals() as cancelled:
            self.assertIsNone(cancelled())
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM,None)
            self.assertEqual(cancelled(),'requested cancellation: SIGTERM')
        self.assertEqual(signal.getsignal(signal.SIGTERM),previous)

    def test_upload_failure_still_writes_summary_and_drains_publisher(self):
        class BrokenPublisher:
            finished=False
            def enqueue(self,path):raise RuntimeError('injected enqueue failure')
            def finish(self):self.finished=True;raise RuntimeError('injected upload failure')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/'manifest.json'
            manifest.write_text('preserved snapshot')
            publisher=BrokenPublisher()
            failure,reason=finish_run(root,manifest,'requested cancellation: SIGTERM',
                RuntimeError('requested cancellation: SIGTERM'),'fixture','fixture',publisher)
            summary=json.loads((root/'run-summary.json').read_text())
            self.assertTrue(publisher.finished)
            self.assertEqual(summary['status'],'INCOMPLETE')
            self.assertEqual(summary['publication_status'],'FAILED')
            self.assertIn('local snapshots and pinned inputs retained',summary['reason'])
            self.assertEqual(manifest.read_text(),'preserved snapshot')
            self.assertEqual(str(failure),'requested cancellation: SIGTERM')

    def test_worker_runtime_and_cli_errors(self):
        self.assertEqual(native_failure('MGBFS_ARCHIVE_WORKER_FATAL device=1 error=write: broken pipe\n'),'write: broken pipe')
        self.assertEqual(native_failure('MGBFS_RUNTIME_FATAL rank=0 error=LIBRARY_RANK_DEPTH_FATAL_16_16'),'LIBRARY_RANK_DEPTH_FATAL_16_16')
        self.assertEqual(native_failure('{"status":"ERROR","error":"ARCHIVE_FINISH_FAULT"}'),'ARCHIVE_FINISH_FAULT')
        self.assertIsNone(native_failure('MGBFS_DEPTH_END rank=1 depth=3 seconds=0.2'))
        self.assertIsNone(native_failure('{"status":"COMPLETE"}'))


if __name__=='__main__':unittest.main()
