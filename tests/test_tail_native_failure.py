import sys
import unittest
import tempfile
import json
import signal
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_tail_bfs import native_failure,finish_run,cancellation_signals


class FailureTests(unittest.TestCase):
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
