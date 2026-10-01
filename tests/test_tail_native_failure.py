import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_tail_bfs import native_failure


class FailureTests(unittest.TestCase):
    def test_worker_runtime_and_cli_errors(self):
        self.assertEqual(native_failure('MGBFS_ARCHIVE_WORKER_FATAL device=1 error=write: broken pipe\n'),'write: broken pipe')
        self.assertEqual(native_failure('MGBFS_RUNTIME_FATAL rank=0 error=LIBRARY_RANK_DEPTH_FATAL_16_16'),'LIBRARY_RANK_DEPTH_FATAL_16_16')
        self.assertEqual(native_failure('{"status":"ERROR","error":"ARCHIVE_FINISH_FAULT"}'),'ARCHIVE_FINISH_FAULT')
        self.assertIsNone(native_failure('MGBFS_DEPTH_END rank=1 depth=3 seconds=0.2'))
        self.assertIsNone(native_failure('{"status":"COMPLETE"}'))


if __name__=='__main__':unittest.main()
