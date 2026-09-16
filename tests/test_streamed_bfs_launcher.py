import sys
import unittest
import tempfile
import time
import os
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import streamed_bfs_launcher as launcher


class StreamedLauncher(unittest.TestCase):
    def test_publication_runs_after_consumers_finish(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / 'drained'
            plan = dict(env={}, consumers=[[sys.executable, '-c',
                f'import time; from pathlib import Path; time.sleep(0.2); Path({str(marker)!r}).touch()']],
                search=['unused'], promotion=[sys.executable, '-c',
                f'import json; from pathlib import Path; assert Path({str(marker)!r}).exists(); '
                'print(json.dumps(dict(status="COMPLETE",commit_url="fixture-receipt")))'])
            with patch('distributed_gpu_bench.run_group', return_value=dict(status='COMPLETE')):
                result = launcher.execute_plan(plan, root, dict(os.environ), 5)
            self.assertEqual(result['publication']['commit_url'], 'fixture-receipt')
            self.assertTrue((root / 'stream-summary.json').exists())

    def test_failed_consumer_prevents_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / 'published'
            plan = dict(env={}, consumers=[[sys.executable, '-c', 'raise SystemExit(7)']],
                        search=['unused'], promotion=[sys.executable, '-c',
                        f'from pathlib import Path; Path({str(marker)!r}).touch()'])
            def search(*args, **kwargs):
                self.assertEqual(len(kwargs['required_processes']), 1)
                return dict(status='COMPLETE')
            with patch('distributed_gpu_bench.run_group', side_effect=search):
                with self.assertRaisesRegex(RuntimeError, 'CONSUMER_FAILED'):
                    launcher.execute_plan(plan, root, dict(os.environ), 5)
            self.assertFalse(marker.exists())

    def test_shared_deadline_terminates_consumers_and_prevents_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = dict(env={}, consumers=[[sys.executable, '-c', 'import time; time.sleep(30)']],
                        search=['unused'], promotion=['unused'])
            processes = []
            def search(*args, **kwargs):
                processes.extend(kwargs['required_processes'])
                return dict(status='COMPLETE')
            with patch('distributed_gpu_bench.run_group', side_effect=search):
                with self.assertRaises(TimeoutError):
                    launcher.execute_plan(plan, root, dict(os.environ), 0.3)
            self.assertTrue(all(p.poll() is not None for p in processes))

    def config(self):
        return dict(world=8, n=13, batch=262144, capacity=60000000,
                    archive_rows=262144, archive_slots=256, upload_slots=8,
                    max_slot_bytes=134217728, rows_per_shard=1000000,
                    profile='DENSE', owner='CUCO_INDEXED', pre_dedup='ON',
                    shards=8, buckets=256, bucket_capacity=2000000,
                    job_buckets=4, library_pool_bytes=1024**3,
                    scratch_bytes=8*1024**3, reserve_bytes=8*1024**3,
                    timeout_seconds=3600, run_id='s13-test',
                    repo_id='TryDotAtwo/multigpubfs-bfs-results')

    def test_eight_rank_plan_routes_every_archive_and_forces_streaming(self):
        plan = launcher.make_plan(self.config(), Path('/repo'), Path('/run'))
        self.assertEqual(len(plan['consumers']), 8)
        self.assertEqual(len(set(plan['fifos'])), 8)
        self.assertIn('--nproc-per-node=8', plan['search'])
        self.assertEqual(plan['search'][5:8], ['bench', '--reference', 's13'])
        self.assertEqual(plan['env']['MGBFS_BENCH_WORLD_SIZE'], '8')
        self.assertEqual(plan['env']['MGBFS_ARCHIVE_STREAM'], '1')
        self.assertEqual(plan['env']['MGBFS_BENCH_SKIP_ARCHIVE'], '0')
        self.assertEqual(plan['env']['MGBFS_STATE_CODEC'], 'permutation_u8')
        self.assertEqual(plan['upload_bytes'], 8*8*134217728)
        self.assertIn('--reference', plan['promotion'])
        self.assertEqual(plan['promotion'][-8:], plan['commits'])

    def test_invalid_configuration_rejected_before_side_effects(self):
        for key, value in [('world', 3), ('capacity', 0), ('batch', True),
                           ('timeout_seconds', -1), ('run_id', '../escape'),
                           ('owner', 'automatic'), ('shards', 3),
                           ('capacity', 2**32), ('job_buckets', 64)]:
            with self.subTest(key=key):
                config = self.config(); config[key] = value
                with self.assertRaises(ValueError):
                    launcher.make_plan(config, Path('/repo'), Path('/run'))


if __name__ == '__main__':
    unittest.main()
