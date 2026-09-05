"""Catch missing, duplicated and fragmented live depth reports."""
import io
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import distributed_gpu_bench as bench


class ProgressRelayTests(unittest.TestCase):
    def test_partial_records_are_emitted_once_and_noise_is_filtered(self):
        self.assertTrue(hasattr(bench, 'ProgressRelay'), 'live progress relay missing')
        source = io.BytesIO(b'noise\nMGBFS_DEPTH_BEGIN rank=0 depth=5\nMGBFS_DEPTH_')
        output = io.StringIO()
        relay = bench.ProgressRelay(source, output)
        relay.poll()
        relay.poll()
        self.assertEqual(output.getvalue(), 'MGBFS_DEPTH_BEGIN rank=0 depth=5\n')
        source.seek(0, 2)
        source.write(b'END rank=0 depth=5\n')
        relay.poll()
        self.assertEqual(output.getvalue(), 'MGBFS_DEPTH_BEGIN rank=0 depth=5\nMGBFS_DEPTH_END rank=0 depth=5\n')

    def test_oversized_record_is_dropped_and_final_record_is_flushed(self):
        self.assertTrue(hasattr(bench, 'ProgressRelay'), 'live progress relay missing')
        source = io.BytesIO(b'MGBFS_DEPTH_BEGIN ' + b'x' * 100000 + b'\nMGBFS_DEPTH_END depth=9')
        output = io.StringIO()
        relay = bench.ProgressRelay(source, output)
        for _ in range(8):
            relay.poll()
        relay.poll(final=True)
        self.assertEqual(output.getvalue(), 'MGBFS_DEPTH_END depth=9\n')


if __name__ == '__main__':
    unittest.main()
