import importlib.util
import unittest
from pathlib import Path
import pyarrow as pa

class CodecBench(unittest.TestCase):
    def test_bounded_replay_reports_all_variants_and_exact_roundtrip(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/bench_archive_codec.py'
        self.assertTrue(path.exists(), 'bounded codec replay is missing')
        spec = importlib.util.spec_from_file_location('codec_bench', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        table = pa.table({'state': [b'abc', b'def'], 'hash128_le': [bytes(16), bytes(range(16))],
                          'rank_ordinal': pa.array([0, 1], pa.uint64())})
        rows = module.bench_table(table, repeats=2, slot_bytes=65536)
        self.assertEqual(len(rows), 4 * 4 * 2)
        self.assertTrue(all(x['roundtrip_equal'] for x in rows))
        self.assertTrue(all(0 < x['parquet_bytes'] <= 65536 for x in rows))
        with self.assertRaises(ValueError):
            module.bench_table(table, repeats=0, slot_bytes=65536)

if __name__ == '__main__':
    unittest.main()
