import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from vram_autotune import select_capacity


class CapacitySelectionTests(unittest.TestCase):
    def test_uses_limiting_rank_with_exact_aligned_queries(self):
        for upper in (32768, 50000, 1000000):
            def query(rows):
                required = ((rows*157+100000+255)//256)*256
                return [dict(rank=rank, required_bytes=required, reserve_bytes=10000,
                    free_after_nccl_warmup_bytes=free)
                    for rank, free in enumerate((10000000, 12000000))]
            rows, cache = select_capacity(query, upper)
            expected = max(x for x in range(32768, upper+1)
                if query(x)[0]['required_bytes']+10000 <= 10000000)
            self.assertEqual(rows, expected)
            self.assertIn(rows, cache)

    def test_nonlinear_padding_is_corrected_by_exact_queries(self):
        def query(rows):
            return [dict(required_bytes=rows*10+(50000 if rows>=5000 else 0),
                         reserve_bytes=100, free_after_nccl_warmup_bytes=120000)]
        rows, _ = select_capacity(query, 20000, minimum=1000)
        self.assertEqual(rows, 6990)

    def test_tiny_graph_is_bounded_by_orbit_size(self):
        def query(rows):
            return [dict(required_bytes=rows, reserve_bytes=1,
                         free_after_nccl_warmup_bytes=100000)]
        self.assertEqual(select_capacity(query, 256)[0], 256)

    def test_probe_errors_are_not_capacity_evidence(self):
        for query in (lambda _: [], lambda _: [dict(required_bytes=-1)],
                      lambda _: (_ for _ in ()).throw(RuntimeError('NCCL failed'))):
            with self.assertRaises((ValueError, RuntimeError)):
                select_capacity(query, 100000)


if __name__ == '__main__':
    unittest.main()
