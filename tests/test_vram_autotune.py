import sys
import unittest
import json
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from vram_autotune import select_capacity, native_query


class CapacitySelectionTests(unittest.TestCase):
    def test_valid_records_from_failed_process_are_rejected(self):
        records=[dict(rank=i,required_bytes=123,reserve_bytes=10,
                      free_after_nccl_warmup_bytes=1000) for i in range(2)]
        output='\n'.join('MGBFS_MEMORY_QUERY '+json.dumps(x) for x in records)+'\nMEMORY_QUERY_DONE'
        process=MagicMock(returncode=1)
        process.communicate.return_value=(output,None)
        with tempfile.TemporaryDirectory() as directory,patch(
                'vram_autotune.subprocess.Popen',return_value=process):
            with self.assertRaisesRegex(RuntimeError,'native memory query failed'):
                native_query(dict(world=2,n=7,r=3,batch=256,env={}),
                             Path(directory),Path(directory)/'query',{})

    def test_native_query_explicitly_selects_archive_free_cli_contract(self):
        records = [dict(rank=i, required_bytes=123, reserve_bytes=10,
                        free_after_nccl_warmup_bytes=1000) for i in range(2)]
        output = '\n'.join('MGBFS_MEMORY_QUERY ' + json.dumps(r) for r in records)
        process = MagicMock()
        process.returncode = 0
        process.communicate.return_value = (output + '\nMEMORY_QUERY_DONE', None)
        with tempfile.TemporaryDirectory() as directory, patch(
                'vram_autotune.subprocess.Popen', return_value=process) as launch:
            cfg = dict(world=2, n=7, r=3, batch=256, env={})
            result = native_query(cfg, Path(directory), Path(directory)/'query', {})
            command = launch.call_args.args[0]
            self.assertEqual(command[-1], '--search-only')
            self.assertEqual(launch.call_args.kwargs['env']['MGBFS_MEMORY_QUERY'], '1')
            for record in records:
                record['query_to_run_margin_bytes'] = 64 << 20
            self.assertEqual(result, records)

    def test_query_to_run_headroom_is_reserved_in_capacity_selection(self):
        def query(rows):
            return [dict(required_bytes=rows*10, reserve_bytes=100,
                         query_to_run_margin_bytes=200,
                         free_after_nccl_warmup_bytes=10000)]
        self.assertEqual(select_capacity(query, 2000, minimum=256)[0], 970)

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
