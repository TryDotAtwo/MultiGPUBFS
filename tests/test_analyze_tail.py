import tempfile
import unittest
from pathlib import Path
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from scripts.analyze_tail import iter_case_batches, unpack_state, layer_statistics


class AnalysisTests(unittest.TestCase):
    def test_shared_shard_selects_case_depth_and_bounds_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            schema=pa.schema([('n',pa.uint8()),('r',pa.uint8()),
                ('depth',pa.uint32()),('ordinal',pa.uint64()),('state',pa.binary(8))])
            rows=[dict(n=n,r=r,depth=d,ordinal=i,state=bytes([i])+bytes(7))
                for n,r,d in [(4,1,2),(4,2,2),(4,1,3)] for i in range(5)]
            pq.write_table(pa.Table.from_pylist(rows,schema=schema),root/'shared.parquet',row_group_size=3)
            manifest=dict(storage_format='parquet',graph=dict(n=4,r=1),
                packing=dict(bytes_per_state=8),layers=[dict(depth=0,states=1)],
                files=[dict(path='shared.parquet',layers=[dict(depth=2),dict(depth=3)])])
            batches=list(iter_case_batches(manifest,fsspec.filesystem('file'),str(root),depths=[3],batch_rows=2))
            self.assertEqual(sum(x.num_rows for x in batches),5)
            self.assertTrue(all(x.num_rows<=2 for x in batches))
            self.assertEqual([v for b in batches for v in b['ordinal'].to_pylist()],list(range(5)))
            self.assertEqual(layer_statistics(manifest),[dict(depth=0,states=1)])
            manifest['packing']['bytes_per_state']=16
            with self.assertRaises(ValueError):list(iter_case_batches(manifest,fsspec.filesystem('file'),str(root)))

    def test_nibble_order_and_wide_padding(self):
        self.assertEqual(unpack_state(bytes([0x21,0x43])+bytes(6),4),(1,2,3,4))
        self.assertEqual(unpack_state(bytes([0x21])*8+bytes(8),17),(1,2)*8+(0,))
        with self.assertRaises(ValueError):unpack_state(bytes(8),17)

    def test_traversal_is_rejected_before_open(self):
        manifest=dict(storage_format='parquet',graph=dict(n=4,r=1),packing=dict(bytes_per_state=8),
            files=[dict(path='../outside.parquet')])
        with self.assertRaises(ValueError):list(iter_case_batches(manifest,fsspec.filesystem('file')))
