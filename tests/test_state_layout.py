import unittest
import tempfile
import json
from pathlib import Path
from scripts.state_layout import state_layout, orbit_layout, symbol_bits
from scripts.bfs_tail_archive import pack_state
from scripts.tail_wire import pack_batch
from scripts.analyze_tail import unpack_state
from scripts.bfs_tail_archive import TailArchive
from scripts.tail_parquet import convert


class LayoutTests(unittest.TestCase):
    def test_runtime_widths_and_batch_roundtrip(self):
        for n,alphabet,width,bits in [(16,16,8,4),(17,16,16,4),
                (17,17,24,8),(33,2,24,4),(128,16,64,4),(128,128,128,8)]:
            with self.subTest(n=n,alphabet=alphabet):
                layout=state_layout(n,alphabet)
                self.assertEqual((layout['bytes_per_state'],layout['bits_per_symbol']),(width,bits))
                state=[i%alphabet for i in range(n)]
                packed=pack_state(state)
                self.assertEqual(len(packed),width)
                self.assertEqual(pack_batch(bytes(state)*3,3,n,bits),packed*3)
                self.assertEqual(unpack_state(packed,n,bits),tuple(state))

    def test_exact_multiword_orbit_without_allocating_state_space(self):
        for n,r in [(17,1),(27,12),(128,1),(128,127)]:
            layout=orbit_layout(n,r)
            value=sum(x<<(64*i) for i,x in enumerate(layout['order_words']))
            self.assertEqual(value,int(layout['states']))
            self.assertEqual(layout['bytes']%8,0)
        self.assertGreater(orbit_layout(128,1)['uint64_words'],2)
        self.assertEqual(orbit_layout(128,127)['states'],'128')
        self.assertEqual(symbol_bits(257),16)

    def test_degree_boundary(self):
        with self.assertRaises(ValueError):state_layout(129,2)

    def test_wide_parquet_roundtrip_and_manifest(self):
        import pyarrow.parquet as pq
        for n,r in [(17,1),(65,64),(128,1)]:
            with self.subTest(n=n), tempfile.TemporaryDirectory() as directory:
                start=list(range(n-r+1))+[n-r]*(r-1)
                root=Path(directory)
                archive=TailArchive(root/'raw',n=n,r=r,start=start,actions=['L','R','X'],
                    program_commit='fixture',launch_config={},sample_interval_seconds=.05)
                archive.completed_layer(0,1,[pack_state(start)],.1,{'0':1})
                manifest=archive.snapshot(False,'fixture')
                convert(archive.root,root/'parquet')
                m=json.loads((root/'parquet/manifest.json').read_text())
                entry=m['files'][0]
                state=pq.read_table(root/'parquet'/entry['path'])['state'][0].as_py()
                self.assertEqual(unpack_state(state,n,m['packing']['bits_per_symbol']),tuple(start))
                self.assertEqual(m['packing']['bytes_per_state']%8,0)
