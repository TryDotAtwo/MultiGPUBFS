import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from tail_parquet import convert


class ParquetTailTests(unittest.TestCase):
    def fixture(self, root, width):
        records = [i.to_bytes(width, 'little') for i in range(9)]
        entries = []
        for depth, subset, offset, full in ((4, records[:3], 7, False),
                                            (5, records[3:], 0, True)):
            data = b''.join(subset)
            path = root / f'{depth}.bin'
            path.write_bytes(data)
            entries.append(dict(depth=depth, states=len(subset), bytes=len(data),
                path=path.name, sha256=hashlib.sha256(data).hexdigest(),
                full_layer=full, first_state_ordinal=offset))
        manifest = dict(graph=dict(n=16 if width == 8 else 24, r=10),
            packing=dict(bytes_per_state=width, byte_order='little'), files=entries,
            status='INCOMPLETE', program_commit='fixture', layers=[])
        (root / 'manifest.json').write_text(json.dumps(manifest))
        return records

    def test_shard_roundtrip_and_partial_offsets(self):
        import pyarrow.parquet as pq
        for width in (8, 16):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                records = self.fixture(root, width)
                result = convert(root, root / 'parquet', shard_bytes=(width + 14) * 4,
                                 row_group_rows=2)
                self.assertEqual(len(result['files']), 3)
                rows = []
                for entry in result['files']:
                    path = root / 'parquet' / entry['path']
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry['sha256'])
                    rows.extend(pq.read_table(path).to_pylist())
                self.assertEqual([x['state'] for x in rows], records)
                self.assertEqual([x['ordinal'] for x in rows], [7,8,9,0,1,2,3,4,5])
                self.assertEqual([x['depth'] for x in rows], [4]*3+[5]*6)
                self.assertEqual(result['retained_packed_bytes'], width*9)
                self.assertFalse(result['retained_layers'][0]['full_layer'])

    def test_corruption_never_publishes_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, 8)
            (root / '5.bin').write_bytes(b'!'*48)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                convert(root, root / 'parquet')
            self.assertFalse((root / 'parquet' / 'manifest.json').exists())


if __name__ == '__main__':
    unittest.main()
