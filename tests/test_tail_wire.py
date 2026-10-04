import hashlib
import io
import struct
import tempfile
import unittest
from pathlib import Path
from scripts.tail_wire import consume, pack_batch
from scripts.bfs_tail_archive import TailArchive, pack_state
import json


def wire(corrupt=False, truncate=False):
    header = b'MGBFSAR1'+struct.pack('<Q',3)+bytes(32)
    chain = hashlib.sha256(header).digest()
    out = bytearray(header)
    for seq,(kind,depth,count,payload) in enumerate([
        (1,0,2,bytes([0,1,2,2,0,1])+bytes(32)), (2,0,2,b''), (3,1,2,b'')]):
        frame = b'MGBFSFR1'+struct.pack('<QQQQQ',kind,depth,count,len(payload),seq)+chain
        chain = hashlib.sha256(frame+payload).digest()
        out.extend(frame+payload+chain)
    if corrupt:
        out[140] ^= 1
    return bytes(out[:-1] if truncate else out)


class WireTests(unittest.TestCase):
    def test_committed_wire_layer_is_preserved_in_tail_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            archive = TailArchive(Path(root)/'tail', n=3, r=1, start=[0,1,2],
                actions={'L':'left','R':'right','X':'swap'}, program_commit='fixture',
                launch_config={'world':1}, sample_interval_seconds=.1)
            def committed(depth, count, path, digest):
                archive.completed_layer(depth, count, [path.read_bytes()], .1, {'0':100})
                path.unlink()
            receipt = consume(io.BytesIO(wire()), Path(root)/'spool', 3, committed)
            manifest_path = archive.snapshot(True, 'graph exhausted')
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(receipt['states'], 2)
            self.assertEqual(manifest['status'], 'COMPLETE')
            self.assertEqual(manifest['files'][0]['states'], 2)
            self.assertEqual((archive.root/manifest['files'][0]['path']).read_bytes(),
                bytes.fromhex('10020000000000000201000000000000'))
            self.assertEqual(list((Path(root)/'spool').glob('*.bin')), [])

    def test_vector_pack_matches_scalar(self):
        for n in (1,15,16,17,32):
            state = [i%16 for i in range(n)]
            self.assertEqual(pack_batch(bytes(state)*3,3,n),pack_state(state)*3)

    def test_commit_and_packed_payload(self):
        with tempfile.TemporaryDirectory() as root:
            layers=[]
            receipt=consume(io.BytesIO(wire()),root,3,
                lambda depth,count,path,digest: layers.append((depth,count,path.read_bytes())))
            self.assertEqual(receipt['states'],2)
            self.assertEqual(layers,[(0,2,pack_state([0,1,2])+pack_state([2,0,1]))])

    def test_corruption_never_commits_layer(self):
        with tempfile.TemporaryDirectory() as root:
            layers=[]
            with self.assertRaises(ValueError):
                consume(io.BytesIO(wire(corrupt=True)),root,3,lambda *args:layers.append(args))
            self.assertEqual(layers,[])

    def test_truncated_trailer_retains_completed_layer(self):
        with tempfile.TemporaryDirectory() as root:
            layers=[]
            with self.assertRaises(EOFError):
                consume(io.BytesIO(wire(truncate=True)),root,3,lambda *args:layers.append(args))
            self.assertEqual(len(layers),1)
            self.assertTrue(Path(layers[0][2]).is_file())


if __name__=='__main__':
    unittest.main()
