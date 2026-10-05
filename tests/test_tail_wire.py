import hashlib
import io
import struct
import tempfile
import unittest
from pathlib import Path
from scripts.tail_wire import consume, pack_batch
from scripts.bfs_tail_archive import pack_state


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
    def test_block_packing_all_widths_and_padding(self):
        for n in range(1,129):
            for bits in (4,8):
                states=[[(i+row*3)%min(1<<bits,128) for i in range(n)] for row in range(7)]
                raw=bytes(x for state in states for x in state)
                width=((n*bits+63)//64)*8
                expected=b''.join(sum(x<<(i*bits) for i,x in enumerate(state)).to_bytes(width,'little') for state in states)
                self.assertEqual(pack_batch(raw,len(states),n,bits),expected)

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

    def test_full_state_only_large_layer(self):
        count=1001
        header=b'MGBFSAS3'+struct.pack('<Q',3)+bytes(32)
        chain=hashlib.sha256(header).digest();out=bytearray(header)
        for seq,(kind,depth,rows,payload) in enumerate([
            (1,0,count,bytes([0,1,2])*count),(2,0,count,b''),(3,1,count,b'')]):
            frame=b'MGBFSFR1'+struct.pack('<QQQQQ',kind,depth,rows,len(payload),seq)+chain
            chain=hashlib.sha256(frame+payload).digest();out.extend(frame+payload+chain)
        with tempfile.TemporaryDirectory() as root:
            layers=[]
            receipt=consume(io.BytesIO(out),root,3,
                lambda depth,rows,path,digest:layers.append(path.read_bytes()))
            self.assertEqual(receipt['states'],count)
            self.assertEqual(layers,[pack_state([0,1,2])*count])

    def test_background_packed_digest_handoff(self):
        with tempfile.TemporaryDirectory() as root:
            observed=[]
            consume(io.BytesIO(wire()),root,3,None,
                    on_packed_layer=lambda depth,count,path,config,digest:
                    observed.append((count,config,digest,path.read_bytes())))
            count,config,digest,data=observed[0]
            self.assertEqual(count,2)
            self.assertEqual(config,bytes(32).hex())
            self.assertEqual(digest,hashlib.sha256(data).hexdigest())

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
