import hashlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from scripts.tail_wire import consume_selected
from scripts.selected_tail import SelectedTailArchive
from scripts.bfs_tail_archive import pack_state


def wire(frames):
    header = b'MGBFSAS2'+struct.pack('<Q',3)+bytes(32)
    chain = hashlib.sha256(header).digest(); out = bytearray(header)
    for seq,(kind,depth,count,payload) in enumerate(frames):
        frame = b'MGBFSFR1'+struct.pack('<QQQQQ',kind,depth,count,len(payload),seq)+chain
        chain = hashlib.sha256(frame+payload).digest(); out.extend(frame+payload+chain)
    return bytes(out)


def archive(root):
    return SelectedTailArchive(root,n=3,r=1,start=[0,1,2],
        actions={'L':'left','R':'right','X':'swap'},program_commit='test',
        launch_config={'retention_policy':'last_complete_small_1000'},
        sample_interval_seconds=.05)


class SelectedTests(unittest.TestCase):
    def test_global_prefix_across_eight_ranks_is_not_eight_thousand_states(self):
        with tempfile.TemporaryDirectory() as root:
            a=archive(Path(root)/'saved')
            first,second=pack_state([0,1,2]),pack_state([2,1,0])
            counts=[500,3000,0,10,1001,4000,0,100]
            parts=[first*500]+[second*min(n,1000) for n in counts[1:]]
            a.selected_layer(0,sum(counts),parts,.1,{str(i):None for i in range(8)})
            m=json.loads(a.snapshot(False).read_text())
            self.assertEqual(m['files'][0]['states'],1000)
            self.assertEqual((a.root/m['files'][0]['path']).read_bytes(),first*500+second*500)

    def test_complete_small_graph_roundtrips_independent_word_oracle(self):
        from scripts.verify_tail_oracle import verify
        with tempfile.TemporaryDirectory() as root:
            a=SelectedTailArchive(Path(root)/'saved',n=4,r=1,start=[0,1,2,3],
                actions={'L':'left','R':'right','X':'swap'},program_commit='test',
                launch_config={'retention_policy':'last_complete_small_1000'},sample_interval_seconds=.05)
            frontier={(0,1,2,3)}; visited=set(frontier); depth=0
            while frontier:
                a.selected_layer(depth,len(frontier),[b''.join(pack_state(list(x)) for x in sorted(frontier))],.1,{'0':None})
                children=set()
                for state in frontier:
                    children.update((state[1:]+state[:1],state[-1:]+state[:-1],(state[1],state[0],*state[2:])))
                frontier=children-visited;visited.update(frontier);depth+=1
            a.snapshot(True,'exhausted')
            self.assertEqual(verify(a.root)['states'],24)

    def test_samples_never_create_spool_files_and_empty_rank_commits(self):
        frames=[(1,0,1000,bytes([0,1,2])*1000),(2,0,1000,b''),
                (2,1,0,b''),(3,2,1000,b'')]
        with tempfile.TemporaryDirectory() as root:
            spool=Path(root)/'spool'; events=[]
            receipt=consume_selected(io.BytesIO(wire(frames)),spool,3,
                                     lambda *args:events.append(args))
            self.assertFalse(spool.exists())
            self.assertEqual([x[1] for x in events],[1000,0])
            self.assertEqual(events[0][2],pack_state([0,1,2])*1000)
            self.assertFalse(receipt['terminal_replaced'])

    def test_terminal_replacement_streams_large_layer_only(self):
        frames=[(1,0,1,bytes([0,1,2])),(2,0,1,b''),
                (1,1,1000,bytes([2,1,0])*1000),(2,1,1000,b''),
                (4,1,0,b''),(1,1,1500,bytes([2,1,0])*1500),
                (1,1,1500,bytes([0,2,1])*1500),(2,1,3000,b''),(3,2,3001,b'')]
        with tempfile.TemporaryDirectory() as root:
            events=[]
            receipt=consume_selected(io.BytesIO(wire(frames)),Path(root)/'spool',3,
                                     lambda *args:events.append(args))
            self.assertTrue(receipt['terminal_replaced'])
            self.assertEqual(events[-1][:2],(1,3000))
            self.assertTrue(events[-1][-1])
            self.assertEqual(events[-1][2].read_bytes(),
                pack_state([2,1,0])*1500+pack_state([0,2,1])*1500)
            self.assertEqual(len(list(Path(root).rglob('*.bin'))),1)

    def test_corrupt_or_truncated_terminal_keeps_committed_ram_sample(self):
        frames=[(1,0,1000,bytes([0,1,2])*1000),(2,0,1000,b''),
                (4,0,0,b''),(1,0,3000,bytes([2,1,0])*3000),(2,0,3000,b'')]
        for broken in (wire(frames)[:-1],wire(frames)[:-40]+b'x'*40):
            with tempfile.TemporaryDirectory() as root:
                events=[]
                with self.assertRaises((EOFError,ValueError)):
                    consume_selected(io.BytesIO(broken),Path(root)/'spool',3,
                                     lambda *args:events.append(args))
                self.assertEqual(len(events),1)
                self.assertEqual(events[0][1],1000)
                self.assertFalse(list(Path(root).rglob('*.bin')))

    def test_oversized_sample_and_duplicate_replacement_rejected(self):
        cases=[[(1,0,1001,bytes([0,1,2])*1001)],
               [(2,0,0,b''),(4,0,0,b''),(4,0,0,b'')]]
        for frames in cases:
            with tempfile.TemporaryDirectory() as root:
                with self.assertRaises(ValueError):
                    consume_selected(io.BytesIO(wire(frames)),root,3,lambda *args:None)

    def test_metadata_and_samples_stay_in_ram_until_final_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            a=archive(Path(root)/'saved'); state=pack_state([0,1,2])
            a.selected_layer(0,1,[state],.1,{'0':None})
            a.selected_layer(1,1000,[state*800,state*200],.2,{'0':None,'1':None})
            a.selected_layer(2,1000000,[state*1000,state*1000],.3,{'0':None,'1':None})
            self.assertFalse(list(a.root.rglob('*.bin')))
            m=json.loads(a.snapshot(False,'resource').read_text())
            self.assertEqual(len(m['layers']),3)
            self.assertEqual(sum(x['states'] for x in m['files']),1000)
            self.assertEqual(m['files'][0]['depth'],2)
            self.assertFalse(m['files'][0]['full_layer'])
            self.assertEqual(m['files'][0]['first_state_ordinal'],0)
            self.assertLessEqual(sum(p.stat().st_size for p in a.tail.glob('*.bin')),16008)

    def test_complete_whole_terminal_and_all_small_layers(self):
        with tempfile.TemporaryDirectory() as root:
            a=archive(Path(root)/'saved'); state=pack_state([0,1,2])
            for at,count in enumerate([1,999,5000,3000]):
                a.selected_layer(at,count,[state*min(count,1000)],.1,{'0':None})
            terminal=Path(root)/'terminal';terminal.write_bytes(state*3000)
            a.terminal(3,[terminal])
            m=json.loads(a.snapshot(True,'exhausted').read_text())
            self.assertEqual([(x['depth'],x['states']) for x in m['files']],[(0,1),(1,999),(3,3000)])
            self.assertTrue(all(x['full_layer'] for x in m['files']))
            self.assertFalse((a.tail/'layer-000002.bin').exists())
            for f in m['files']:
                self.assertEqual(hashlib.sha256((a.root/f['path']).read_bytes()).hexdigest(),f['sha256'])

    def test_terminal_descriptors_adopt_files_without_reading_or_recopy(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as root:
            a=archive(Path(root)/'saved');state=pack_state([0,1,2])
            a.selected_layer(0,3000,[state*1000],.1,{'0':None,'1':None})
            path=Path(root)/'terminal';data=state*2500;path.write_bytes(data)
            inode=path.stat().st_ino
            a.terminal(0,[(path,2500,hashlib.sha256(data).hexdigest()),state*500])
            with patch('scripts.selected_tail.hashlib.file_digest',side_effect=AssertionError('unexpected reread')):
                manifest=json.loads(a.snapshot(True).read_text())
            self.assertFalse(path.exists())
            self.assertEqual([x['first_state_ordinal'] for x in manifest['files']],[0,2500])
            self.assertEqual(sum(x['states'] for x in manifest['files']),3000)
            self.assertEqual((a.root/manifest['files'][0]['path']).stat().st_ino,inode)
            self.assertTrue(all(x['layer_complete'] for x in manifest['files']))

    def test_invalid_terminal_rolls_back_and_allows_incomplete_snapshot(self):
        with tempfile.TemporaryDirectory() as root:
            a=archive(Path(root)/'saved'); state=pack_state([0,1,2])
            a.selected_layer(0,1,[state],.1,{'0':None})
            a.selected_layer(1,3000,[state*1000],.1,{'0':None})
            a.terminal(1,[state*2000])
            with self.assertRaisesRegex(ValueError,'shape'):a.snapshot(True)
            self.assertFalse(list(a.tail.glob('*.bin')))
            self.assertEqual(a.retained,[])
            m=json.loads(a.snapshot(False,'export failed').read_text())
            self.assertEqual([(f['depth'],f['states']) for f in m['files']],[(1,1000)])

    def test_terminal_missing_fails_before_any_payload_write(self):
        with tempfile.TemporaryDirectory() as root:
            a=archive(Path(root)/'saved');state=pack_state([0,1,2])
            a.selected_layer(0,3000,[state*1000],.1,{'0':None})
            with self.assertRaisesRegex(ValueError,'missing'):a.snapshot(True)
            self.assertFalse(list(a.tail.glob('*.bin')))
            self.assertFalse(json.loads(a.snapshot(False).read_text())['files'][0]['full_layer'])


if __name__ == '__main__': unittest.main()
