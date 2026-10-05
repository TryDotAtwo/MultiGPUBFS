import json
import tempfile
import unittest
from pathlib import Path
from scripts.bfs_tail_archive import TailArchive, pack_state, packed_width, publish_snapshot


class TailTests(unittest.TestCase):
    def test_last_complete_and_thousand_state_incomplete(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            archive=TailArchive(Path(d)/'run',n=3,r=1,start=[0,1,2],
                actions={'L':'left','R':'right','X':'swap'},program_commit='abc',
                launch_config={'retention_policy':'last_complete_small_1000'},
                sample_interval_seconds=.05)
            for depth,count in enumerate([1,1000,1001,3000]):
                archive.completed_layer(depth,count,[pack_state([0,1,2])*count],.1,{'0':None})
            complete=json.loads(archive.snapshot(True,'exhausted').read_text())
            self.assertEqual([(x['depth'],x['states']) for x in complete['files']],[(0,1),(1,1000),(3,3000)])
            self.assertTrue(all(x['full_layer'] for x in complete['files']))
            incomplete=json.loads(archive.snapshot(False,'resource').read_text())
            self.assertEqual(sum(x['states'] for x in incomplete['files']),1000)
            self.assertEqual(incomplete['files'][0]['first_state_ordinal'],2000)
            self.assertFalse(incomplete['files'][0]['full_layer'])
            self.assertEqual(len(incomplete['layers']),4)
            for x in incomplete['files']:
                data=(archive.root/x['path']).read_bytes()
                self.assertEqual(len(data),x['bytes'])
                self.assertEqual(hashlib.sha256(data).hexdigest(),x['sha256'])
            archive.release_working_tail()

    def test_fixed_five_layers_for_both_statuses_without_byte_target(self):
        with tempfile.TemporaryDirectory() as d:
            archive=self.make(Path(d)/'run',retained_layers=5,complete_bytes=10000,incomplete_bytes=8)
            for depth in range(9):
                archive.completed_layer(depth,2,[pack_state([0,1,2])*2],.1,{'0':100})
            for complete in (False,True):
                manifest=json.loads(archive.snapshot(complete,'test').read_text())
                self.assertEqual([f['depth'] for f in manifest['files']],[4,5,6,7,8])
                self.assertTrue(all(f['full_layer'] and f['states']==2 for f in manifest['files']))
                self.assertEqual(len(manifest['layers']),9)
                self.assertEqual(manifest['retention']['layers'],5)

    def test_fixed_layer_count_must_be_positive_integer(self):
        with tempfile.TemporaryDirectory() as d:
            for count in (0,-1,True,1.5):
                with self.assertRaises(ValueError):self.make(Path(d)/'run',retained_layers=count)
    def make(self, root, **kw):
        return TailArchive(root, n=3, r=1, start=[0, 1, 2],
                           actions={'L': 'rotate left', 'R': 'rotate right', 'X': 'swap first two'},
                           program_commit='abc', launch_config={'world': 2},
                           sample_interval_seconds=.1, **kw)

    def test_final_only_keeps_metadata_in_ram_until_seal(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            archive=self.make(Path(d)/'run',snapshot_each_layer=False)
            with patch('scripts.bfs_tail_archive.os.fsync') as sync:
                archive.completed_layer(0,1,[pack_state([0,1,2])],.2,{'0':100})
                self.assertEqual(sync.call_count,0)
                self.assertEqual(list(archive.root.glob('snapshot-*')),[])
                self.assertEqual(archive.manifest['last_completed_layer'],0)
                stored=json.loads((archive.root/'manifest.json').read_text())
                self.assertEqual(stored['last_completed_layer'],-1)
                final=json.loads(archive.snapshot(True,'exhausted').read_text())
                self.assertGreater(sync.call_count,0)
                self.assertEqual(final['last_completed_layer'],0)
                self.assertEqual(final['files'][0]['states'],1)

    def test_rank_parts_are_adopted_without_recopy_and_trimmed_by_depth(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            archive=self.make(root/'run',retained_layers=3,snapshot_each_layer=False)
            for depth in range(5):
                parts=[]
                for rank,count in enumerate((2,3)):
                    path=root/f'input-{depth}-{rank}'
                    data=pack_state([0,1,2])*count;path.write_bytes(data)
                    inode=path.stat().st_ino
                    parts.append((count,path,hashlib.sha256(data).hexdigest()))
                archive.completed_parts(depth,parts,.2,{'0':100,'1':120})
                self.assertTrue(all(not part[1].exists() for part in parts))
                self.assertEqual((archive.tail/f'layer-{depth:06d}-rank-0001.bin').stat().st_ino,inode)
            self.assertEqual({x['depth'] for x in archive.retained},{2,3,4})
            self.assertEqual(len(archive.retained),6)
            final=json.loads(archive.snapshot(True,'exhausted').read_text())
            self.assertEqual(len(final['layers']),5)
            for depth in (2,3,4):
                entries=[x for x in final['files'] if x['depth']==depth]
                self.assertEqual([x['first_state_ordinal'] for x in entries],[0,2])
                self.assertEqual(sum(x['states'] for x in entries),5)
                self.assertTrue(all(x['layer_complete'] for x in entries))
            archive.release_working_tail()
            self.assertTrue(all((archive.root/x['path']).exists() for x in final['files']))

    def test_packing(self):
        self.assertEqual(pack_state([1, 2, 3]), bytes.fromhex('2103000000000000'))
        self.assertEqual(packed_width(17, 16), 16)
        self.assertEqual(packed_width(32,17),32)

    def test_complete_keeps_three_and_byte_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            archive = self.make(Path(d)/'run', complete_bytes=40, incomplete_bytes=16)
            for depth in range(8):
                archive.completed_layer(depth, 1, [pack_state([0, 1, 2])], .2, {'0': 100, '1': 120})
            manifest = json.loads(archive.snapshot(True, 'graph exhausted').read_text())
            self.assertEqual([f['depth'] for f in manifest['files']], [3, 4, 5, 6, 7])
            self.assertTrue(all(f['full_layer'] for f in manifest['files']))
            self.assertEqual(len(manifest['layers']), 8)

    def test_incomplete_suffix_and_manifest_last(self):
        with tempfile.TemporaryDirectory() as d:
            archive = self.make(Path(d)/'run', complete_bytes=80, incomplete_bytes=17)
            archive.completed_layer(0, 3, [pack_state([0, 1, 2])*3], .2, {'0': 100})
            path = archive.snapshot(reason='capacity')
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest['files'][0]['bytes'], 16)
            self.assertFalse(manifest['files'][0]['full_layer'])
            self.assertEqual(manifest['files'][0]['first_state_ordinal'], 1)
            uploaded = []
            publish_snapshot(path, lambda local, remote: uploaded.append(remote))
            self.assertEqual(uploaded[-1], 'manifest.json')
            def fail(local, remote):
                raise OSError('offline')
            with self.assertRaises(OSError):
                publish_snapshot(path, fail)

    def test_final_tail_release_preserves_exact_complete_and_partial_snapshots(self):
        for complete in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                archive = self.make(Path(directory)/'run', complete_bytes=80, incomplete_bytes=16)
                for depth in range(4):
                    archive.completed_layer(depth, 3, [pack_state([0,1,2])*3], .2, {'0':None})
                path = archive.snapshot(complete, 'stopped')
                manifest = json.loads(path.read_text())
                saved = [(archive.root/f['path'], (archive.root/f['path']).read_bytes())
                         for f in manifest['files']]
                archive.release_working_tail()
                self.assertFalse(list(archive.tail.glob('*.bin')))
                for payload, data in saved:
                    self.assertEqual(payload.read_bytes(), data)
                self.assertEqual(json.loads(path.read_text()), manifest)
                with self.assertRaisesRegex(ValueError, 'released'):
                    archive.snapshot()
                with self.assertRaisesRegex(ValueError, 'released'):
                    archive.completed_layer(4, 1, [pack_state([0,1,2])], .2, {'0':None})

    def test_failed_layer_does_not_advance(self):
        with tempfile.TemporaryDirectory() as d:
            archive = self.make(Path(d)/'run')
            with self.assertRaises(ValueError):
                archive.completed_layer(0, 2, [pack_state([0, 1, 2])], .2, {'0': 100})
            self.assertEqual(archive.manifest['last_completed_layer'], -1)


if __name__ == '__main__':
    unittest.main()
