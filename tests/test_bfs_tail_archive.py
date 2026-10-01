import json
import tempfile
import unittest
from pathlib import Path
from scripts.bfs_tail_archive import TailArchive, pack_state, packed_width, publish_snapshot


class TailTests(unittest.TestCase):
    def make(self, root, **kw):
        return TailArchive(root, n=3, r=1, start=[0, 1, 2],
                           actions={'L': 'rotate left', 'R': 'rotate right', 'X': 'swap first two'},
                           program_commit='abc', launch_config={'world': 2},
                           sample_interval_seconds=.1, **kw)

    def test_packing(self):
        self.assertEqual(pack_state([1, 2, 3]), bytes.fromhex('2103000000000000'))
        self.assertEqual(packed_width(17, 16), 16)
        with self.assertRaises(ValueError):
            packed_width(32, 17)

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

    def test_failed_layer_does_not_advance(self):
        with tempfile.TemporaryDirectory() as d:
            archive = self.make(Path(d)/'run')
            with self.assertRaises(ValueError):
                archive.completed_layer(0, 2, [pack_state([0, 1, 2])], .2, {'0': 100})
            self.assertEqual(archive.manifest['last_completed_layer'], -1)


if __name__ == '__main__':
    unittest.main()
