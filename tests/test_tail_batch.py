import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from scripts.publish_tail_batch import plan,publish


class BatchTests(unittest.TestCase):
    def fixture(self,root):
        saved=root/'n3-m1/saved';(saved/'snapshot').mkdir(parents=True)
        payload=saved/'snapshot/layer.bin';payload.write_bytes(b'12345678')
        entry=dict(path='snapshot/layer.bin',bytes=8,sha256=hashlib.sha256(payload.read_bytes()).hexdigest())
        (saved/'manifest.json').write_text(json.dumps(dict(files=[entry],status='COMPLETE')))
        (root/'sweep.json').write_text(json.dumps(dict(configuration=dict(base=dict(run_id='batch1')),
            cases={'n3-m1':dict(attempted=True,status='COMPLETE')})))
        return saved,entry

    def test_payload_commit_precedes_all_manifests(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root);calls=[]
            class Api:
                def create_commit(self,**kwargs):
                    calls.append([op.path_in_repo for op in kwargs['operations']])
                    return 'receipt'
            result=publish(root,'test/data',Api())
            self.assertEqual(result['bytes'],8)
            self.assertEqual(calls[0],['tail-runs/batch1-n3-m1/snapshot/layer.bin'])
            self.assertIn('tail-runs/batch1-n3-m1/manifest.json',calls[1])
            self.assertEqual(len(calls),2)

    def test_corrupt_payload_refuses_publication(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);saved,_=self.fixture(root)
            (saved/'snapshot/layer.bin').write_bytes(b'corrupt!')
            with self.assertRaisesRegex(ValueError,'checksum'):plan(root)

    def test_traversal_and_byte_bound_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);saved,entry=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'byte bound'):plan(root,max_bytes=7)
            entry['path']='../../outside.bin'
            (saved/'manifest.json').write_text(json.dumps(dict(files=[entry])))
            with self.assertRaisesRegex(ValueError,'unsafe'):plan(root)


if __name__=='__main__':unittest.main()
