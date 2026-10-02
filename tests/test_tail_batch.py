import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from scripts.publish_tail_batch import plan,publish


class BatchTests(unittest.TestCase):
    def test_many_layers_publish_one_parquet_payload_before_manifest(self):
        from scripts.bfs_tail_archive import TailArchive
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);saved=root/'n4-m2'/'saved'
            archive=TailArchive(saved,n=4,r=2,start=[0,1,2,2],
                actions=dict(L='left',R='right',X='swap'),program_commit='fixture',
                launch_config={},sample_interval_seconds=.05,complete_bytes=100000)
            expected=[]
            for depth in range(50):
                state=depth.to_bytes(8,'little');expected.append(state)
                archive.completed_layer(depth,1,[state],.01,{'0':100})
            archive.snapshot(complete=True,reason='fixture complete')
            ledger=dict(configuration=dict(base=dict(run_id='parquet-fixture',
                archive_format='parquet')),cases={'n4-m2':dict(attempted=True)})
            (root/'sweep.json').write_text(json.dumps(ledger))
            payloads,manifests,_=plan(root)
            self.assertEqual(len(payloads),1)
            self.assertTrue(payloads[0][1].endswith('.parquet'))
            rows=pq.read_table(payloads[0][0]).to_pylist()
            self.assertEqual([x['state'] for x in rows],expected)
            result=json.loads(manifests[0][0].read_text())
            self.assertEqual(result['status'],'COMPLETE')
            self.assertEqual(len(result['layers']),50)
            self.assertEqual(len(result['retained_layers']),50)
            self.assertTrue(all(x['full_layer'] for x in result['retained_layers']))
            self.assertEqual(result['retained_packed_bytes'],400)

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

    def test_new_case_cannot_race_into_published_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            frozen=json.loads((root/'sweep.json').read_text());seen=[]
            class Api:
                def create_commit(self,**kwargs):
                    for op in kwargs['operations']:
                        if op.path_in_repo.endswith('/sweep.json'):
                            seen.append(json.loads(Path(op.path_or_fileobj).read_text()))
                    changed=json.loads(json.dumps(frozen))
                    changed['cases']['n4-m1']=dict(attempted=True,status='COMPLETE')
                    (root/'sweep.json').write_text(json.dumps(changed))
                    return 'receipt'
            publish(root,'test/data',Api())
            self.assertEqual(seen,[frozen])

    def test_total_larger_than_batch_is_split_before_any_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);saved,entry=self.fixture(root)
            entries=[]
            for i in range(3):
                path=saved/f'snapshot/{i}.bin';path.write_bytes(bytes([i])*8)
                entries.append(dict(path=f'snapshot/{i}.bin',bytes=8,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            (saved/'manifest.json').write_text(json.dumps(dict(files=entries,status='COMPLETE')))
            calls=[]
            class Api:
                def create_commit(self,**kwargs):
                    calls.append([op.path_in_repo for op in kwargs['operations']]);return 'receipt'
            result=publish(root,'test/data',Api(),max_batch_bytes=10)
            self.assertEqual(result['payload_commits'],3)
            self.assertEqual(result['bytes'],24)
            self.assertEqual(len(calls),4)
            self.assertTrue(all(not any(p.endswith('.json') for p in row) for row in calls[:3]))

    def test_traversal_and_byte_bound_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);saved,entry=self.fixture(root)
            with self.assertRaisesRegex(ValueError,'byte bound'):plan(root,max_bytes=7)
            entry['path']='../../outside.bin'
            (saved/'manifest.json').write_text(json.dumps(dict(files=[entry])))
            with self.assertRaisesRegex(ValueError,'unsafe'):plan(root)


if __name__=='__main__':unittest.main()
