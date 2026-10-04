import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from scripts.bfs_tail_archive import TailArchive,pack_state
from scripts.tail_upload import Publisher,retry_upload


class UploadTests(unittest.TestCase):
    def test_latest_queued_snapshot_replaces_only_pending_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); archive=self.archive(root/'run')
            entered,release=threading.Event(),threading.Event()
            depths=[]
            class Api:
                def upload_file(self,**kw):
                    if kw['path_in_repo'].endswith('/manifest.json'):
                        depths.append(json.loads(Path(kw['path_or_fileobj']).read_text())['last_completed_layer'])
                    elif not entered.is_set():
                        entered.set()
                        if not release.wait(5): raise TimeoutError('test release deadline')
                    return 'receipt'
            publisher=Publisher(root/'pins','test/data','run1',api=Api())
            try:
                publisher.enqueue(archive.snapshot())
                self.assertTrue(entered.wait(3))
                archive.completed_layer(1,1,[pack_state([1,2,0])],.1,{'0':124})
                publisher.enqueue(archive.snapshot())
                archive.completed_layer(2,1,[pack_state([2,0,1])],.1,{'0':125})
                publisher.enqueue(archive.snapshot())
                self.assertEqual(len(list((root/'pins').glob('*/manifest.json'))),2)
            finally:
                release.set()
                publisher.finish()
            self.assertEqual(depths,[0,2])
            self.assertEqual(publisher.pending,0)

    def test_live_parquet_uses_stable_shards_and_payload_before_manifest(self):
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=self.archive(root/'run');uploaded={};order=[]
            class Api:
                def upload_file(self,**kw):
                    remote=kw['path_in_repo'];data=Path(kw['path_or_fileobj']).read_bytes()
                    uploaded[remote]=data;order.append(remote);return 'receipt'
            publisher=Publisher(root/'pins','fixture','run',api=Api(),
                storage_format='parquet',prefix='tail-live/run')
            publisher.enqueue(archive.snapshot());publisher.finish()
            self.assertEqual(order[-1],'tail-live/run/manifest.json')
            self.assertEqual(set(uploaded),{'tail-live/run/manifest.json','tail-live/run/states-00000.parquet'})
            local=root/'download.parquet';local.write_bytes(uploaded['tail-live/run/states-00000.parquet'])
            self.assertEqual(pq.read_table(local)['state'].to_pylist(),[pack_state([0,1,2])])
            receipt=json.loads((root/'pins/receipt.json').read_text())
            self.assertEqual(receipt['payload_paths'],['tail-live/run/states-00000.parquet'])

    def test_transport_retry_is_bounded(self):
        calls=[]
        RemoteProtocolError=type('RemoteProtocolError',(Exception,),{})
        def operation():
            calls.append(1)
            raise RemoteProtocolError('disconnected')
        with patch('scripts.tail_upload.time.sleep'):
            with self.assertRaises(RemoteProtocolError):
                retry_upload(operation)
        self.assertEqual(len(calls),4)

    def test_transport_retry_can_recover(self):
        calls=[]
        RemoteProtocolError=type('RemoteProtocolError',(Exception,),{})
        def operation():
            calls.append(1)
            if len(calls)==1: raise RemoteProtocolError('disconnected')
            return 'receipt'
        with patch('scripts.tail_upload.time.sleep'):
            self.assertEqual(retry_upload(operation),'receipt')
        self.assertEqual(len(calls),2)

    def test_non_transient_error_is_not_retried(self):
        calls=[]
        def operation():
            calls.append(1)
            raise ValueError('invalid operation')
        with self.assertRaises(ValueError): retry_upload(operation)
        self.assertEqual(len(calls),1)

    def archive(self,root):
        archive=TailArchive(root,n=3,r=1,start=[0,1,2],actions={'L':'left','R':'right','X':'swap'},
            program_commit='abc',launch_config={},sample_interval_seconds=.05)
        archive.completed_layer(0,1,[pack_state([0,1,2])],.1,{'0':123})
        return archive

    def test_batch_payload_commit_precedes_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            archive=self.archive(root/'run')
            calls=[]
            class Api:
                def create_commit(self, **kw):
                    operations=kw['operations']
                    self.assert_payloads=all(not op.path_in_repo.endswith('/manifest.json') for op in operations)
                    calls.append(('payloads',len(operations),self.assert_payloads))
                def upload_file(self, **kw):
                    calls.append(('manifest',kw['path_in_repo']))
                    return 'receipt'
            publisher=Publisher(root/'pins','test/data','run1',api=Api())
            publisher.enqueue(archive.snapshot())
            publisher.finish()
            self.assertEqual(calls[0],('payloads',1,True))
            self.assertEqual(calls[1][0],'manifest')

    def test_failed_batch_retains_snapshot_without_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            archive=self.archive(root/'run')
            class Api:
                def create_commit(self, **kw):
                    raise OSError('batch failed')
                def upload_file(self, **kw):
                    raise AssertionError('manifest must not be published')
            publisher=Publisher(root/'pins','test/data','run1',api=Api())
            publisher.enqueue(archive.snapshot())
            with self.assertRaises(RuntimeError):
                publisher.finish()
            self.assertTrue(list((root/'pins').glob('*/manifest.json')))

    def test_pins_survive_next_snapshot_and_manifest_is_last(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            archive=self.archive(root/'run')
            calls=[]
            class Api:
                def upload_file(self,**kw):
                    self_data=Path(kw['path_or_fileobj']).read_bytes()
                    calls.append((kw['path_in_repo'],self_data))
                    return 'https://huggingface.co/datasets/test/data/commit/abc'
            with patch('huggingface_hub.HfApi',return_value=Api()):
                publisher=Publisher(root/'pins','test/data','run1','fake-token')
                publisher.enqueue(archive.snapshot(reason='running'))
                archive.completed_layer(1,1,[pack_state([1,2,0])],.1,{'0':124})
                publisher.finish()
            self.assertTrue(calls[-1][0].endswith('/manifest.json'))
            self.assertEqual(json.loads(calls[-1][1])['last_completed_layer'],0)

    def test_failed_file_does_not_publish_manifest_and_retains_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            archive=self.archive(root/'run')
            calls=[]
            class Api:
                def upload_file(self,**kw):
                    calls.append(kw['path_in_repo'])
                    raise OSError('offline')
            with patch('huggingface_hub.HfApi',return_value=Api()):
                publisher=Publisher(root/'pins','test/data','run1','fake-token')
                publisher.enqueue(archive.snapshot())
                with self.assertRaises(RuntimeError):
                    publisher.finish()
            self.assertFalse(any(path.endswith('manifest.json') for path in calls))
            self.assertTrue(list((root/'pins').glob('*/manifest.json')))


if __name__=='__main__':
    unittest.main()


class WrappedTransportRetryTests(unittest.TestCase):
    def test_sdk_wrapped_network_failure_retries_same_operation(self):
        from scripts.tail_upload import retry_upload
        class RemoteProtocolError(Exception): pass
        calls=[]
        def operation():
            calls.append('same payload')
            if len(calls)==1:
                try: raise RemoteProtocolError('disconnected')
                except RemoteProtocolError as error:
                    raise RuntimeError('Error while uploading a file to the Hub.') from error
            return 'published'
        with patch('scripts.tail_upload.time.sleep') as sleep:
            self.assertEqual(retry_upload(operation),'published')
            self.assertEqual(calls,['same payload','same payload'])
            sleep.assert_called_once_with(1)

    def test_wrapped_permanent_failure_does_not_retry(self):
        from scripts.tail_upload import retry_upload
        def operation():
            try: raise ValueError('payload checksum mismatch')
            except ValueError as error: raise RuntimeError('upload failed') from error
        with patch('scripts.tail_upload.time.sleep') as sleep:
            with self.assertRaises(RuntimeError):retry_upload(operation)
            sleep.assert_not_called()

    def test_implicit_context_and_cyclic_cause_are_not_transport_evidence(self):
        from scripts.tail_upload import transient_upload_error
        class RemoteProtocolError(Exception): pass
        error=RuntimeError('unknown failure');error.__context__=RemoteProtocolError()
        self.assertFalse(transient_upload_error(error))
        error.__cause__=error
        self.assertFalse(transient_upload_error(error))
