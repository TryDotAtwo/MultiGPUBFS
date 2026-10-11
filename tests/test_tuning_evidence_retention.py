import tempfile,unittest
from pathlib import Path
from multigpubfs.autotune import _evidence_directory
class EvidenceRetention(unittest.TestCase):
 def test_failure_keeps_original_producer_log_and_trace(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory)/'evidence'
   with self.assertRaisesRegex(RuntimeError,'producer failure') as caught:
    with _evidence_directory({'MGBFS_TUNING_EVIDENCE_ROOT':str(root)},'test-') as temporary:
     (Path(temporary)/'rank-0.log').write_text('original native error')
     raise RuntimeError('producer failure')
   logs=list(root.glob('*/rank-0.log'));self.assertEqual(len(logs),1);self.assertEqual(logs[0].read_text(),'original native error')
   self.assertTrue(list(root.glob('*/failure-context.json')));self.assertFalse(Path(temporary).exists())
   self.assertTrue(any('Tuning evidence:' in note for note in caught.exception.__notes__))
 def test_success_cleans_temporary_workspace(self):
  with _evidence_directory({},'test-') as temporary:Path(temporary,'receipt').write_text('ok')
  self.assertFalse(Path(temporary).exists())
