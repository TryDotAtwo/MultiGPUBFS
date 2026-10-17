import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

class NativeFailureEvidenceTests(unittest.TestCase):
 def fixture(self, root):
  worker=root/'ephemeral';worker.mkdir()
  (worker/'worker.log').write_text('earlier job\nCURRENT FAILED JOB\n')
  phase={'schema':1,'pid':123,'sequence':'23','phase':'bfs','layer':7,'last_completed_layer':6,'native_call':'cudaMemcpyAsync'}
  (worker/'phase-00000023.json').write_text(json.dumps(phase))
  (worker/'response-00000023.json').write_text(json.dumps({'schema':1,'sequence':23,'ok':False,'error':'GENERIC_CUDA_999 call=cudaMemcpyAsync'}))
  (worker/'job-00000023.json').write_text('{"env":{"MGBFS_PRIVATE_TOKEN":"never copy"}}')
  return SimpleNamespace(pid=123,sequence=23,returncode=1,log_offset=len('earlier job\n'),worker=SimpleNamespace(root=worker,rank=5,device=5))
 def test_error_preserves_phase_response_and_current_log_before_worker_cleanup(self):
  from multigpubfs.failure_evidence import preserve_native_failure
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);job=self.fixture(root);out=root/'pilot-0'
   receipt=preserve_native_failure([job],out,'NATIVE_DISTRIBUTED_RANK_FAILED [(5, 1)]',when='before_cleanup')
   self.assertEqual(receipt['status'],'CAPTURED')
   rank=receipt['ranks'][0];self.assertEqual((rank['rank'],rank['device'],rank['sequence']),(5,5,23))
   evidence=out/'failure-evidence'/'rank-5'
   self.assertEqual(json.loads((evidence/'phase.json').read_text())['layer'],7)
   self.assertIn('GENERIC_CUDA_999',(evidence/'response.json').read_text())
   self.assertEqual((evidence/'worker.log').read_text(),'CURRENT FAILED JOB\n')
   self.assertFalse((evidence/'job.json').exists())
   self.assertNotIn('never copy',''.join(p.read_text() for p in (out/'failure-evidence').rglob('*') if p.is_file()))
 def test_final_capture_retains_before_cleanup_snapshot(self):
  from multigpubfs.failure_evidence import preserve_native_failure
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);job=self.fixture(root);out=root/'pilot'
   preserve_native_failure([job],out,'failure',when='before_cleanup')
   (job.worker.root/'phase-00000023.json').unlink()
   preserve_native_failure([job],out,'failure',when='after_cleanup')
   report=json.loads((out/'failure-evidence'/'summary.json').read_text())
   self.assertEqual([v['when'] for v in report['captures']],['before_cleanup','after_cleanup'])
   self.assertTrue((out/'failure-evidence'/'rank-5'/'phase.json').is_file())
 def test_missing_files_and_one_shot_process_are_reported_without_masking_original_error(self):
  from multigpubfs.failure_evidence import preserve_native_failure
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);out=root/'pilot';out.mkdir()
   (out/'rank-0.phase.json').write_text('{"phase":"bfs","layer":2}')
   job=SimpleNamespace(pid=456,returncode=-15)
   report=preserve_native_failure([job],out,'original CUDA failure',when='before_cleanup')
   self.assertEqual(report['error'],'original CUDA failure')
   self.assertEqual(report['ranks'][0]['pid'],456)
   self.assertTrue((out/'failure-evidence'/'rank-0'/'phase.json').is_file())
 def test_corrupt_phase_is_kept_as_raw_evidence(self):
  from multigpubfs.failure_evidence import preserve_native_failure
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);job=self.fixture(root)
   (job.worker.root/'phase-00000023.json').write_text('interrupted-json')
   preserve_native_failure([job],root/'pilot','original',when='after_cleanup')
   self.assertEqual((root/'pilot/failure-evidence/rank-5/phase.json').read_text(),'interrupted-json')
 def test_diagnostic_file_copy_is_bounded(self):
  from multigpubfs.failure_evidence import preserve_native_failure
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);job=self.fixture(root)
   (job.worker.root/'worker.log').write_text('old'*10000+'END-OF-FAILED-JOB')
   report=preserve_native_failure([job],root/'pilot','failure',when='before_cleanup',max_bytes=1024)
   data=(root/'pilot/failure-evidence/rank-5/worker.log').read_bytes()
   self.assertLessEqual(len(data),1024);self.assertTrue(data.endswith(b'END-OF-FAILED-JOB'))
   self.assertTrue(report['ranks'][0]['files']['worker.log']['truncated'])


class DiagnosticFailureBundleTests(unittest.TestCase):
 def test_doctor_archive_keeps_native_phase_response_and_summary_without_job_environment(self):
  import contextlib,io,zipfile
  from multigpubfs.diagnostics import Runner
  with tempfile.TemporaryDirectory() as temp:
   root=Path(temp);runner=Runner(root,60)
   evidence=root/'pilot-0/failure-evidence';rank=evidence/'rank-1';rank.mkdir(parents=True)
   for name in ['phase.json','response.json']:(rank/name).write_text('{"code":999}')
   (rank/'worker.log').write_text('MGBFS_NATIVE_ERROR')
   (evidence/'summary.json').write_text('{"schema":1}')
   (rank/'job.json').write_text('{"env":{"PRIVATE":"never bundle"}}')
   with contextlib.redirect_stdout(io.StringIO()):runner.finish()
   with zipfile.ZipFile(root/'diagnostics.zip') as archive:
    for name in ['rank-1/phase.json','rank-1/response.json','rank-1/worker.log','summary.json']:
     self.assertIn('pilot-0/failure-evidence/'+name,archive.namelist())
    self.assertNotIn('pilot-0/failure-evidence/rank-1/job.json',archive.namelist())

if __name__=='__main__':unittest.main()
