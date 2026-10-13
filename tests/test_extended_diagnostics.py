import errno,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from multigpubfs.system_diagnostics import device_access,elf,_valid_contrast,cgroups,survey
from multigpubfs.diagnostic_analysis import analyze
from multigpubfs.diagnostics import Runner

def step(name,data=None,status='PASS',**kwargs):
 out={'name':name,'status':status,**kwargs}
 if data is not None:out['result']={'data':data}
 return out

class ExtendedDiagnosticTests(unittest.TestCase):
 def test_no_seccomp_or_cgroup_guess(self):
  report={'steps':[step('environment-evidence',{'process_security':{'Seccomp':'2'},'cgroups':{'records':[{'version':2}]}})]};a=analyze(report)
  self.assertFalse(a['confirmed_findings']);self.assertFalse(a['hypotheses']);self.assertEqual(len(a['unknowns']),2)
 def test_actual_denial_is_fact_not_policy_guess(self):
  data={'device_nodes':[{'path':'/dev/nvidiactl','open':{'status':'FAIL','errno':errno.EPERM}}]};a=analyze({'steps':[step('environment-evidence',data),step('driver-api',{'status':'FAIL'},'FAIL')]})
  self.assertEqual(a['confirmed_findings'][0]['code'],'DEVICE_OPEN_DENIED');self.assertEqual(a['hypotheses'][0]['code'],'GPU_ACCESS_RESTRICTION')
 def test_nvml_count_is_not_gpu_permission(self):
  a=analyze({'steps':[step('nvml-evidence',{'status':'PASS','management_device_count':8})]});self.assertIn('not proof',a['confirmed_findings'][0]['statement'])
 def test_torch_native_contrast_has_limited_conclusion(self):
  a=analyze({'steps':[step('torch-evidence',{'status':'PASS'}),step('driver-api',{'status':'FAIL'},'FAIL')]});self.assertIn('TORCH_CUDA_WORKS_NATIVE_PROBE_FAILS',[x['code'] for x in a['confirmed_findings']])
 def test_loader_success_requires_baseline_reproduction(self):
  variants=[{'name':'variant','status':'PASS'}];a=analyze({'steps':[step('loader-contrasts',{'results':variants})]});self.assertEqual(a['hypotheses'][0]['code'],'TRANSIENT_OR_LOADER_FAILURE')
  variants.append({'name':'baseline-recheck','status':'FAIL'});a=analyze({'steps':[step('loader-contrasts',{'results':variants})]});self.assertEqual(a['confirmed_findings'][0]['code'],'LOADER_CONTRAST_REPRODUCED')
 def test_original_allocation_must_be_preserved(self):
  env={'CUDA_VISIBLE_DEVICES':'0','NVIDIA_VISIBLE_DEVICES':'GPU-secret','CUDA_DEVICE_ORDER':'PCI_BUS_ID'};changed=dict(env,LD_PRELOAD='different');self.assertTrue(_valid_contrast(changed,env));changed.pop('CUDA_VISIBLE_DEVICES');self.assertFalse(_valid_contrast(changed,env))
 def test_permission_error_does_not_mean_device_absent(self):
  with patch('pathlib.Path.stat',side_effect=PermissionError(errno.EACCES,'denied')):self.assertIsNone(device_access(Path('/dev/nvidiactl'))['exists'])
 def test_mask_is_observed_without_suggesting_bypass(self):
  a=analyze({'steps':[step('environment-evidence',{'environment':{'CUDA_VISIBLE_DEVICES':'-1'}})]});self.assertEqual(a['confirmed_findings'][0]['code'],'CUDA_VISIBILITY_DISABLED');self.assertIn('do not bypass',a['user_next_actions'][0])
 def test_no_cuda_discovery_no_bfs_blame(self):
  a=analyze({'steps':[step('cuda-discovery',status='FAIL')]});self.assertIn('not reached',a['ruled_out_for_this_observation'][0]['statement'])
 def test_cgroup_path_cannot_escape_mount(self):
  with patch('multigpubfs.system_diagnostics._read',return_value={'status':'READ','text':'0::/../../outside\n'}):self.assertFalse(cgroups()['records'])
 def test_bad_elf_identified_without_loading(self):
  with tempfile.TemporaryDirectory() as tmp:
   file=Path(tmp)/'fake.so';file.write_bytes(b'plain text');self.assertFalse(elf(file)['valid'])
 def test_unavailable_auxiliary_is_skipped(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Runner(Path(tmp)/'report',3);folder=r.root/'optional';folder.mkdir();(folder/'result.json').write_text(json.dumps({'data':{'status':'UNAVAILABLE','reason':'not installed'}}))
   result=r.run('optional',[sys.executable,'-c','pass'],optional=True);self.assertEqual(result['status'],'SKIPPED');self.assertEqual(result['reason'],'not installed');r.finish()
 def test_partial_snapshot_retained_and_named_phase(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Runner(Path(tmp)/'report',.2);folder=r.root/'driver-api';folder.mkdir();(folder/'driver.json').write_text(json.dumps({'status':'RUNNING','driver':{'api_version':11080}}))
   result=r.run('driver-api',[sys.executable,'-c','import time;print("PHASE cuInit",flush=True);time.sleep(30)']);self.assertEqual(result['last_phase'],'cuInit');self.assertEqual(result['partial_evidence']['driver.json']['status'],'RUNNING');self.assertIn('cuInit',r.finish()['analysis']['unknowns'][0])
 def test_local_survey_needs_no_gpu_or_tools(self):
  with tempfile.TemporaryDirectory() as tmp,patch('multigpubfs.system_diagnostics.library_candidates',return_value=[]):
   result=survey(Path(tmp));self.assertEqual(result['status'],'PASS');self.assertEqual(result['loopback']['status'],'PASS');self.assertTrue((Path(tmp)/'environment.json').is_file())
