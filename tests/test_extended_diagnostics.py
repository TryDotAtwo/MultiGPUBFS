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

 def test_spawn_failure_is_not_cuda_failure(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Runner(Path(tmp)/'report',3)
   with patch('multigpubfs.diagnostics.subprocess.Popen',side_effect=PermissionError(errno.EPERM,'spawn denied')):result=r.run('blocked',[sys.executable,'-c','pass'])
   self.assertEqual(result['errno'],errno.EPERM);report=r.finish();self.assertEqual(report['analysis']['confirmed_findings'][0]['code'],'DIAGNOSTIC_PROCESS_SPAWN_FAILED');self.assertIn('parent_resource_limits',report)
 def test_v1_wildcards_and_malformed_policy(self):
  from multigpubfs.diagnostic_analysis import v1_device_allows
  self.assertTrue(v1_device_allows('a *:* rwm',195,255));self.assertTrue(v1_device_allows('c 195:* rw',195,255));self.assertFalse(v1_device_allows('c 195:0 rw',195,255));self.assertIsNone(v1_device_allows('malformed',195,0))
 def test_visibility_contrast_reproduces_original_failure(self):
  from multigpubfs.system_diagnostics import contrasts
  original={'CUDA_VISIBLE_DEVICES':'1','NVIDIA_VISIBLE_DEVICES':'restricted','LD_PRELOAD':'fake.so'}
  with tempfile.TemporaryDirectory() as tmp:
   parent=Path(tmp);root=parent/'matrix';root.mkdir();count=parent/'cuda-discovery';count.mkdir();(count/'result.json').write_text('{"status":"FAIL"}')
   def fake(command,**kwargs):
    good='LD_PRELOAD' not in kwargs['env'];self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'1');self.assertEqual(kwargs['env']['NVIDIA_VISIBLE_DEVICES'],'restricted')
    if '--output' in command:Path(command[command.index('--output')+1]).write_text(json.dumps({'status':'PASS' if good else 'FAIL'}));return subprocess.CompletedProcess(command,0,'','')
    return subprocess.CompletedProcess(command,0 if good else 1,'2' if good else '', 'fixture failure' if not good else '')
   with patch('multigpubfs.native_distribution.native_runtime',return_value=('/fixture/native',original)),patch('multigpubfs.system_diagnostics.subprocess.run',side_effect=fake):result=contrasts(root,60)
   self.assertEqual([x['status'] for x in result['results']],['PASS','FAIL']);self.assertEqual(original['LD_PRELOAD'],'fake.so')
 def test_kernel_image_error_has_own_explanation(self):
  a=analyze({'steps':[step('gpu-0-kernel',status='FAIL',log_tail='CUDA status 209: no kernel image is available for execution on the device')]});self.assertEqual(a['confirmed_findings'][0]['code'],'NATIVE_KERNEL_IMAGE_UNAVAILABLE')
 def test_sigkill_is_not_proof_of_oom(self):
  a=analyze({'steps':[step('torch-evidence',status='WARNING',exit_code=-9)]});self.assertFalse(a['confirmed_findings']);self.assertEqual(a['hypotheses'][0]['code'],'PROCESS_KILLED_NOT_GPU_PROOF')
 def test_driver_version_difference_is_not_automatic_mismatch(self):
  a=analyze({'steps':[step('driver-api',{'status':'PASS','driver':{'resolved_library':'/compat/libcuda.so.580.178.04'}}),step('nvml-evidence',{'status':'PASS','driver_version':'535.129.03'})]});self.assertEqual(a['confirmed_findings'][0]['code'],'DRIVER_LIBRARY_MANAGEMENT_VERSION_DIFFERENCE');self.assertIn('valid forward compatibility',a['confirmed_findings'][0]['statement']);self.assertIsNone(a['primary_failure'])
 def test_optional_torch_respects_memory_headroom(self):
  from multigpubfs.system_diagnostics import torch_probe
  with patch('multigpubfs.system_diagnostics.importlib.util.find_spec',return_value=object()),patch('multigpubfs.system_diagnostics.cgroups',return_value={'records':[{'files':{'memory.max':'1024','memory.current':'1000'}}]}):self.assertEqual(torch_probe()['status'],'UNAVAILABLE')
