
import copy,json,zipfile,hashlib
from pathlib import Path
import pytest
from multigpubfs import recovery_diagnostics as d

def good():
 return {'status':'PASS','result':{'status':'PASS','data':{'devices':[0,1],
   'cpu_oracles_verified':True,'measured_autotune':True,'prefix_verified':True}},'cleanup_complete':True,'policy':{}}

@pytest.mark.parametrize('field',['devices','cpu_oracles_verified','measured_autotune','prefix_verified'])
def test_acceptance_requires_evidence(field):
 x=good();x['result']['data'].pop(field);assert not d.acceptance(x,[0,1])

@pytest.mark.parametrize('status',['FAIL','TIMEOUT','SKIPPED','WARNING'])
def test_acceptance_never_masks_failures(status):
 x=good();x['status']=status;assert not d.acceptance(x,[0,1])

def test_acceptance_exact_set_cleanup():
 assert d.acceptance(good(),[0,1]);assert not d.acceptance(good(),[0])
 x=good();x['cleanup_complete']=False;assert not d.acceptance(x,[0,1])

def production():return {'devices':[0,1],'layer_sizes':d.EXPECTED,'status':'INCOMPLETE','autotune':{'measured':True}}
def test_measured_prefix_is_not_full_graph():
 assert d.validate_production(production(),[0,1])['full_graph_complete'] is False
@pytest.mark.parametrize('mutation',[{'devices':[0]},{'layer_sizes':[1,6]},{'autotune':{}},{'status':'ERROR'},
 {'autotune':{'measured':True,'candidate_failures':['CUDA999']}},{'status':'COMPLETE'}])
def test_production_rejects_unverified(mutation):
 r=production();r.update(mutation)
 with pytest.raises(RuntimeError):d.validate_production(r,[0,1])

def test_health_failure_blocks_subsequent_gpu(monkeypatch,tmp_path):
 s=d.Supervisor(tmp_path/'run',30);s.devices=[0,1]
 monkeypatch.setattr(s,'case',lambda *a,**k:{'status':'FAIL'})
 assert not s.health('failed');assert s.blocked

def test_blocked_and_budget_do_not_spawn(monkeypatch,tmp_path):
 s=d.Supervisor(tmp_path/'run',30);s.blocked=True
 monkeypatch.setattr(s.runner,'run',lambda *a,**k:pytest.fail('spawned'))
 assert s.case('gpu',{},gpu=True)['reason']=='CUDA_HEALTH_BLOCKED'
 s.blocked=False;s.deadline=0
 assert s.case('budget',{},gpu=True)['reason']=='GLOBAL_DIAGNOSTIC_BUDGET_EXHAUSTED'

def test_archive_excludes_states_jobs_secrets(tmp_path):
 for name in ['report.json','states.json','job-secret.json','credentials.json']:(tmp_path/name).write_text('{}')
 (tmp_path/'rank-0.log').write_text('phase')
 d.package(tmp_path)
 with zipfile.ZipFile(tmp_path/'diagnostics.zip') as z:
  assert 'report.json' in z.namelist();assert 'rank-0.log' in z.namelist()
  assert not set(['states.json','job-secret.json','credentials.json'])&set(z.namelist())
  manifest=json.loads(z.read('manifest.json'))
  for item in manifest['files']:assert hashlib.sha256(z.read(item['path'])).hexdigest()==item['sha256']

def test_findings_do_not_claim_unique_hardware_cause(tmp_path):
 (tmp_path/'rank-0.log').write_text('IBV_WC_LOC_PROT_ERR CUDA_999')
 rows=d.findings(tmp_path,[])
 assert {x['code'] for x in rows}=={'IB_LOCAL_PROTECTION','CUDA_UNKNOWN'}
 assert all('not established' in x['certainty'] for x in rows)

def test_safe_first_never_enables_gdr():
 name,policy=d.POLICIES[0];assert policy['NCCL_IB_DISABLE']=='1';assert policy['NCCL_NET_GDR_LEVEL']=='LOC'

def test_recipe_requires_identity_and_proof(monkeypatch,tmp_path):
 monkeypatch.setattr(d,'fingerprint',lambda:{'identity':1})
 proof=good();(tmp_path/'recovery-proof.json').write_text(json.dumps(proof))
 profile={'schema':1,'status':'VERIFIED','devices':[0,1],'identity':{'identity':1},'policy':{},
   'proof':{'path':'recovery-proof.json','sha256':hashlib.sha256((tmp_path/'recovery-proof.json').read_bytes()).hexdigest()}}
 path=tmp_path/'profile.json';path.write_text(json.dumps(profile));assert d.read_profile(path)==profile
 profile['policy']={'NCCL_IB_DISABLE':'1'};path.write_text(json.dumps(profile))
 with pytest.raises(RuntimeError,match='POLICY'):d.read_profile(path)

@pytest.mark.parametrize('devices',[[0,0],[-1],[],['0']])
def test_invalid_devices(devices):
 with pytest.raises(ValueError):d.run(devices=devices)


@pytest.mark.parametrize('unhealthy',[False,True])
def test_full_controller_recovery_or_blocked(monkeypatch,tmp_path,unhealthy):
 import multigpubfs.diagnostics as base
 monkeypatch.setattr(base,'prepare_delivery',lambda *a:None)
 calls=[]
 def fake(self,name,config,policy=None,seconds=60,gpu=False):
  calls.append(name);folder=self.runner.root/name;folder.mkdir()
  step=good();step.update(name=name,policy=policy or {},diagnostic_case=config)
  if name=='discovery':step['result']['data']={'visible_devices':2}
  elif name=='runtime-identity':step['result']['data']={'fake':'identity'}
  elif name=='transport-local-shm' or unhealthy and name=='transport-local-shm-fresh-cuda-health':step['status']='FAIL'
  elif self.blocked and gpu:step.update(status='SKIPPED',reason='CUDA_HEALTH_BLOCKED')
  (folder/'result.json').write_text(json.dumps(step['result']))
  self.runner.report['steps'].append(step);return step
 monkeypatch.setattr(d.Supervisor,'case',fake)
 report=d.run(tmp_path/'run',[0,1],1800)
 assert report['recovery']['status']==('BLOCKED' if unhealthy else 'READY')
 if unhealthy:
  assert 'transport-local-p2p' not in calls
  assert not (tmp_path/'run/recovery-profile.json').exists()
 else:
  assert 'transport-local-p2p' in calls
  assert report['recovery']['selected']=='local-p2p'
  assert 'public-launcher-check' in calls
 assert (tmp_path/'run/diagnostics.zip').exists()

def test_preflight_children_stay_in_supervised_group(monkeypatch,tmp_path):
 import multigpubfs.nccl_preflight as n
 monkeypatch.setattr(n,'_identity',lambda *a:'test')
 sessions=[]
 class Done:
  returncode=0
  def poll(self):return 0
  def wait(self,timeout=None):return 0
 def start(args,**kwargs):
  sessions.append(kwargs['start_new_session']);root=Path(args[-2]);rank=int(args[-1])
  (root/f'rank-{rank}.json').write_text(json.dumps({'verified':True,'rank':rank,'device':rank,'nccl_device_bytes':0}))
  return Done()
 monkeypatch.setattr(n.subprocess,'Popen',start)
 n.prepare('fake',[0,1],{'MGBFS_DIAGNOSTIC_SUPERVISED':'1'},tmp_path/'out')
 assert sessions==[False,False]

def test_unavailable_result_is_not_a_pass():
 row=good();row['result']['data']['status']='UNAVAILABLE';assert not d.verified(row)
