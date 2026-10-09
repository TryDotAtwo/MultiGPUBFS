"""Bounded startup capability observation, separate from graph hot paths."""
import json,subprocess,sys

def library_capabilities(env):
 code="import ctypes,json;lib=ctypes.CDLL('libmgbfs_cuda.so');f=getattr(lib,'mgbfs_nccl_lsa_compiled',None);print(json.dumps({'lsa_compiled':None if f is None else bool(f())}))"
 try:
  p=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=10)
 except (OSError,subprocess.TimeoutExpired) as error:
  return {'lsa_compiled':None,'status':'CAPABILITY_QUERY_UNAVAILABLE','detail':str(error)}
 if p.returncode:return {'lsa_compiled':None,'status':'CAPABILITY_QUERY_UNAVAILABLE','detail':p.stderr[-1000:]}
 try:v=json.loads(p.stdout)
 except ValueError:return {'lsa_compiled':None,'status':'CAPABILITY_QUERY_UNAVAILABLE','detail':'invalid native metadata'}
 if type(v.get('lsa_compiled')) not in (bool,type(None)):raise RuntimeError('INVALID_LIBRARY_CAPABILITY')
 return dict(v,status='OBSERVED_LIBRARY_METADATA',scope='compiled feature only, not device support or throughput')


def select_peer_transport(native,env,devices,root,*,request='auto',seconds=30):
 """One fixed exact GPU capability fixture. Unknown failures never fall back."""
 import time
 from pathlib import Path
 if request not in ('auto','host','lsa'):raise ValueError('INVALID_PEER_TRANSPORT')
 root=Path(root);root.mkdir(parents=True);started=time.monotonic()
 decision={'selected':'HOST_SIZED_NCCL','status':'HOST_REQUESTED','request':request,'scope':'startup capability only; throughput is measured separately'}
 if request!='host':
  capabilities=library_capabilities(env);decision['library']=capabilities
  if len(devices)==1:decision['status']='SINGLE_RANK_NO_PEER_TRANSPORT'
  elif capabilities['lsa_compiled'] is not True:decision['status']='LSA_NOT_COMPILED' if capabilities['lsa_compiled'] is False else 'LSA_CAPABILITY_UNKNOWN'
  elif seconds<5:decision['status']='INSUFFICIENT_CAPABILITY_BUDGET'
  else:
   from .graph_definition import GraphDefinition
   from .specialized import run_specialized
   fixture=GraphDefinition.permutation([[1,2,3,0],[3,0,1,2],[1,0,2,3]],[0,1,2,3]);probe=root/'lsa-probe'
   try:
    report=run_specialized(fixture,probe,native=native,env=dict(env,MGBFS_SPECIALIZED_TRANSPORT='NCCL_LSA',NCCL_CUMEM_ENABLE='1'),devices=devices,capacity=256,batch=2,max_seconds=min(30,seconds))
    states=json.loads((probe/'states.json').read_text())
    expected_previous=[[0,1,3,2],[0,3,2,1],[2,1,0,3]]
    if report['status']!='COMPLETE' or report['layer_sizes']!=[1,3,5,6,5,3,1] or sorted(states['current'])!=[[1,0,3,2]] or sorted(states['previous_small'])!=expected_previous:raise RuntimeError('LSA_CAPABILITY_ORACLE_MISMATCH')
    decision.update(selected='NCCL_LSA',status='VERIFIED_FIXED_EXACT_GPU_CAPABILITY')
   except RuntimeError as failure:
    logs=[(probe/f'rank-{rank}.log').read_text() for rank in range(len(devices))]
    # Native status4 is exactly the unsupported deviceAPI/LSA-team condition.
    errors=[]
    for text in logs:
     rank_errors=[]
     for line in text.splitlines():
      try:value=json.loads(line)
      except ValueError:continue
      if isinstance(value,dict) and value.get('status')=='ERROR':rank_errors.append(value.get('error'))
     errors.append(rank_errors)
    if not str(failure).startswith('SPECIALIZED_COMMITTED_SNAPSHOT_MISSING') or errors!=[['LSA_PREPARE_GROUP: CUDA_STATUS_4'] for _ in devices]:raise
    decision.update(status='LSA_DEVICE_TOPOLOGY_UNSUPPORTED',native_error='LSA_PREPARE_GROUP: CUDA_STATUS_4')
 if request=='lsa' and decision['selected']!='NCCL_LSA':
  (root/'decision.json').write_text(json.dumps(decision,indent=2));raise RuntimeError('REQUESTED_LSA_NOT_VERIFIED: '+decision['status'])
 decision['seconds']=time.monotonic()-started;(root/'decision.json').write_text(json.dumps(decision,indent=2));return decision
