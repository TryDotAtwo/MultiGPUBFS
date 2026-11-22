"""Public two-GPU measured tuning with failed and healthy advisory cache."""
import json,math,os,subprocess,tempfile,time
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('MGBFS_NATIVE_OFFLINE','1')
os.environ.update(MGBFS_SEGMENT_ROWS='65536')
os.environ.pop('MGBFS_SEGMENT_MAX_RANK_STATES',None)
from cayleypy import CayleyGraph,PermutationGroups
import multigpubfs
from multigpubfs import NativeOptions,persistent_graph_workers
from multigpubfs import cayleypy_backend
expected_package=os.environ.get('MGBFS_TEST_EXPECT_PACKAGE_PREFIX')
if expected_package:assert str(multigpubfs.__file__).startswith(expected_package),multigpubfs.__file__
print('PACKAGE',multigpubfs.__file__,flush=True)
root=Path(os.environ.get('MGBFS_TEST_OUTPUT') or (Path(tempfile.mkdtemp(prefix='mgbfs-test-'))/'cache'))
root.mkdir(parents=True,exist_ok=False)
before=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True)
(root/'gpu-before.txt').write_text(before)
assert not before.strip(),before
n=9
oracle=[1]
for j in range(1,n+1):
    nxt=[0]*(len(oracle)+j-1)
    for i,v in enumerate(oracle):
        for k in range(j):nxt[i+k]+=v
    oracle=nxt
graph=CayleyGraph(PermutationGroups.coxeter(n),device='cpu')
options=NativeOptions(devices=(0,1),autotune=True,memory_backend='segmented24',
                      memory_preset='balanced',segment_rows=65536,segment_strategy='atomic')
records=[]
multigpubfs.enable_native(recovery=False)
try:
 with persistent_graph_workers() as pool:
  for name,cache in [('broken','/proc/mgbfs-unavailable-profiles'),('good-write',str(root/'cache')),('good-read',str(root/'cache'))]:
   print('START',name,flush=True)
   os.environ['MGBFS_PROFILE_CACHE']=cache
   with patch.object(cayleypy_backend,'run_graph',wraps=cayleypy_backend.run_graph) as production:
    result=graph.bfs(native_options=options,output=root/name)
   report=result.report;profile=report['autotune']
   assert production.call_count==1,production.call_count
   assert result.bfs_completed and report['layer_sizes']==oracle and sum(oracle)==math.factorial(n)
   assert profile['status']=='MEASURED_EQUAL_PREFIX_GPU_PROFILE' and profile['measured'],profile
   if name=='broken':
    assert profile['profile_cache_write_status']['status']=='UNAVAILABLE',profile
    assert profile['profile_cache_write_status']['stage']=='mkdir',profile
   if name=='good-write':assert profile['profile_cache_write_status']['status']=='WRITTEN'
   if name=='good-read':assert profile.get('cache_hit') is True,profile
   if name!='broken':assert not list((root/'cache').glob('*.tmp'))
   ids=[w.process.pid for w in pool.workers]
   snap=json.loads((result.output/'states.json').read_text())
   rev=tuple(range(n-1,-1,-1))
   prev=[]
   for i in range(n-1):
    x=list(rev);x[i],x[i+1]=x[i+1],x[i];prev.append(tuple(x))
   assert sorted(map(tuple,snap['current']))==[rev]
   assert sorted(map(tuple,snap['previous_small']))==sorted(prev)
   record={'name':name,'production_calls':production.call_count,'profile_status':profile['status'],
           'cache_hit':profile['cache_hit'],'read_status':profile.get('profile_cache_read_status'),
           'write_status':profile.get('profile_cache_write_status'),'pids':ids,'layers':len(oracle),
           'states':sum(oracle),'bfs_seconds':report['bfs_seconds']}
   records.append(record);print('PASS',json.dumps(record),flush=True)
  assert len({tuple(r['pids']) for r in records})==1
finally:multigpubfs.disable_native()
(root/'evidence.json').write_text(json.dumps(records,indent=2))
print('PUBLIC_CACHE_TWO_GPU_PASS',root,flush=True)
