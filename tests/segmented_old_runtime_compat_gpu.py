"""The new Python launcher must not falsely enable exact exchange on dev38."""
import json,os,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

world=int(os.environ.get('MGBFS_TEST_DEVICE_COUNT','8'));shards=int(os.environ.get('MGBFS_TEST_SHARDS','8'))
n=10; gens=[]
for i in range(n-1):
 g=list(range(n));g[i],g[i+1]=g[i+1],g[i];gens.append(g)
graph=GraphDefinition.permutation(gens,list(range(n)));oracle=[1]
for degree in range(1,n+1):
 nxt=[0]*(len(oracle)+degree-1)
 for j,x in enumerate(oracle):
  for k in range(degree):nxt[j+k]+=x
 oracle=nxt
root=Path(tempfile.mkdtemp(prefix='old-runtime-compat-',dir='/workspace'))
os.environ['MGBFS_PROFILE_CACHE']=str(root/'cache');os.environ['MGBFS_SEGMENT_ROWS']='65536';os.environ.pop('MGBFS_SEGMENT_EXACT_EXCHANGE',None)
with persistent_graph_workers():
 for backend in ('segmented24','permutation_bitmap'):
  result=run_graph(graph,root/backend,devices=list(range(world)),shards=shards,autotune=True,memory_backend=backend,executable=os.environ['MGBFS_EXECUTABLE'])
  profile=result['autotune']
  assert result['status']=='COMPLETE' and result['layer_sizes']==oracle,result
  assert profile['status']=='MEASURED_EQUAL_PREFIX_GPU_PROFILE' and profile['segment_exact_exchange'] is False,profile
  assert all(p['segment_exact_exchange'] is False for p in profile['pilots']),profile
  print(json.dumps({'backend':backend,'status':result['status'],'chosen_exact':profile['segment_exact_exchange'],'pilots':len(profile['pilots'])}),flush=True)
 os.environ['MGBFS_SEGMENT_EXACT_EXCHANGE']='1'
 try:run_graph(graph,root/'unsupported',devices=list(range(world)),shards=shards,autotune=False,memory_backend='permutation_bitmap',executable=os.environ['MGBFS_EXECUTABLE'])
 except RuntimeError as e:assert 'NATIVE_SEGMENTED_EXCHANGE_UNAVAILABLE' in str(e),e
 else:raise AssertionError('old runtime silently accepted unsupported exact exchange')
print('OLD_RUNTIME_COMPAT_PASS',root,flush=True)
