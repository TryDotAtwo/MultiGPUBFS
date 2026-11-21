"""A post-vote local receipt failure invalidates a live communicator epoch."""
import importlib,os,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
m=importlib.import_module('multigpubfs.launch');native=os.environ['MGBFS_EXECUTABLE'];world=int(os.environ.get('MGBFS_TEST_WORLD','2'));devices=list(range(world));root=Path(tempfile.mkdtemp(prefix=f'segmented-receipt-{world}gpu-',dir='/workspace'))
g=GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],[0,1,2,3])
def run(name):return m.run_graph(g,root/name,devices=devices,shards=2,autotune=False,executable=native,memory_backend='segmented24',segment_strategy='warp',segment_rows=3,_batch=8,_profile_layers=999)
with persistent_graph_workers() as pool:
 a=run('before');assert a['status']=='COMPLETE';workers=list(pool.workers);pids=[w.process.pid for w in workers]
 original=m._receipt
 def fail_one(path,digest):
  if Path(path).name=='rank-0':raise OSError('TEST_AGGREGATE_RECEIPT_FAILURE')
  return original(path,digest)
 m._receipt=fail_one
 try:
  try:run('fault')
  except OSError as e:assert 'TEST_AGGREGATE_RECEIPT_FAILURE' in str(e)
  else:raise AssertionError('receipt failure missing')
 finally:m._receipt=original
 assert pool.poisoned and all(w.process.poll() is not None for w in workers)
 try:run('reuse')
 except RuntimeError as e:assert 'FAILED_GROUP_NO_RESTART' in str(e),str(e)
 else:raise AssertionError('poisoned epoch reused')
 print('PASS RECEIPT FAILURE POISONS GROUP',world,pids,flush=True)
with persistent_graph_workers() as pool:
 a=run('new-group');assert a['status']=='COMPLETE' and a['layer_sizes']==[1,3,5,6,5,3,1]
 print('PASS RECEIPT CLEAN NEW GROUP',world,[w.process.pid for w in pool.workers],flush=True)
print('RECEIPT_POISON_GPU_PASS',root,flush=True)
