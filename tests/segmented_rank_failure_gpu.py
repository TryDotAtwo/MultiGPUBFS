"""A rank fails mid-job; launcher must stop peers, retain evidence and run fresh."""
import os,tempfile,time
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.launch import run_graph
from multigpubfs.generic_session import persistent_graph_workers
root=Path(tempfile.mkdtemp(prefix='seg24-failure-',dir='/workspace'));native=os.environ['MGBFS_EXECUTABLE']
env=dict(os.environ,MGBFS_EXPERIMENTAL_SEGMENTED24='1',MGBFS_SEGMENT_ROWS='3',MGBFS_GENERIC_GENERATOR='cuda',MGBFS_GENERIC_HISTORY='hash',MGBFS_GENERIC_TRANSPORT='full',MGBFS_GENERIC_SORT='none')
g=GraphDefinition.permutation([[1,0,2,3,4,5],[0,2,1,3,4,5],[0,1,3,2,4,5],[0,1,2,4,3,5],[0,1,2,3,5,4]],list(range(6)))
bad=dict(env,MGBFS_SEGMENT_TEST_FATAL_RANK='1',MGBFS_SEGMENT_TEST_FATAL_AFTER_DEPTH='2')
t0=time.monotonic();pids=[]
try:
 with persistent_graph_workers() as pool:
  try:run_graph(g,root/'failed',devices=[0,1],autotune=False,shards=4,executable=native,memory_backend='segmented24',_profile_layers=30,_native_env=bad)
  finally:pids=[w.process.pid for w in pool.workers]
except RuntimeError as e:
 assert 'NATIVE_DISTRIBUTED_RANK_FAILED' in str(e),repr(e)
else:raise AssertionError('one rank must fail')
assert time.monotonic()-t0<60,'failure cleanup took too long'
assert (root/'failed'/'failure-evidence').is_dir()
assert 'SEGMENTED_TEST_ONE_RANK_FAILURE' in (root/'failed'/'rank-1.log').read_text()
for pid in pids:assert not Path('/proc',str(pid)).exists(),('worker still alive',pid)
print('PASS rank failure and supervisor cleanup in %.2f sec'%(time.monotonic()-t0),flush=True)
with persistent_graph_workers() as pool:
 report=run_graph(g,root/'fresh',devices=[0,1],autotune=False,shards=4,executable=native,memory_backend='segmented24',_profile_layers=30,_native_env=env)
 assert report['status']=='COMPLETE' and report['layer_sizes']==[len(x) for x in g.exact_layers(1000)],report
 print('PASS fresh persistent job after fault',flush=True)
print('EVIDENCE',root)
