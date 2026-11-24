"""Pack4 ABI/strategy guards and per-rank configuration agreement."""
import os,subprocess,tempfile,time
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.launch import run_graph
root=Path(tempfile.mkdtemp(prefix='pack4-guard-',dir='/workspace'))
native=os.environ['MGBFS_EXECUTABLE']
graph=GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],[0,1,2,3])
definition=root/'graph.json';definition.write_text(graph.to_json())
env=dict(os.environ,MGBFS_SEGMENT_PACK4='1',MGBFS_SEGMENT_TEST_BATCH='1')
for backend,strategy,library,expected in [
 ('permutation_bitmap','atomic',os.environ['MGBFS_TEST_OLD_LIBRARY'],'BITMAP_PACK4_NATIVE_ABI_MISSING'),
 ('permutation_bitmap','warp',None,'BITMAP_PACK4_STRATEGY_UNSUPPORTED'),
 ('permutation_bitmap','batch',None,'BITMAP_PACK4_STRATEGY_UNSUPPORTED'),
 ('segmented24','atomic',None,'BITMAP_PACK4_REQUIRES_BITMAP'),
 ('legacy','atomic',None,'BITMAP_PACK4_REQUIRES_BITMAP')]:
 local=dict(env)
 if library:local['LD_LIBRARY_PATH']=library
 try:
  run_graph(graph,root/(backend+'-'+strategy+('-old' if library else '')),devices=[0,1],autotune=False,
    shards=4,memory_backend=backend,segment_strategy=strategy,executable=native,_native_env=local)
  raise AssertionError('incompatible configuration was accepted')
 except (ValueError,RuntimeError) as e:
  assert expected in str(e),(backend,strategy,expected,str(e))
 print('PASS guard',backend,strategy,expected,flush=True)
base_env=dict(env,MGBFS_SEGMENT_PACK4='0')
base=run_graph(graph,root/'seed',devices=[0,1],autotune=False,shards=4,_batch=2,
  memory_backend='permutation_bitmap',segment_strategy='atomic',executable=native,_profile_layers=1,_native_env=base_env)
assert base['layer_sizes']==[1,3]
config=root/'seed'/'launch.json';folder=root/'rank-mismatch';folder.mkdir();jobs=[];logs=[]
try:
 for rank,packed in enumerate(('1','0')):
  runenv=dict(os.environ,MGBFS_EXPERIMENTAL_SEGMENTED24='1',MGBFS_EXPERIMENTAL_BITMAP='1',
      MGBFS_SEGMENT_PACK4=packed,MGBFS_SEGMENT_TEST_BATCH='1',MGBFS_SEGMENT_TEST_HOST_TIMEOUT_MS='5000',
      MGBFS_SEGMENT_STRATEGY='atomic',RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='2')
  log=(folder/f'rank-{rank}.log').open('w');logs.append(log)
  jobs.append(subprocess.Popen([native,'graph-rank',str(definition),str(config),str(rank),
    str(folder/'id'),str(folder/f'rank-{rank}'),'0'],env=runenv,stdout=log,stderr=subprocess.STDOUT))
 deadline=time.monotonic()+30
 while any(p.poll() is None for p in jobs) and time.monotonic()<deadline:time.sleep(.05)
 assert all(p.poll() is not None for p in jobs),[p.poll() for p in jobs]
 for log in logs:log.flush()
 texts=[(folder/f'rank-{i}.log').read_text() for i in range(2)]
 assert all(p.returncode!=0 for p in jobs),(texts,[p.returncode for p in jobs])
 assert all('SEGMENTED_HOST_MISMATCH' in t for t in texts),texts
 assert all(not (folder/f'rank-{i}'/'report.json').exists() for i in range(2))
 print('PASS mismatched rank codec fails collectively before BFS',flush=True)
finally:
 for proc in jobs:
  if proc.poll() is None:proc.terminate()
 for proc in jobs:
  if proc.poll() is None:
   try:proc.wait(timeout=5)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
 for log in logs:log.close()
print('PACK4_GUARD_TWO_GPU_PASS',root,flush=True)
