"""Hard-cap selector receipts on two GPUs across a resident backend switch."""
import json,math,os,subprocess,tempfile,time
from pathlib import Path
os.environ.setdefault('MGBFS_NATIVE_OFFLINE','1')
os.environ.update(MGBFS_SEGMENT_ROWS='65536', MGBFS_SEGMENT_MAX_RANK_STATES='50000')
from cayleypy import CayleyGraph,PermutationGroups
import multigpubfs
from multigpubfs import NativeOptions,persistent_graph_workers
expected_package=os.environ.get('MGBFS_TEST_EXPECT_PACKAGE_PREFIX')
if expected_package:assert str(multigpubfs.__file__).startswith(expected_package),multigpubfs.__file__
print('PACKAGE',multigpubfs.__file__,flush=True)
root=Path(os.environ.get('MGBFS_TEST_OUTPUT') or (Path(tempfile.mkdtemp(prefix='mgbfs-test-'))/'capacity'))
root.mkdir(parents=True,exist_ok=False)
os.environ['MGBFS_PROFILE_CACHE']=str(root/'cache')
before=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True)
(root/'gpu-before.txt').write_text(before)
assert not before.strip(),before
n=9;oracle=[1]
for j in range(1,n+1):
    z=[0]*(len(oracle)+j-1)
    for i,v in enumerate(oracle):
        for k in range(j):z[i+k]+=v
    oracle=z
graph=CayleyGraph(PermutationGroups.coxeter(n),device='cpu')
records=[];pids=None
multigpubfs.enable_native(recovery=False)
try:
 with persistent_graph_workers() as pool:
  for i,backend in enumerate(('segmented24','permutation_bitmap','segmented24')):
   print('START',i,backend,flush=True)
   opts=NativeOptions(devices=(0,1),autotune=True,memory_backend=backend,memory_preset='max_capacity',segment_rows=65536)
   start=time.monotonic()
   result=graph.bfs(native_options=opts,output=root/f'job-{i}')
   report=result.report;profile=report['autotune']
   assert result.bfs_completed and report['layer_sizes']==oracle and sum(oracle)==math.factorial(n)
   assert profile['status']=='MEASURED_EQUAL_PREFIX_GPU_PROFILE' and profile['measured']
   for pilot in profile['pilots']:
    assert pilot['capacity_limit_per_rank_by_rank']==[50000,50000],pilot
    assert pilot['capacity_limit_scope']=='one next layer per rank',pilot
    assert len(pilot['capacity_allocated_rows_per_bank_by_rank'])==2
   priorities=[min(min(v,50000) for v in p['capacity_score_rows_by_rank']) for p in profile['pilots']]
   assert all(v==50000 for v in priorities),priorities
   winner=profile['winner_index'];scores=profile['scores_seconds']
   assert scores[winner]==min(scores),(winner,scores,priorities)
   ids=[w.process.pid for w in pool.workers]
   if pids is None:pids=ids
   assert ids==pids
   rec={'job':i,'backend':backend,'cache_hit':profile.get('cache_hit'),'winner':winner,
        'scores_seconds':scores,'bounded_planned_rows':priorities,
        'capacity_inventory':profile.get('identity',{}).get('capacity_inventory'),
        'actual_queue_by_rank':profile['pilots'][winner].get('segment_queue_capacity_by_rank'),
        'pids':ids,'bfs_seconds':report['bfs_seconds'],'wall_seconds':time.monotonic()-start}
   records.append(rec);print('PASS',json.dumps(rec),flush=True)
finally:multigpubfs.disable_native()
(root/'evidence.json').write_text(json.dumps(records,indent=2))
print('CAPACITY_SELECTOR_TWO_GPU_PASS',root,flush=True)
