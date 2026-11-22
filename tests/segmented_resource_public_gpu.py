"""Two-GPU public API: measured probes hit cap, then a healthy resident job."""
import json,os,subprocess,tempfile,time,traceback
from collections import Counter
from pathlib import Path
world=int(os.environ.get('MGBFS_TEST_DEVICE_COUNT','2'))
assert world in (2,8)
devices=tuple(range(world))
cap=32 if world==2 else 8
os.environ.setdefault('MGBFS_NATIVE_OFFLINE','1')
os.environ.update(MGBFS_SEGMENT_ROWS='16', MGBFS_SEGMENT_MAX_RANK_STATES=str(cap))
from cayleypy import CayleyGraph,PermutationGroups
import multigpubfs
expected_package=os.environ.get('MGBFS_TEST_EXPECT_PACKAGE_PREFIX')
if expected_package:assert str(multigpubfs.__file__).startswith(expected_package),multigpubfs.__file__
print('PACKAGE',multigpubfs.__file__,flush=True)
from multigpubfs import NativeOptions,persistent_graph_workers

root=Path(os.environ.get('MGBFS_TEST_OUTPUT') or (Path(tempfile.mkdtemp(prefix='mgbfs-test-'))/'resource'))
root.mkdir(parents=True,exist_ok=False)
os.environ['MGBFS_PROFILE_CACHE']=str(root/'profiles')
before=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True)
(root/'gpu-before.txt').write_text(before)
assert not before.strip(),before
n=9
graph=CayleyGraph(PermutationGroups.coxeter(n),device='cpu')
start=tuple(range(n));front={start};seen=set(front);expected=[front]
for depth in range(3):
    nxt=set()
    for state in front:
        for i in range(n-1):
            c=list(state);c[i],c[i+1]=c[i+1],c[i];nxt.add(tuple(c))
    front=nxt-seen
    seen.update(front);expected.append(front)
records=[]
multigpubfs.enable_native(recovery=False)
try:
 with persistent_graph_workers() as pool:
  for backend in ('segmented24','permutation_bitmap'):
   for preset in ('balanced','max_capacity'):
    print('START',backend,preset,flush=True)
    opts=NativeOptions(devices=devices,autotune=True,memory_backend=backend,
                       memory_preset=preset,segment_rows=16,segment_strategy='atomic')
    result=graph.bfs(native_options=opts,output=root/(backend+'-'+preset))
    report=result.report
    profile=report['autotune']
    print('PROFILE',backend,preset,profile['status'],len(profile.get('resource_rejections',[])),flush=True)
    assert report['status']=='INCOMPLETE' and report['reason']=='RESOURCE_10',(backend,preset,report)
    assert profile['status']=='VALIDATED_RESOURCE_LIMIT_UNMEASURED_PROFILE' and not profile['measured'],profile
    assert len(profile['resource_rejections'])>=2,profile
    assert all(r['validated_ranks']==world for r in profile['resource_rejections'])
    depth=len(report['layer_sizes'])-1
    assert depth<=2 and report['layer_sizes']==[len(v) for v in expected[:depth+1]]
    snapshot=json.loads((result.output/'states.json').read_text())
    for d,key in ((depth,'current'),(depth-1,'previous_small')):
        if d<0:continue
        assert Counter(map(tuple,snapshot[key]))==Counter(expected[d]),(backend,preset,d)
    if depth>=1:assert expected[0]=={start}
    ids=[w.process.pid for w in pool.workers]
    records.append({'backend':backend,'preset':preset,'status':report['status'],'reason':report['reason'],
                    'depth':depth,'layer_sizes':report['layer_sizes'],'profile_status':profile['status'],
                    'pilots':len(profile['resource_rejections']),'pids':ids,'exact_completed_layers':depth+1})
    print('PASS',json.dumps(records[-1]),flush=True)
  os.environ.pop('MGBFS_SEGMENT_MAX_RANK_STATES')
  small=CayleyGraph(PermutationGroups.coxeter(5),device='cpu')
  final=small.bfs(native_options=NativeOptions(devices=devices,autotune=False,
                  memory_backend='permutation_bitmap',memory_preset='balanced',segment_rows=16),
                  output=root/'healthy-after')
  assert final.bfs_completed and sum(final.layer_sizes)==120
  assert [w.process.pid for w in pool.workers]==ids
  print('PASS healthy-after',flush=True)
finally:
 multigpubfs.disable_native()
(root/'evidence.json').write_text(json.dumps(records,indent=2))
print('PUBLIC_RESOURCE_GPU_PASS',root,flush=True)
