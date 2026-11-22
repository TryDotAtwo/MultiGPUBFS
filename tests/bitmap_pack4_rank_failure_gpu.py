"""Packed bitmap: a failed rank poisons the pool, then a fresh pool succeeds."""
import json,os,tempfile,time
from collections import Counter
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.launch import run_graph
from multigpubfs.generic_session import persistent_graph_workers

world=int(os.environ.get('MGBFS_TEST_DEVICE_COUNT','2'))
devices=list(range(world))
root=Path(tempfile.mkdtemp(prefix='pack4-rank-failure-',dir='/workspace'))
native=os.environ['MGBFS_EXECUTABLE']
env=dict(os.environ,MGBFS_SEGMENT_PACK4='1',MGBFS_SEGMENT_ROWS='3',
         MGBFS_SEGMENT_TEST_BATCH='1',MGBFS_SEGMENT_EXACT_EXCHANGE='1')
gens=[]
for i in range(4):
    x=list(range(5));x[i],x[i+1]=x[i+1],x[i];gens.append(x)
g=GraphDefinition.permutation(gens,list(range(5)))
front={tuple(range(5))};seen=set(front);expected=[]
while front:
    expected.append(front)
    nxt={tuple(row[i] for i in gen) for row in front for gen in gens}-seen
    seen.update(nxt);front=nxt
kw=dict(devices=devices,autotune=False,shards=4,_batch=2,executable=native,
        memory_backend='permutation_bitmap',memory_preset='max_capacity',
        segment_rows=3,segment_strategy='atomic')
old=[]
try:
    with persistent_graph_workers() as pool:
        warm=run_graph(g,root/'warm',_native_env=env,**kw)
        assert warm['status']=='COMPLETE'
        old=[w.process.pid for w in pool.workers]
        bad=dict(env,MGBFS_SEGMENT_TEST_FATAL_RANK='1',MGBFS_SEGMENT_TEST_FATAL_AFTER_DEPTH='2')
        start=time.monotonic()
        try:run_graph(g,root/'failed',_native_env=bad,**kw)
        finally:print('FAULT_WAIT_SECONDS',time.monotonic()-start,flush=True)
except RuntimeError as e:
    assert 'NATIVE_DISTRIBUTED_RANK_FAILED' in str(e),repr(e)
    print('EXPECTED_FAILURE',str(e),flush=True)
else:raise AssertionError('one rank must fail')
assert (root/'failed'/'failure-evidence').is_dir()
assert 'SEGMENTED_TEST_ONE_RANK_FAILURE' in (root/'failed'/'rank-1.log').read_text()
for pid in old:assert not Path('/proc',str(pid)).exists(),('worker still alive',pid)
with persistent_graph_workers() as pool:
    report=run_graph(g,root/'fresh',_native_env=env,**kw)
    assert report['status']=='COMPLETE' and report['layer_sizes']==list(map(len,expected)),report
    snap=json.loads((root/'fresh'/'states.json').read_text())
    assert Counter(map(tuple,snap['current']))==Counter(expected[-1])
    assert Counter(map(tuple,snap['previous_small']))==Counter(expected[-2])
    assert all(r['resident_codec']=='pack4-v1' for r in report['segment_memory_by_rank'])
    print('PASS fresh workers',[w.process.pid for w in pool.workers],flush=True)
print('PACK4_RANK_FAILURE_PASS',world,root,flush=True)
