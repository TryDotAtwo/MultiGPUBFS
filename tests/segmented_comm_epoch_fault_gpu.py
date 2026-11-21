"""Real multi-rank epoch/collective failure gates; no production retries.

Each injected fault is scoped to one fresh persistent group and rank. The
next calculation is explicitly opened in a new pool after verifying that the
failed group rejects reuse. Layers are compared with a CPU BFS multiset.
"""
from collections import Counter
from itertools import permutations
from pathlib import Path
import json,os,tempfile,time
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

native=os.environ['MGBFS_EXECUTABLE'];world=int(os.environ.get('MGBFS_TEST_WORLD','2'))
devices=list(range(world));root=Path(tempfile.mkdtemp(prefix=f'segmented-epoch-{world}gpu-',dir='/workspace'))
def graph(n):
 gs=[]
 for i in range(n-1):
  p=list(range(n));p[i],p[i+1]=p[i+1],p[i];gs.append(p)
 return GraphDefinition.permutation(gs,list(range(n)))
def oracle(n):
 cur={tuple(range(n))};seen=set(cur);layers=[]
 while cur:
  layers.append(Counter(cur));nxt=set()
  for x in cur:
   for i in range(n-1):
    y=list(x);y[i],y[i+1]=y[i+1],y[i];y=tuple(y)
    if y not in seen:nxt.add(y)
  seen.update(nxt);cur=nxt
 return layers
def launch(pool,label,n=4,fail=None,strategy='atomic'):
 env=dict(os.environ,MGBFS_SEGMENT_TEST_BATCH='1',MGBFS_SEGMENT_EXACT_EXCHANGE='1')
 env.pop('MGBFS_SEGMENT_TEST_HOST_FAIL',None);env.pop('MGBFS_SEGMENT_TEST_FAIL_RANK',None)
 if fail:env.update(MGBFS_SEGMENT_TEST_HOST_FAIL=fail,MGBFS_SEGMENT_TEST_FAIL_RANK=str(world-1))
 return run_graph(graph(n),root/label,devices=devices,shards=2,segment_rows=3,_batch=8,
   autotune=False,memory_backend='segmented24',segment_strategy=strategy,executable=native,
   _profile_layers=999,_native_env=env)
def check(r,n):
 expected=oracle(n)
 assert r['status']=='COMPLETE',(r['status'],r['reason'])
 assert r['layer_sizes']==[sum(x.values()) for x in expected]
 assert Counter(map(tuple,json.loads((Path(r['_test_output'])/'states.json').read_text())['current']))==expected[-1]
def execute(label,n=4,fail=None,strategy='atomic',reuse=True):
 with persistent_graph_workers() as pool:
  if reuse:
   initial=launch(pool,label+'-before',n=n);initial['_test_output']=str(root/(label+'-before'));check(initial,n)
   pids=[w.process.pid for w in pool.workers]
  t=time.monotonic()
  try:launch(pool,label+'-fault',n=n,fail=fail,strategy=strategy)
  except Exception as e:
   message=str(e);print('FAULT',world,label,'seconds',round(time.monotonic()-t,3),message,flush=True)
   assert 'NATIVE_DISTRIBUTED' in message or 'SEGMENT' in message,(label,message)
  else:raise AssertionError((label,'fault was not observed'))
  assert not (root/(label+'-fault')/'report.json').exists(),label
  try:launch(pool,label+'-forbidden-reuse',n=n)
  except Exception as e:
   assert any(x in str(e) for x in ('FAILED_GROUP_NO_RESTART','FAILED_NO_RESTART')),(label,str(e))
  else:raise AssertionError((label,'failed session unexpectedly reused'))
  print('PASS FAILED GROUP CLOSED',world,label,flush=True)
 with persistent_graph_workers() as pool:
  result=launch(pool,label+'-clean',n=n);result['_test_output']=str(root/(label+'-clean'));check(result,n)
  if reuse:assert [w.process.pid for w in pool.workers]!=pids
  print('PASS CLEAN NEW GROUP',world,label,flush=True)
 if fail not in ('drop-comm',):
  base=root/(label+'-fault')
  combined='\n'.join(p.read_text(errors='replace') for p in base.rglob('*') if p.is_file() and p.suffix in ('.log','.json') and p.stat().st_size<2_000_000)
  needles={'finish-write':'SEGMENTED_TEST_FINISH_WRITE','finish-publish':'SEGMENTED_PUBLISH_STATES',
    'after-seed':'SEGMENTED_TEST_HOST_FAIL_AFTER-SEED','count-payload':'SEGMENTED_TEST_HOST_FAIL_COUNT-PAYLOAD',
    'strategy-copy':'SEGMENTED_TEST_HOST_FAIL_STRATEGY-COPY','warmup-clear':'SEGMENTED_TEST_HOST_FAIL_WARMUP-CLEAR',
    'batch-copy':'SEGMENTED_TEST_HOST_FAIL_BATCH-COPY'}
  assert needles[fail] in combined,(label,needles[fail],combined[-5000:])
 else:
  combined='\n'.join(p.read_text(errors='replace') for p in (root/(label+'-fault')).rglob('*') if p.is_file() and p.suffix in ('.log','.json') and p.stat().st_size<2_000_000)
  assert 'SEGMENTED_TEST_DROPPED_COMM' in combined or 'SEGMENTED_HOST_MISMATCH' in combined,(label,combined[-5000:])
 print('PASS EVIDENCE',world,label,flush=True)

with persistent_graph_workers() as pool:
 pids=None
 for n in (4,5,4):
  label=f'warm-{n}-{time.monotonic_ns()}'
  r=launch(pool,label,n=n);r['_test_output']=str(root/label);check(r,n)
  now=[w.process.pid for w in pool.workers]
  if pids is None:pids=now
  assert now==pids,(now,pids)
 print('PASS WARM DIFFERENT GRAPHS',world,'pids',pids,flush=True)
for name in ('finish-write','finish-publish','drop-comm','after-seed','count-payload'):
 execute(name,fail=name)
if world==2:
 for name in ('strategy-copy','warmup-clear'):
  execute(name,fail=name,reuse=False)
 execute('batch-copy',fail='batch-copy',strategy='batch',reuse=False)
print('SEGMENTED_COMM_EPOCH_PHYSICAL_PASS',world,root,flush=True)
