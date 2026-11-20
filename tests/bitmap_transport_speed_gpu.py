"""Matched-input S9 two-rank diagnostic benchmark; no target-hardware claims."""
import json,os,subprocess,tempfile,time,statistics
from pathlib import Path
from multigpubfs import GraphDefinition
root=Path(tempfile.mkdtemp(prefix='seg24-speed-',dir='/workspace'));native=os.environ['MGBFS_EXECUTABLE'];base=dict(os.environ,MGBFS_GENERIC_GENERATOR='cuda',MGBFS_GENERIC_HISTORY='hash',MGBFS_GENERIC_TRANSPORT='full',MGBFS_GENERIC_SORT='none')
import math
n=int(os.environ.get('SEG24_BENCH_N','9'));capacity=int(os.environ.get('SEG24_BENCH_CAPACITY','200000'));rows=os.environ.get('SEG24_BENCH_ROWS','65536');shards=int(os.environ.get('SEG24_BENCH_SHARDS','4'));batch=int(os.environ.get('SEG24_BENCH_BATCH','8192'))
gens=[]
for i in range(n-1):
 g=list(range(n));g[i],g[i+1]=g[i+1],g[i];gens.append(g)
g=GraphDefinition.permutation(gens,list(range(n)));definition=root/'graph.json';definition.write_text(g.to_json());records=[];reference=None
# Independent Coxeter distance distribution: Mahonian inversion polynomial.
oracle=[1]
for degree in range(1,n+1):
 nxt=[0]*(len(oracle)+degree-1)
 for i,value in enumerate(oracle):
  for j in range(degree):nxt[i+j]+=value
 oracle=nxt
runs=[('legacy',0),('segmented',1),('segmented',4),('segmented',4),('segmented',1),('legacy',0)]
if os.environ.get('SEG24_BENCH_COMPARE_BITMAP')=='1':runs=[('legacy',0),('segmented',4),('bitmap',4),('bitmap',4),('segmented',4),('legacy',0)]
if os.environ.get('SEG24_BENCH_ONCE')=='1':runs=[('legacy',0),('segmented',4),('bitmap',4)]
only=os.environ.get('SEG24_BENCH_ONLY')
if only:runs=[(only,0 if only=='legacy' else int(os.environ.get('SEG24_BENCH_LANES','4')))]
if os.environ.get('SEG24_BENCH_TRACE')=='1':runs=[('segmented',4)]
for mode,lanes in runs:
 env=base.copy()
 if mode in ('segmented','bitmap'):env.update(MGBFS_EXPERIMENTAL_SEGMENTED24='1',MGBFS_SEGMENT_ROWS=rows,MGBFS_SEGMENT_TEST_LANES=str(lanes))
 if mode=='bitmap':env['MGBFS_EXPERIMENTAL_BITMAP']='1'
 case=root/('case-%d-%s-%d'%(len(records),mode,lanes));case.mkdir()
 out=subprocess.run([native,'graph-info',str(definition),'0,1',str(shards),str(capacity),str(batch)],env=env,text=True,capture_output=True,timeout=60,check=True)
 plan=json.loads(out.stdout)
 if mode=='bitmap':plan['owner_cuts']=None
 config=case/'launch.json';config.write_text(json.dumps(plan));jobs=[];files=[];t=time.monotonic()
 try:
  for rank in range(2):
   f=(case/f'rank-{rank}.log').open('w');files.append(f);jobs.append(subprocess.Popen([native,'graph-rank',str(definition),str(config),str(rank),str(case/'id'),str(case/f'rank-{rank}'),'0'],env=env,stdout=f,stderr=subprocess.STDOUT))
  while any(p.poll() is None for p in jobs):
   if any(p.poll() not in (None,0) for p in jobs):raise RuntimeError(('rank failed',[(case/f'rank-{i}.log').read_text()[-3000:] for i in range(2)]))
   if time.monotonic()-t>float(os.environ.get('SEG24_BENCH_RUN_SECONDS','240')):raise RuntimeError('timeout')
   time.sleep(.02)
  parts=[json.loads((case/f'rank-{i}'/'report.json').read_text()) for i in range(2)]
  assert all(x['status']=='COMPLETE' for x in parts),parts
  assert sum(parts[0]['layer_sizes'])==math.factorial(n),parts[0]['layer_sizes']
  assert all(x['layer_sizes']==oracle for x in parts),(n,[x['layer_sizes'] for x in parts],oracle)
  if reference is None:reference=parts[0]['layer_sizes']
  assert all(x['layer_sizes']==reference for x in parts)
  record=dict(mode=mode,lanes=lanes,shards=shards,batch=batch,rows=int(rows),bfs=max(x['bfs_seconds'] for x in parts),setup=max(x['setup_seconds'] for x in parts),wall=time.monotonic()-t,peak_owner_lanes=max(x.get('peak_owner_lanes',0) for x in parts),source_retries=[x['source_retries'] for x in parts],segment_memory=[x['segment_memory'] for x in parts]);records.append(record);print(json.dumps(record),flush=True)
 finally:
  for p in jobs:
   if p.poll() is None:p.terminate()
  for p in jobs:
   if p.poll() is None:
    try:p.wait(timeout=5)
    except subprocess.TimeoutExpired:p.kill();p.wait()
  for f in files:f.close()
(root/'benchmark.json').write_text(json.dumps(records,indent=2));print('EVIDENCE',root)
