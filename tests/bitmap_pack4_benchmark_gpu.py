"""Multi-GPU comparison; Mahonian counts and terminal state sets.

VRAM sampling is off by default. Set MGBFS_BENCH_SAMPLE_PATTERN to one
comma-separated 0/1 per run (for example, 0,1,0) to request it explicitly.
"""
import json,os,subprocess,sys,time
from collections import Counter
from pathlib import Path
from bitmap_pack4_timing import measured_layer_window
from bitmap_pack4_sampling import sample_pattern, collect_vram_samples, vram_summary
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

n=int(sys.argv[1]);preset=sys.argv[2];sequence=sys.argv[3].split(',')
sampling=sample_pattern(os.environ.get('MGBFS_BENCH_SAMPLE_PATTERN'),len(sequence))
native=os.environ['MGBFS_EXECUTABLE']
world=int(os.environ.get('MGBFS_TEST_DEVICE_COUNT','2'))
devices=list(range(world))
root=Path(os.environ.get('MGBFS_PACK4_EVIDENCE_ROOT','/workspace/mgbfs-pack4-evidence'))/f'benchmark-s{n}-{preset}-{world}gpu-{os.getpid()}'
root.mkdir(parents=True,exist_ok=True)
gens=[]
for i in range(n-1):
 g=list(range(n));g[i],g[i+1]=g[i+1],g[i];gens.append(g)
graph=GraphDefinition.permutation(gens,list(range(n)))
segment_rows=32768 if n<=10 else 1048576
mahonian=[1]
for k in range(1,n+1):
 nxt=[0]*(len(mahonian)+k-1)
 for i,x in enumerate(mahonian):
  for j in range(k):nxt[i+j]+=x
 mahonian=nxt
reverse=tuple(reversed(range(n)))
previous=set()
for i in range(n-1):
 row=list(reverse);row[i],row[i+1]=row[i+1],row[i];previous.add(tuple(row))
running=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'],text=True).strip()
assert not running,('unexpected GPU processes before benchmark',running)
def sample(stop,samples):
 while not stop.is_set():
  try:
   raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True,timeout=3)
   samples.append({'monotonic':time.monotonic(),'gpus':{int(a):int(b) for a,b in (line.split(',') for line in raw.splitlines())}})
  except Exception as e:samples.append({'error':str(e)})
  stop.wait(.5)

records=[]
with persistent_graph_workers() as pool:
 ids=None
 for j,codec in enumerate(sequence):
  assert codec in ('0','1')
  env=dict(os.environ,MGBFS_SEGMENT_PACK4=codec,MGBFS_SEGMENT_EXACT_EXCHANGE='1',
     MGBFS_SEGMENT_ROWS=str(segment_rows),MGBFS_SEGMENT_QUEUE_CAP='8192',
     MGBFS_SEGMENT_OWNER_LANES='4',MGBFS_SEGMENT_BATCH_RECOVERY='1')
  warm=GraphDefinition.permutation([[1,0,2],[0,2,1]],[0,1,2])
  kw=dict(devices=devices,autotune=False,shards=4,_batch=4096,
      memory_backend='permutation_bitmap',memory_preset=preset,segment_rows=segment_rows,
      segment_strategy='atomic',executable=native,_native_env=env)
  w=run_graph(warm,root/f'warm-{j}',**dict(kw,_batch=2))
  assert w['status']=='COMPLETE'
  current=[x.process.pid for x in pool.workers]
  if ids is None:ids=current
  assert current==ids
  print('START',n,preset,'pack4' if codec=='1' else 'unpacked',j,flush=True)
  with collect_vram_samples(sampling[j],sample) as samples:
   start=time.perf_counter()
   r=run_graph(graph,root/f'run-{j}-{codec}',**kw)
  wall=time.perf_counter()-start
  assert r['status']=='COMPLETE' and r['layer_sizes']==mahonian,(j,r['status'],r['layer_sizes'])
  state=json.loads((root/f'run-{j}-{codec}'/'states.json').read_text())
  assert Counter(map(tuple,state['current']))==Counter([reverse])
  assert Counter(map(tuple,state['previous_small']))==Counter(previous)
  assert all(x['resident_codec']==('pack4-v1' if codec=='1' else 'unpacked-soa') for x in r['segment_memory_by_rank'])
  assert [x.process.pid for x in pool.workers]==ids
  geom=[]
  for rr in r['segment_memory_by_rank']:
   rows=sum(sum(sh) for sh in rr['segment_rows_per_bank_by_shard'])
   expected=rows*rr['banks']*(8 if codec=='1' else n)
   assert rr['segment_state_allocated_bytes']==expected,(codec,rr,expected)
   geom.append({k:rr[k] for k in ('rank','banks','actual_queue_capacity','owner_lanes','queue_batch_ceiling',
     'segment_rows_per_bank_by_shard','segment_state_allocated_bytes','segment_counter_allocated_bytes',
     'storage_peak_requested_bytes','bitmap_bytes','peak_live_rows','queues_allocated_bytes','reserve_bytes')})
  center=len(mahonian)//2;lo,hi=(34,44) if n==13 else (max(1,center-6),min(len(mahonian)-1,center+5))
  item={'n':n,'preset':preset,'codec':'pack4-v1' if codec=='1' else 'unpacked-soa',
    'run':j,'status':r['status'],'bfs_seconds':r['bfs_seconds'],'launch_wall_seconds':r['launch_wall_seconds'],
    'caller_wall_seconds':wall,'setup_seconds':r['setup_seconds'],'source_retries':r['source_retries'],
    'window_depths':[lo,hi],'window_seconds':measured_layer_window(r['layer_seconds'],len(r['layer_sizes']),lo,hi),
    'states':sum(mahonian),'all_layer_counts_mahonian':True,'terminal_full_sets':True,
    **vram_summary(sampling[j],samples,devices),'geometry':geom,'pids':ids}
  (root/f'samples-{j}-{codec}.json').write_text(json.dumps(samples))
  records.append(item);(root/'summary.json').write_text(json.dumps(records,indent=2))
  print('DONE',json.dumps(item),flush=True)
print('PACK4_BENCHMARK_COMPLETE',root,flush=True)
