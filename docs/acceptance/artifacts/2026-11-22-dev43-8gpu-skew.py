import importlib.metadata,json,math,os,subprocess,time,traceback
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
assert importlib.metadata.version('multigpubfs')=='0.2.0.dev43'
root=Path('/workspace/capacity-eight/results/skew-dev43')
root.mkdir(parents=True,exist_ok=False)
os.environ.update(MGBFS_NATIVE_CACHE='/workspace/capacity-eight/native-cache',
 MGBFS_NATIVE_OFFLINE='1',MGBFS_SEGMENT_ROWS='1048576',
 MGBFS_SEGMENT_OWNER_LANES='4',MGBFS_SEGMENT_EXACT_EXCHANGE='1',
 MGBFS_SEGMENT_BATCH_RECOVERY='0',MGBFS_SEGMENT_LAYER_STATS='1',
 MGBFS_SEGMENT_TRANSPORT_TRACE='1')
busy=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True,capture_output=True,check=True)
(root/'gpu-before.txt').write_text(busy.stdout);assert not busy.stdout.strip()
n=14;k=10;perms=[]
for i in range(k-1):
 a=list(range(n));a[i],a[i+1]=a[i+1],a[i];perms.append(a)
graph=GraphDefinition.permutation(perms,list(range(n)))
oracle=[1]
for j in range(1,k+1):
 w=[0]*(len(oracle)+j-1)
 for i,x in enumerate(oracle):
  for z in range(j):w[i+z]+=x
 oracle=w
assert sum(oracle)==math.factorial(k)
records=[];ids=None
try:
 with persistent_graph_workers() as pool:
  for page in (0,64):
   print('START page',page,flush=True);start=time.monotonic()
   r=run_graph(graph,root/('page-'+str(page)),devices=list(range(8)),autotune=False,
    shards=8,_batch=8192,memory_backend='permutation_bitmap',memory_preset='balanced',
    segment_rows=2**32,segment_strategy='atomic',bitmap_owner_page=page)
   wall=time.monotonic()-start
   assert r['status']=='COMPLETE' and r['layer_sizes']==oracle
   current=json.loads((root/('page-'+str(page))/'states.json').read_text())
   last=tuple(reversed(range(k)))+tuple(range(k,n));previous=set()
   for i in range(k-1):
    a=list(last);a[i],a[i+1]=a[i+1],a[i];previous.add(tuple(a))
   assert [tuple(x) for x in current['current']]==[last]
   pp=list(map(tuple,current['previous_small']))
   assert len(pp)==len(previous) and set(pp)==previous
   ranks=r['segment_memory_by_rank'];assert len(ranks)==8
   assert all(x['bitmap_domain']==math.factorial(k) for x in ranks)
   assert all(x['bitmap_bytes']==56704 for x in ranks)
   now=[w.process.pid for w in pool.workers]
   if ids is None:ids=now
   assert ids==now and len(set(now))==8
   peaks=[x['peak_live_rows'] for x in ranks]
   rec={'page':page,'status':r['status'],'bfs_seconds':r['bfs_seconds'],'wall_seconds':wall,
    'layer_counts_match_mahonian':True,'last_two_exact_full_state_sets':True,
    'peak_live_rows_by_rank':peaks,'max_to_mean_per_rank_peak':max(peaks)/(sum(peaks)/len(peaks)),
    'remote_records_sent_by_rank':[x['remote_records_sent'] for x in ranks],
    'remote_padded_records_by_rank':[x['remote_padded_records'] for x in ranks],
    'bitmap_bytes_by_rank':[x['bitmap_bytes'] for x in ranks],
    'bitmap_domain_by_rank':[x['bitmap_domain'] for x in ranks],
    'rank_pids':now,'ranks':ranks,'runtime_identity':r.get('runtime_identity'),
    'instrumentation':'diagnostic transport and per-layer stats enabled; timings include their overhead'}
   records.append(rec);(root/'records.json').write_text(json.dumps(records,indent=2))
   print('PASS',json.dumps({k:rec[k] for k in ('page','bfs_seconds','peak_live_rows_by_rank','max_to_mean_per_rank_peak','remote_records_sent_by_rank')}),flush=True)
except BaseException:
 (root/'failure.txt').write_text(traceback.format_exc());raise
(root/'complete').write_text('ok\n')