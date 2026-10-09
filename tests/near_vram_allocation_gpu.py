from pathlib import Path
import os,json,subprocess,time,threading,traceback
from cayleypy import PermutationGroups
from multigpubfs import from_cayleypy,run_graph
r=Path('/root/universal/near-vram-gate');g=from_cayleypy(PermutationGroups.lrx(14));layers=[];seen=set();front={tuple(g.start)}
for depth in range(7):
 layers.append(sorted(map(list,front)));seen.update(front);front={tuple(g.successor(list(x),i)) for x in front for i in range(g.generator_count)}-seen
samples=[];stop=threading.Event()
def sample():
 while not stop.is_set():
  x=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
  if x.returncode==0:samples.append({'time':time.time(),'gpus':[list(map(int,row.split(','))) for row in x.stdout.splitlines()]})
  stop.wait(.15)
t=threading.Thread(target=sample);t.start();checks=[]
try:
 for mode in ('hash','sorted'):
  os.environ['MGBFS_GENERIC_HISTORY']=mode;os.environ['MGBFS_GENERIC_OWNER_LANES']='4';start=time.time()
  report=run_graph(g,r/(mode+'-prefix'),devices=[0,1],shards=4,autotune=False,backend='generic',max_seconds=40,_profile_layers=6)
  end=time.time();assert report['status']=='INCOMPLETE' and report['reason']=='PROFILE_LAYER_LIMIT',report
  assert report['layer_sizes']==list(map(len,layers)),report
  state=json.loads((r/(mode+'-prefix')/'states.json').read_text());assert sorted(state['current'])==layers[-1] and sorted(state['previous_small'])==layers[-2],(len(state['current']),len(layers[-1]))
  relevant=[x for x in samples if start<=x['time']<=end];peaks={i:max((gpu[1] for x in relevant for gpu in x['gpus'] if gpu[0]==i),default=0) for i in (0,1)}
  assert all(v>=8000 for v in peaks.values()),peaks
  checks.append({'mode':mode,'report':report,'peak_used_mib':peaks,'sample_start':start,'sample_end':end,'states_exact':True})
finally:stop.set();t.join();(r/'vram-samples.json').write_text(json.dumps(samples,indent=2))
receipt={'status':'VERIFIED_NEAR_VRAM_AUTOMATIC_ALLOCATION_PREFIX','checks':checks,'oracle_layer_sizes':list(map(len,layers)),'scope':'Actual >8GiB/card automatic buffer allocations on two RTX3060, six completed LRX14 expansion layers and exact current/previous states; allocation acceptance, not saturated-frontier throughput or full graph completion'};(r/'verification.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({'status':receipt['status'],'peaks':[c['peak_used_mib'] for c in checks],'layer_sizes':receipt['oracle_layer_sizes']}))
