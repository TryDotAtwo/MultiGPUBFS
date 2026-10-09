"""Automatic inverse-closed three-bank dispatch and exact terminal boundaries."""
import os,json,subprocess,time,signal,hashlib
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
r=Path(os.environ.get('MGBFS_ROLLING_PUBLIC_ROOT','/root/universal/parent-terminal'));r.mkdir(exist_ok=True);checks=[]
def prefix(g,layers):
 seen={tuple(g.start)};frontier=seen.copy();out=[]
 for _ in range(layers):
  out.append(sorted(map(list,frontier)));future={tuple(g.successor(list(x),i)) for x in frontier for i in range(len(g.generator_names))}-seen;seen|=future;frontier=future
 return out
for selection in ([0],[1,0]):
 for name,g,capacity in [('cycle',GraphDefinition.matrix(2,1,[([1,1,0,1],257),([1,-1,0,1],257)],[256,2]),3),('wide',GraphDefinition.permutation([list(range(1,17))+[0],[16]+list(range(16))],list(range(17))),4),('signed',GraphDefinition.matrix(2,1,[([-1,0,0,1],0),([1,0,0,-1],0)],[-(1<<40),2]),4)]:
  expected=g.exact_layers(300);folder=r/f'{name}-{len(selection)}';v=run_graph(g,folder,devices=selection,capacity=capacity,shards=1,max_seconds=30,autotune=False,_batch=2)
  assert v['status']=='COMPLETE' and v['plan']['history_layers']==3 and v['backend']=='GENERIC_DISTRIBUTED_EXACT_THREE_BANK_HISTORY';assert v['layer_sizes']==list(map(len,expected));data=json.loads((folder/'states.json').read_text());assert sorted(data['current'])==expected[-1] and sorted(data['previous_small'])==expected[-2]
  checks.append({'case':name,'devices':selection,'automatic_three_bank':True,'complete_exact':True})
# Native admission rejects the int64-wrap pseudo-inverse and keeps directed history.
g=GraphDefinition.matrix(1,1,[([(1<<63)-1],3)],[2]);folder=r/'overflow-fallback';v=run_graph(g,folder,devices=[0,1],shards=4,autotune=False,max_seconds=30);assert v['plan']['history_layers']==1 and v['status']=='COMPLETE';assert v['layer_sizes']==list(map(len,g.exact_layers(8)));checks.append({'case':'overflow-fallback','retained_history':True})
# Deadline on a long modular cycle leaves exact completed layers in both banks.
g=GraphDefinition.matrix(2,1,[([1,1,0,1],65537),([1,-1,0,1],65537)],[0,1]);folder=r/'deadline';v=run_graph(g,folder,devices=[0,1],capacity=3,shards=4,_batch=2,autotune=False,max_seconds=1);assert v['status']=='INCOMPLETE' and v['reason']=='DEADLINE';expected=prefix(g,len(v['layer_sizes']));assert v['layer_sizes']==list(map(len,expected));data=json.loads((folder/'states.json').read_text());assert sorted(data['current'])==expected[-1] and sorted(data['previous_small'])==expected[-2];checks.append({'case':'deadline','completed_layers':len(expected),'previous_current_exact':True})
# Signal actual native rank processes after both readiness receipts, not PID files.
degree=25;g=GraphDefinition.permutation([list(range(1,degree))+[0],[degree-1]+list(range(degree-1)),[1,0]+list(range(2,degree))],list(range(degree)));folder=r/'cancel';definition=r/'cancel-definition.json';definition.write_text(g.to_json());worker=r/'cancel-worker.py';worker.write_text("import json,sys\nfrom pathlib import Path\nfrom multigpubfs import GraphDefinition,run_graph\ng=GraphDefinition.from_dict(json.loads(Path(sys.argv[1]).read_text()))\nrun_graph(g,sys.argv[2],devices=[0,1],capacity=10000,shards=4,_batch=32,autotune=False,max_seconds=60)\n")
log=(r/'cancel-worker.log').open('w');job=subprocess.Popen(['python3',str(worker),str(definition),str(folder)],stdout=log,stderr=subprocess.STDOUT);started=time.monotonic();rank_pids=[]
while time.monotonic()-started<30:
 if job.poll() is not None:raise RuntimeError('CANCEL_WORKER_ENDED_BEFORE_READY')
 ready=all((folder/f'rank-{i}.log').exists() and 'MGBFS_GRAPH_READY' in (folder/f'rank-{i}.log').read_text() for i in (0,1))
 if ready:
  for proc in Path('/proc').iterdir():
   if not proc.name.isdigit():continue
   try:argv=(proc/'cmdline').read_bytes().split(b'\0')
   except OSError:continue
   if b'graph-rank' in argv and any(str(folder/f'rank-{i}').encode() in argv for i in (0,1)):rank_pids.append(int(proc.name))
  assert len(rank_pids)==2,rank_pids
  for pid in rank_pids:os.kill(pid,signal.SIGTERM)
  break
 time.sleep(.01)
else:raise RuntimeError('CANCEL_READINESS_OBSERVATION_TIMEOUT_NO_RESTART')
job.wait(timeout=30);log.close();assert job.returncode==0;v=json.loads((folder/'report.json').read_text());assert v['status']=='INCOMPLETE' and v['reason']=='CANCELLED';expected=prefix(g,len(v['layer_sizes']));assert v['layer_sizes']==list(map(len,expected));data=json.loads((folder/'states.json').read_text());assert sorted(data['current'])==expected[-1];assert sorted(data['previous_small'])==(expected[-2] if len(expected)>1 and len(expected[-2])<1000 else []);checks.append({'case':'cancel','actual_native_ranks':rank_pids,'previous_current_exact':True})
degree=25;g=GraphDefinition.permutation([list(range(1,degree))+[0],[degree-1]+list(range(degree-1)),[1,0]+list(range(2,degree))],list(range(degree)));folder=r/'large-resource-sample';v=run_graph(g,folder,devices=[0,1],capacity=2000,shards=4,_batch=32,autotune=False,max_seconds=30);assert v['status']=='INCOMPLETE' and v['reason'].startswith('RESOURCE_');expected=prefix(g,len(v['layer_sizes']));assert v['layer_sizes']==list(map(len,expected));data=json.loads((folder/'states.json').read_text());assert len(expected[-1])>1000;assert len(data['current'])==1000 and {tuple(x) for x in data['current']}<={tuple(x) for x in expected[-1]};assert sorted(data['previous_small'])==(expected[-2] if len(expected)>1 and len(expected[-2])<1000 else []);checks.append({'case':'large-resource-sample','current_global_size':len(expected[-1]),'sample_limit':1000,'previous_threshold_exact':True})
receipt={'status':'VERIFIED_AUTOMATIC_ROLLING_ONE_TWO_GPU_AND_TERMINAL_BOUNDARIES','checks':checks,'scope':'actual one/two RTX3060 automatic dispatch, deadline/cancel exact prefix, directed overflow fallback; larger scales unverified'};(r/'verification.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
