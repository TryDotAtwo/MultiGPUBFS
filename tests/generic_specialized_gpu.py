from pathlib import Path
import os,json,sys,traceback
r=Path(os.environ.get('MGBFS_SPECIALIZED_GATE_ROOT','/root/universal/specialized-package-gate'));r.mkdir(exist_ok=True);from multigpubfs import GraphDefinition
from multigpubfs.native_distribution import native_runtime
from multigpubfs.specialized import query,run_specialized
try:
 native,env=native_runtime();env['NCCL_P2P_DISABLE']='1';root=r/'cases';root.mkdir();checks=[]
 def graph(start):
  n=len(start);return GraphDefinition.permutation([list(range(1,n))+[0],[n-1]+list(range(n-1)),[1,0]+list(range(2,n))],start)
 for devices in ([0],[1,0]):
  for mode in ('HASH','SORT_MERGE'):
   for name,start,capacity in [('relabel',[1<<40,-9,7,2,5],256),('repeated',[-9,1<<40,7,7,7],256),('wide-all-equal',[-(1<<60)]*25,256),('large-small',list(range(8)),40320)]:
    g=graph(start);folder=root/(name+'-'+str(len(devices))+'-'+mode);v=run_specialized(g,folder,native=native,env=env,devices=devices,capacity=capacity,batch=4096,max_seconds=60,mode=mode)
    expected=g.exact_layers(50000);assert v['status']=='COMPLETE' and v['layer_sizes']==list(map(len,expected)),v
    saved=json.loads((folder/'states.json').read_text());assert sorted(saved['current'])==expected[-1];assert sorted(saved['previous_small'])==(expected[-2] if len(expected)>1 and len(expected[-2])<1000 else [])
    checks.append({'case':name,'devices':devices,'mode':mode,'states':sum(v['layer_sizes']),'all_layers_terminal_exact':True});print(checks[-1],flush=True)
 g=graph(list(range(8)));expected=g.exact_layers(50000);v=run_specialized(g,root/'resource',native=native,env=env,devices=[0,1],capacity=256,batch=32,max_seconds=30)
 assert v['status']=='INCOMPLETE' and v['reason']=='RESOURCE_STOP',v
 saved=json.loads((root/'resource/states.json').read_text());depth=len(v['layer_sizes'])-1;assert v['layer_sizes']==list(map(len,expected[:depth+1]));assert len(saved['current'])<=1000 and all(row in expected[depth] for row in saved['current']);assert sorted(saved['previous_small'])==(expected[depth-1] if depth>0 and len(expected[depth-1])<1000 else [])
 checks.append({'case':'resource','sample_count':len(saved['current']),'completed_depth':depth,'unprocessed_sample_in_completed_layer':True})
 g=graph(list(range(12)));v=run_specialized(g,root/'deadline',native=native,env=env,devices=[0,1],capacity=1000000,batch=2,max_seconds=5)
 assert v['status']=='INCOMPLETE' and v['reason']=='DEADLINE',v
 checks.append({'case':'actual-sigterm-deadline','layer_sizes':v['layer_sizes'],'bounded_sample':True})
 v={'status':'VERIFIED_SPECIALIZED_PREFIX_ORACLES_HASH_SORT_TERMINAL','checks':checks};(r/'specialized-acceptance-status.json').write_text(json.dumps(v))
except Exception:
 (r/'specialized-acceptance-status.json').write_text(json.dumps({'status':'ERROR','trace':traceback.format_exc()}));raise
