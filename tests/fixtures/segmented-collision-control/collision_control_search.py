"""Test-only forced warp interleaving, traced same-origin replay, CPU oracle."""
import os,sys,json,tempfile
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parent))
from collision_fixture_common import adjacent,exact,identity,owner,logs,verify
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
root=Path(tempfile.mkdtemp(prefix='collision-controlled-e2e-',dir=os.environ.get('MGBFS_COLLISION_OUTPUT','/tmp')));native=os.environ['MGBFS_EXECUTABLE'];gens=adjacent(5)

def events(out,marker):
 found=[]
 for path in sorted(out.glob('rank-*.log')):
  for line in path.read_text(errors='replace').splitlines():
   if marker in line:found.append(json.loads(line.split(marker,1)[1]))
 return found

def attempt(kind,start,cap,need,layers,tag,full=False):
 graph=GraphDefinition.permutation(gens,start);out=root/tag
 env=dict(os.environ,MGBFS_SEGMENT_TEST_BATCH='1',MGBFS_SEGMENT_TEST_HASH_BITS='63',MGBFS_TEST_HASH_TWO_SHARDS='1',MGBFS_TEST_COLLISION_TRACE='1',MGBFS_TEST_COLLISION_CONTROL='1',MGBFS_SEGMENT_OWNER_LANES='2',MGBFS_SEGMENT_EXACT_EXCHANGE='1',MGBFS_SEGMENT_LAYER_STATS='1')
 if cap is None:env.pop('MGBFS_SEGMENT_MAX_RANK_STATES',None)
 else:env['MGBFS_SEGMENT_MAX_RANK_STATES']=str(cap)
 with persistent_graph_workers():
  r=run_graph(graph,out,devices=[0,1],shards=2,autotune=False,_batch=16,memory_backend='segmented24',segment_strategy='warp',segment_rows=1,executable=native,_native_env=env,_profile_layers=999)
  last=len(r['layer_sizes'])-1
  if cap is None or cap>=need:assert r['status']=='COMPLETE',(tag,r)
  else:assert r['status']=='INCOMPLETE' and r['reason']=='RESOURCE_10',(tag,r)
  assert r['layer_sizes']==list(map(len,layers[:last+1])),(tag,r)
  verify(out,last,layers)
  controls=events(out,'COLLISION_TEST_CONTROL ');arms=events(out,'COLLISION_TEST_ARM ');owner_events,intervals=logs(out)
  result=[]
  for c in controls:
   if not(c['held'] and c['deferred'] and c['released'] and c['replayed'] and not c['timeouts']):continue
   earlier=[a for a in controls if a['rank']==c['rank'] and a['accept_id']==c['accept_id'] and a['trace_id']<c['trace_id'] and a['child']==c['child'] and a['held'] and a['deferred'] and a['released'] and not a['timeouts']]
   if not earlier:continue
   x=earlier[-1]
   retry=[a for a in owner_events if a['rank']==x['rank'] and a['accept_id']==x['accept_id'] and a['trace_id']==x['trace_id'] and a['shard']==0 and a['error']&16 and a['spill']]
   later=[a for a in owner_events if a['rank']==c['rank'] and a['accept_id']==c['accept_id'] and a['trace_id']==c['trace_id'] and a['shard']==0 and a['accepted_after']>a['accepted_before']]
   ov=[v for v in intervals if v['rank']==x['rank'] and v['trace_id']==x['trace_id'] and v['overlap_ms']>0]
   if retry and later and ov:result.append(dict(control_before=x,control_replayed=c,retry=retry[0],later=later[0],interval=ov[0]))
  if full:
   for depth in range(last+1):
    snapshot=root/(tag+'-snapshot-'+str(depth))
    q=run_graph(graph,snapshot,devices=[0,1],shards=2,autotune=False,_batch=16,memory_backend='segmented24',segment_strategy='warp',segment_rows=1,executable=native,_native_env=env,_profile_layers=depth)
    assert q['status']=='INCOMPLETE' and q['reason']=='PROFILE_LAYER_LIMIT' and q['layer_sizes']==list(map(len,layers[:depth+1])),(str(snapshot),q)
    verify(snapshot,depth,layers)
  summary=dict(kind=kind,start=start,cap=cap,required_cap=need,status=r['status'],reason=r['reason'],last_complete=last,layer_snapshots_checked=last+1 if full else 0,arms=len(arms),control_records=len(controls),successful_same_origin_replays=len(result),out=str(out))
  print('CONTROL_SEARCH',json.dumps(summary),flush=True)
  return summary,result

def main():
    found=[]

    for kind in ('packed','wide'):
     hit=None
     for i in range(32):
      start=[(i*5 if kind=='packed' else 1000+i*31)+j for j in range(5)]
      layers,inc=exact(gens,start);ro=identity(tuple(start))>>63
      counts=[Counter(owner(x,ro) for x in L) for L in layers];need=max(max(c.values(),default=0) for c in counts)
      crossing=[(d+1,c,sorted({owner(p,ro) for p in ps})) for d,a in enumerate(inc) for c,ps in a.items() if len({owner(p,ro) for p in ps})>1]
      if not crossing:continue
      for cap in (need-1,need-2):
       if cap<1:continue
       summary,evidence=attempt(kind,start,cap,need,layers,f'{kind}-{i}-cap{cap}')
       if evidence:
        hit=dict(summary=summary,evidence=evidence[0],cross_source_duplicate=crossing[0]);print('CONTROL_FOUND',json.dumps(hit),flush=True);break
      if hit:break
     if hit:
      found.append(hit)
      summary=hit['summary'];layers,_=exact(gens,summary['start'])
      for cap,label in ((None,'uncapped'),(summary['required_cap'],'min-cap'),(summary['required_cap']-1,'min-cap-minus-one')):
       rec,ev=attempt(kind,summary['start'],cap,summary['required_cap'],layers,f'{kind}-verify-{label}',full=True)
       hit.setdefault('full_checks',[]).append(rec)
     else:print('CONTROL_NOT_FOUND',kind,flush=True)
    (root/'evidence.json').write_text(json.dumps(found,indent=2));print('CONTROL_SEARCH_RESULTS',root,json.dumps(found),flush=True)
    if {x['summary']['kind'] for x in found}!={'packed','wide'}:raise SystemExit('CONTROL_GATE_OPEN')


if __name__=="__main__":main()
