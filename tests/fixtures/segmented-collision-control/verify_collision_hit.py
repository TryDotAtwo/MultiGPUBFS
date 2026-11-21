"""Test-only follow-up: every pre-failure layer + actual physical route counts."""
import sys,json
from pathlib import Path
from collections import Counter,defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parent))
import collision_control_search as control
from collision_fixture_common import shard
ns=control.__dict__
ns['shard']=shard
cases=Path(sys.argv[1]);hits=json.loads((cases/'evidence.json').read_text());assert hits
for hit in hits:
 h=hit['summary'];start=h['start'];layers,incoming=ns['exact'](ns['gens'],start)
 rec,events=ns['attempt'](h['kind'],start,h['cap'],h['required_cap'],layers,h['kind']+'-verify-exact-trigger-cap',full=True)
 expected_last=next(i for i,L in enumerate(layers) if max(Counter(ns['owner'](x,0) for x in L).values(),default=0)>h['cap'])-1
 assert rec['last_complete']==expected_last,(h,rec,expected_last)
 target=cases/Path(h['out']).name
 summary=json.loads((target/'report.json').read_text());assert summary['owner_cuts']==[0,2**31,2**32]
 per_rank=[]
 for rank in range(2):
  for line in (target/f'rank-{rank}.log').read_text().splitlines():
   if 'SEGMENT_LAYER_STATS ' not in line:continue
   v=json.loads(line.split('SEGMENT_LAYER_STATS ',1)[1]);depth=v['depth'];counts=[0]*4
   for parent in layers[depth-1]:
    if ns['owner'](parent,0)!=rank:continue
    for g in ns['gens']:
     child=tuple(parent[i] for i in g);counts[2*ns['owner'](child,0)+ns['shard'](child)]+=1
   assert counts==v['send_by_peer_shard'],(h['kind'],rank,depth,counts,v['send_by_peer_shard'])
   per_rank.append({'rank':rank,'depth':depth,'expected_and_recorded':counts})
 d,c,_=hit['cross_source_duplicate'];parents=[p for p in incoming[d-1][tuple(c)]]
 parent_sources=sorted({ns['owner'](p,0) for p in parents});assert parent_sources==[0,1]
 target_rank=ns['owner'](tuple(c),0);target_shard=ns['shard'](tuple(c))
 buckets=defaultdict(list)
 for layer in layers:
  for state in layer:buckets[ns['identity'](state)].append(state)
 collision_pair=next((v[:2] for v in buckets.values() if len(v)>=2),None)
 assert collision_pair and collision_pair[0]!=collision_pair[1]
 assert ns['identity'](collision_pair[0])==ns['identity'](collision_pair[1])
 assert ns['owner'](collision_pair[0],0)==ns['owner'](collision_pair[1],0)
 # Every completed snapshot was materialized by the real native BFS, and
 # this additionally checks the physical owner of each exact state.
 for depth in range(rec['last_complete']+1):
  snapshot=ns['root']/(h['kind']+'-verify-exact-trigger-cap-snapshot-'+str(depth))
  for rank in range(2):
   got=Counter(tuple(x) for x in json.loads((snapshot/f'rank-{rank}'/'states.json').read_text())['current'])
   want=Counter({x:1 for x in layers[depth] if ns['owner'](x,0)==rank})
   assert got==want,(h['kind'],rank,depth,got-want,want-got)
 print('COLLISION_PHYSICAL_ROUTE_AND_LAYERS_PASS',json.dumps({'kind':h['kind'],'cap':h['cap'],'last_complete':rec['last_complete'],'routes':per_rank,'cross_source_duplicate':{'depth':d,'state':c,'parents':parents,'source_ranks':parent_sources,'owner':target_rank,'shard':target_shard},'same_origin_replays_again':len(events),'same_full_hash_distinct_states':collision_pair,'same_full_hash_hex':hex(ns['identity'](collision_pair[0])),'snapshot_root':str(ns['root'])}),flush=True)
print('COLLISION_TRIGGER_CAP_FULL_LAYER_CHECKS_PASS',flush=True)
