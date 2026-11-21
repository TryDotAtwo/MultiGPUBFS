"""Independent layer multisets for capped segmented BFS on 2 physical GPUs.

The rank-cap oracle follows exact source hashing/owner assignment but computes
layers from explicit permutations and a CPU set-based BFS, independently of
MultiGPUBFS and CayleyPy traversal. All stored states are compared as multisets.
"""
import json,os,tempfile
from collections import Counter,defaultdict
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

MASK=(1<<64)-1
SEED=0x19f856a2

def mix(x):
 x^=x>>30;x=(x*0xbf58476d1ce4e5b9)&MASK;x^=x>>27;x=(x*0x94d049bb133111eb)&MASK;return x^(x>>31)
def identity(state,bits):
 h=SEED
 for i,v in enumerate(state):h=mix(h^(v&MASK)^i)
 h=mix(h)
 return h if bits==64 else h&((1<<bits)-1) if bits else 0
def physical_owner(state,root,bits):
 o=((identity(state,bits)>>32)*2)>>32
 return 0 if o==root else root if o==0 else o

def exact(gens,start):
 cur={tuple(start)};seen=set(cur);layers=[];predecessors=[]
 while cur:
  layers.append(cur);future=set();incoming=defaultdict(set)
  for p in cur:
   for g in gens:
    ch=tuple(p[i] for i in g)
    if ch not in seen:future.add(ch);incoming[ch].add(p)
  predecessors.append(incoming);seen.update(future);cur=future
 return layers,predecessors

def rank_counts(layers,bits,directed):
 root=((identity(next(iter(layers[0])),bits)>>32)*2)>>32
 cumulative=Counter();by_depth=[]
 for layer in layers:
  current=Counter(physical_owner(s,root,bits) for s in layer)
  if directed:cumulative.update(current);by_depth.append(cumulative.copy())
  else:by_depth.append(current)
 return root,by_depth

def verify_snapshot(out,depth,expected):
 rank_states=[json.loads((out/f'rank-{rank}'/'states.json').read_text()) for rank in range(2)]
 found=Counter(tuple(x) for rank in rank_states for x in rank['current'])
 wanted=Counter({x:1 for x in expected[depth]})
 assert found==wanted,(out,depth,'missing',wanted-found,'extra_or_duplicates',found-wanted)
 assert all(v==1 for v in found.values())
 if depth:
  prev=Counter(tuple(x) for rank in rank_states for x in rank['previous_small'])
  assert prev==Counter({x:1 for x in expected[depth-1]}),(out,depth,'previous',prev)

def adjacent(n):
 result=[]
 for i in range(n-1):
  g=list(range(n));g[i],g[i+1]=g[i+1],g[i];result.append(g)
 return result
cases=[('s4',adjacent(4),list(range(4)),False),
       ('directed-c6',[[(i+k)%6 for i in range(6)] for k in (1,4)],list(range(6)),True)]
native=os.environ['MGBFS_EXECUTABLE'];root=Path(tempfile.mkdtemp(prefix='warp-race-oracle-',dir='/workspace'))
records=[]
with persistent_graph_workers() as pool:
 for strategy in ('atomic','warp','batch'):
  for name,gens,start,directed in cases:
   layers,preds=exact(gens,start);graph=GraphDefinition.permutation(gens,start)
   for bits in (64,0):
    root_owner,counts=rank_counts(layers,bits,directed)
    need=max(max(c.values(),default=0) for c in counts)
    assert need>1,(name,bits,counts)
    failing=next(i for i,c in enumerate(counts) if max(c.values(),default=0)>need-1)
    if name=='s4' and bits==64:
     crossing=[]
     for i,incoming in enumerate(preds):
      for child,parents in incoming.items():
       sources={physical_owner(p,root_owner,bits) for p in parents}
       if len(sources)>1:crossing.append((i+1,child,sorted(sources)))
     assert crossing,'oracle did not produce duplicate children from different ranks'
     print('CROSS_RANK_DUPLICATE_EXPECTED',crossing[0],flush=True)
    for cap in (None,need,need-1):
     max_depth=(failing-1 if cap==need-1 else len(layers)-1)
     env=dict(os.environ,MGBFS_SEGMENT_TEST_BATCH='1',MGBFS_SEGMENT_TEST_HASH_BITS=str(bits),
              MGBFS_SEGMENT_OWNER_LANES='2',MGBFS_SEGMENT_EXACT_EXCHANGE='1',MGBFS_SEGMENT_LAYER_STATS='1')
     if cap is None:env.pop('MGBFS_SEGMENT_MAX_RANK_STATES',None)
     else:env['MGBFS_SEGMENT_MAX_RANK_STATES']=str(cap)
     for depth in range(max_depth+1):
      out=root/f'{strategy}-{name}-bits{bits}-cap{cap}-d{depth}'
      r=run_graph(graph,out,devices=[0,1],shards=4,autotune=False,_batch=8,
       memory_backend='segmented24',segment_strategy=strategy,segment_rows=3,executable=native,
       _native_env=env,_profile_layers=depth)
      assert r['status']=='INCOMPLETE' and r['reason']=='PROFILE_LAYER_LIMIT',(out,r)
      assert r['layer_sizes']==list(map(len,layers[:depth+1])),(out,r['layer_sizes'],list(map(len,layers)))
      verify_snapshot(out,depth,layers)
     out=root/f'{strategy}-{name}-bits{bits}-cap{cap}-terminal'
     r=run_graph(graph,out,devices=[0,1],shards=4,autotune=False,_batch=8,
      memory_backend='segmented24',segment_strategy=strategy,segment_rows=3,executable=native,
      _native_env=env,_profile_layers=999)
     expected_depth=max_depth
     if cap==need-1:assert r['status']=='INCOMPLETE' and r['reason']=='RESOURCE_10',(out,r)
     else:assert r['status']=='COMPLETE',(out,r)
     assert r['layer_sizes']==list(map(len,layers[:expected_depth+1])),(out,r)
     verify_snapshot(out,expected_depth,layers)
     if name=='s4' and bits==64 and cap is None:
      target_depth,target_child,sources=crossing[0]
      expected_send=[[0]*8 for _ in range(2)]
      for parent in layers[target_depth-1]:
       source=physical_owner(parent,root_owner,bits)
       for perm in gens:
        child=tuple(parent[i] for i in perm);owner=physical_owner(child,root_owner,bits)
        shard=((identity(child,bits)&0xffffffff)*4)>>32
        expected_send[source][owner*4+shard]+=1
      for source in (0,1):
       log=(out/f'rank-{source}.log').read_text()
       events=[json.loads(line.split('SEGMENT_LAYER_STATS ',1)[1]) for line in log.splitlines() if 'SEGMENT_LAYER_STATS ' in line]
       actual=[x for x in events if x['depth']==target_depth]
       assert len(actual)==1 and actual[0]['send_by_peer_shard']==expected_send[source],(source,target_depth,actual,expected_send[source])
      print('CROSS_RANK_ROUTE_COUNTS_MATCH',target_depth,target_child,sources,flush=True)
     rec=dict(strategy=strategy,case=name,hash_bits=bits,cap=cap,required_cap=need,
              expected_failure_depth=failing if cap==need-1 else None,
              status=r['status'],depth=expected_depth,layers_checked=max_depth+1,root=str(out))
     records.append(rec);print('PASS LAYER MULTISETS',json.dumps(rec),flush=True)
(root/'evidence.json').write_text(json.dumps(records,indent=2));print('ALL_LAYER_MULTISET_CASES_PASS',root,flush=True)
