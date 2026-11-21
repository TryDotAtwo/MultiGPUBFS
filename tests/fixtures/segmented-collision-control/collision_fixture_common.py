"""Test-only CPU oracle and routing helpers for the collision-control fixture."""
import json,os,tempfile,sys
from pathlib import Path
from collections import Counter,defaultdict
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph
MASK=(1<<64)-1;SEED=0x19f856a2

def mix(x):
 x^=x>>30;x=(x*0xbf58476d1ce4e5b9)&MASK;x^=x>>27;x=(x*0x94d049bb133111eb)&MASK;return x^(x>>31)
def identity(state):
 h=SEED
 for i,v in enumerate(state):h=mix(h^(v&MASK)^i)
 return mix(h)&0x8000000080000000
def owner(state,root):
 # The tested plans set explicit owner_cuts [0, 2^31, 2^32], so the
 # CUDA/Rust owner map stays identity even when the seed hashes to rank 1.
 return identity(state)>>63
def shard(state):return (identity(state)>>31)&1

def adjacent(n):
 gs=[]
 for i in range(n-1):
  g=list(range(n));g[i],g[i+1]=g[i+1],g[i];gs.append(g)
 return gs
def exact(gens,start):
 cur={tuple(start)};seen=set(cur);layers=[];incoming=[]
 while cur:
  layers.append(cur);nxt=set();inc=defaultdict(set)
  for p in cur:
   for g in gens:
    c=tuple(p[i] for i in g)
    if c not in seen:nxt.add(c);inc[c].add(p)
  incoming.append(inc);seen.update(nxt);cur=nxt
 return layers,incoming

def logs(out):
 events=[];intervals=[]
 for path in sorted(out.glob('rank-*.log')):
  for line in path.read_text(errors='replace').splitlines():
   if 'COLLISION_TEST_OWNER ' in line:events.append(json.loads(line.split('COLLISION_TEST_OWNER ',1)[1]))
   if 'COLLISION_TEST_GPU_INTERVAL ' in line:intervals.append(json.loads(line.split('COLLISION_TEST_GPU_INTERVAL ',1)[1]))
 return events,intervals

def verify(out,depth,layers):
 data=[json.loads((out/f'rank-{r}'/'states.json').read_text()) for r in range(2)]
 for key,number in [('current',depth),('previous_small',depth-1)]:
  if number<0:continue
  got=Counter(tuple(x) for d in data for x in d[key]);want=Counter({x:1 for x in layers[number]})
  assert got==want,(str(out),depth,key,'missing',list((want-got).items())[:4],'extra',list((got-want).items())[:4])
