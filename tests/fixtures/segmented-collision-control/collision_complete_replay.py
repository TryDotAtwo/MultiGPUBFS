import json,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parent))
import collision_control_search as c
hits=[]
for kind in ('packed','wide'):
 hit=None
 for i in range(128):
  base=i if kind=='packed' else 1000+i*31
  start=[base+j for j in range(5)]
  layers,inc=c.exact(c.gens,start)
  ro=c.identity(tuple(start))>>63
  need=max(max(Counter(c.owner(x,ro) for x in layer).values(),default=0) for layer in layers)
  crossing=next(((depth+1,child,parents) for depth,items in enumerate(inc) for child,parents in items.items() if len({c.owner(p,ro) for p in parents})>1),None)
  if not crossing:continue
  summary,ev=c.attempt(kind,start,need,need,layers,f'complete-{kind}-{i}')
  if summary['status']=='COMPLETE' and ev:
   full,ev2=c.attempt(kind,start,need,need,layers,f'complete-{kind}-{i}-verified',full=True)
   hit={'summary':summary,'evidence':ev[0],'rerun':full,'rerun_successful_replays':len(ev2),'crossing_depth':crossing[0]}
   print('COMPLETE_REPLAY_FOUND',json.dumps(hit),flush=True);hits.append(hit);break
 if hit is None:print('COMPLETE_REPLAY_NOT_FOUND',kind,flush=True)
(c.root/'complete-replay-evidence.json').write_text(json.dumps(hits,indent=2))
print('COMPLETE_REPLAY_RESULTS',str(c.root),json.dumps(hits),flush=True)
if len(hits)!=2:raise SystemExit(1)
