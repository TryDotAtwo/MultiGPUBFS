"""Select compatible GPU profiles separately by measured parent-frontier size."""
import math

def select_size_profiles(pilots, world, *, threshold=None):
 threshold=threshold or world*16384
 if not pilots:return []
 depth=min(len(p['layer_seconds']) for p in pilots)
 sizes=pilots[0]['layer_sizes'][:depth+1]
 if any(p['layer_sizes'][:depth+1]!=sizes for p in pilots):raise RuntimeError('AUTOTUNE_PREFIX_CORRECTNESS_MISMATCH')
 results=[]
 for lower,upper in ((0,threshold),(threshold,None)):
  indices=[i for i in range(1,depth) if sizes[i]>=lower and (upper is None or sizes[i]<upper)]
  scores=[sum(p['layer_seconds'][i] for i in indices) for p in pilots]
  if any(not math.isfinite(v) or v<0 for v in scores):raise RuntimeError('AUTOTUNE_INVALID_TIMING')
  measured=len(pilots)>=2 and len(indices)>=2 and min(scores,default=0)>=.002
  winner=min(range(len(scores)),key=scores.__getitem__) if measured else 0
  if scores and scores[winner]>=scores[0]*.95:winner=0
  p=pilots[winner]
  results.append(dict(minimum_frontier=lower,batch=p['batch'],generator_backend=p['generator_backend'],measured=measured,scores_seconds=scores,depths=indices,maximum_measured_frontier=max((sizes[i] for i in indices),default=0),status='MEASURED_SIZE_PROFILE' if measured else 'UNMEASURED_CONSERVATIVE_PROFILE'))
 return results


def save_online_profiles(report):
 """Persist validated live-chunk choices separately from matched prefix evidence."""
 import hashlib,json,os,tempfile
 from pathlib import Path
 profile=report.get('autotune',{})
 identity=profile.get('identity');events=report.get('online_size_profile_events',[])
 if not identity or not events or profile.get('status')!='MEASURED_FRONTIER_SIZE_PROFILES':return
 entries=list(profile.get('size_profiles',[]))
 for e in events:
  winner=e.get('winner');choices=e.get('choices',[])
  if type(winner) is not int or not 0<=winner<len(choices):continue
  batch,gemm=choices[winner];frontier=e.get('frontier_global',0)
  if type(batch) is not int or not 0<batch<=profile['batch'] or type(frontier) is not int or frontier<=0:continue
  entries=[p for p in entries if p['minimum_frontier']!=frontier]
  entries.append(dict(minimum_frontier=frontier,batch=batch,generator_backend='gemm' if gemm else 'cuda',measured=True,status='LIVE_CHUNK_EMPIRICAL_PROFILE',maximum_measured_frontier=frontier))
 entries.sort(key=lambda p:p['minimum_frontier'])
 saved=dict(profile,size_profiles=entries,online_profile_evidence=events)
 root=Path(os.environ.get('MGBFS_PROFILE_CACHE',str(Path.home()/'.cache/multigpubfs/profiles')))
 key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();root.mkdir(parents=True,exist_ok=True)
 fd,path=tempfile.mkstemp(prefix='size-'+key+'-',suffix='.tmp',dir=root)
 with os.fdopen(fd,'w') as f:json.dump(saved,f)
 os.replace(path,root/('size-'+key+'.json'))
