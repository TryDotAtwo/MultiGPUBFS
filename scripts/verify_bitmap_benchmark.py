"""Compare complete rank-local final layers without exporting state files.

By default compare rank-local sets for the same owner mapping. With
--across-owners compare global sets when different backends assign owners
differently. Counts must show both layers are complete rather than sampled.
Only digests and counts are emitted; input state files remain on the GPU host.
"""
import argparse,hashlib,json
from pathlib import Path


def canonical(value):
 encoded=[json.dumps(x,sort_keys=True,separators=(',',':')) for x in value]
 if len(set(encoded))!=len(encoded):raise ValueError('duplicate states in retained layer')
 return sorted(encoded)


def digest(encoded):
 return hashlib.sha256(('\n'.join(encoded)).encode()).hexdigest()


def compare(cases, across_owners=False):
 if len(cases)<2:raise ValueError('at least two cases are required')
 baseline=None;out=[]
 for case in cases:
  case=Path(case); ranks=[];rank=0
  while (case/f'rank-{rank}'/'report.json').is_file():
   report=json.loads((case/f'rank-{rank}'/'report.json').read_text())
   state_bytes=(case/f'rank-{rank}'/'states.json').read_bytes()
   if hashlib.sha256(state_bytes).hexdigest()!=report['states_sha256']:raise ValueError('retained state file hash mismatch')
   state=json.loads(state_bytes)
   if report['rank']!=rank or report['status']!='COMPLETE':raise ValueError('rank did not finish')
   current=canonical(state['current']);previous=canonical(state['previous_small'])
   ranks.append((report,current,previous));rank+=1
  if not ranks or len(ranks)!=ranks[0][0]['world']:raise ValueError('missing rank receipt')
  vector=ranks[0][0]['layer_sizes']
  if any(x[0]['layer_sizes']!=vector or x[0]['graph_digest']!=ranks[0][0]['graph_digest'] or x[0]['world']!=len(ranks) for x in ranks):raise ValueError('rank graph identity or layer vectors disagree')
  if sum(len(x[1]) for x in ranks)!=vector[-1] or sum(len(x[2]) for x in ranks)!=vector[-2]:raise ValueError('retained final layers are incomplete')
  # Even rank-local comparisons must reject the same state present on two
  # ranks; two identically wrong runs must never verify each other.
  current=sorted(s for _,cur,_ in ranks for s in cur)
  previous=sorted(s for _,_,prev in ranks for s in prev)
  current_set=set(current);previous_set=set(previous)
  if len(current)!=len(current_set) or len(previous)!=len(previous_set):raise ValueError('duplicate state across owners')
  if not current_set.isdisjoint(previous_set):raise ValueError('retained layers overlap')
  owners=[{'rank':i,'current_count':len(cur),'previous_count':len(prev),'current_set_sha256':digest(cur),'previous_set_sha256':digest(prev)} for i,(_,cur,prev) in enumerate(ranks)]
  value={'graph_digest':ranks[0][0]['graph_digest'],'world':len(ranks),'layer_sizes':vector}
  if across_owners:
   value['global_retained']={'current_count':len(current),'previous_count':len(previous),'current_set_sha256':digest(current),'previous_set_sha256':digest(previous)}
  else:value['owners']=owners
  if baseline is None:baseline=value
  elif value!=baseline:raise ValueError('retained state sets or layer vectors disagree')
  out.append({'case':case.name,'original_report_sha256':[hashlib.sha256((case/f'rank-{i}'/'report.json').read_bytes()).hexdigest() for i in range(len(ranks))]})
 scope='global layer counts plus exact global sets of the last two retained layers' if across_owners else 'global layer counts plus exact sets of last two retained layers per rank'
 return {'status':'VERIFIED_SAME_COMPLETE_RETAINED_LAYERS','scope':scope+', not a comparison of all intermediate state sets','cases':out,'comparison':baseline}

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--across-owners',action='store_true');parser.add_argument('cases',nargs='+');args=parser.parse_args()
 print(json.dumps(compare(args.cases,across_owners=args.across_owners),indent=2))
