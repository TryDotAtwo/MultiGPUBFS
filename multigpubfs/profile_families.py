"""Compatible-family startup reuse; borrowed evidence is not target measurement."""
import hashlib,json,os,tempfile,math
from pathlib import Path
from .profile_cache import read_mapping, write_mapping

def family(graph,plan):
 from .generation import gemm_supported
 from .specialized import match_lrx,exact_specialized_supported
 if not all(k in plan for k in ('state_bytes','history_layers')):return None
 a=graph.action;n=graph.state_elements
 return {'action':a['kind'],'shape_bucket':1<<(n-1).bit_length(),'state_bytes':plan['state_bytes'],
  'packed_permutation':a['kind']=='permutation' and n<=24 and all(0<=v<=255 for v in graph.start),
  'generators':graph.generator_count,'history_layers':plan['history_layers'],
  'matrix_shape':[a.get('rows'),a.get('cols')] if a['kind']=='matrix' else None,
  'matrix_moduli':[g['modulo'] for g in a['generators']] if a['kind']=='matrix' else None,
  'gemm_exact':gemm_supported(graph),'lrx':match_lrx(graph) is not None,
  'specialized_exact':exact_specialized_supported(graph)}

def identity(graph,plan,exact):
 if family(graph,plan) is None:return None
 return {'schema':1,'family':family(graph,plan),'selector':exact['selector_sha256'],
  'configuration':exact['configuration_digest'],'dependencies':exact['native_dependencies'],
  'native':exact['native_sha256'],'devices':exact['devices'],'hardware':exact['hardware'],
  'topology':exact['topology'],'allow_specialized':exact['allow_specialized'],'environment':exact['environment']}

def path(root,identity):
 return Path(root)/('family-'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()+'.json')

def store(root,identity,profile):
 if identity is None:return {'status':'SKIPPED','stage':'identity','errno':None}
 if profile.get('status')!='MEASURED_EQUAL_PREFIX_GPU_PROFILE' or not profile.get('measured'):return {'status':'SKIPPED','stage':'unmeasured','errno':None}
 return write_mapping(path(root,identity),{'identity':identity,'profile':profile},prefix='family-')

def load(root,identity):
 if identity is None:return None
 data,_status=read_mapping(path(root,identity))
 if data is None or data.get('identity')!=identity:return None
 p=data.get('profile')
 if not isinstance(p,dict) or p.get('status')!='MEASURED_EQUAL_PREFIX_GPU_PROFILE' or p.get('measured') is not True:return None
 if p.get('backend') not in ('generic','shard_ab_hash','shard_ab_sort_merge'):return None
 if type(p.get('shards')) is not int or not 1<=p['shards']<=4096 or p.get('batch_fraction') not in (1.,.25):return None
 if p.get('generator_backend') not in ('cuda','gemm') or (p['generator_backend']=='gemm' and not identity['family']['gemm_exact']):return None
 if p.get('history_algorithm') not in ('HASH','SORTED_RUNS'):return None
 if p.get('backend')=='generic' and (p.get('transport') not in ('full','parent') or p.get('candidate_order') not in ('none','radix')):return None
 if p['history_algorithm']=='SORTED_RUNS' and (p.get('owner_lanes') not in (1,2,4,8) or p.get('candidate_order')!='none'):return None
 if p['backend']!='generic' and (not identity['allow_specialized'] or not identity['family']['specialized_exact'] or p.get('specialized_transport') not in ('HOST_SIZED_NCCL','NCCL_LSA')):return None
 if not p.get('scores_seconds') or any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in p['scores_seconds']):return None
 return p

def transferred(saved,target,plan,elapsed):
 keys=('backend','shards','batch_fraction','transport','candidate_order','history_algorithm','owner_lanes','generator_backend','specialized_mode','specialized_transport')
 out={k:saved[k] for k in keys if k in saved}
 out.update(status='REUSED_COMPATIBLE_GPU_PROFILE',measured=False,cache_hit=True,
  seconds=elapsed,identity=target,source_graph_digest=saved['identity']['graph'],
  source_profile_identity=saved['identity'],source_scores_seconds=saved['scores_seconds'],
  source_pilots=saved.get('pilots',[]),batch=max(1,int(plan['batch']*saved['batch_fraction'])),
  scope='compatible-family source measurement; target graph freshly memory-admitted; target throughput not measured')
 return out
