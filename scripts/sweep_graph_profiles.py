"""Resident sweep reuses verified Graph-window policy at unchanged geometry."""
import hashlib,json,math
from pathlib import Path

def calibrate_or_reuse(calibrator,config,source,root,runtime,*,deadline,cancelled=None,session=None):
 binary=Path(source)/config.get('binary_path','target/release/mgbfs')
 eligible=(session is not None and binary.is_file() and math.factorial(config['n'])//math.factorial(config['r'])>=config['world']*config.get('batch',1)*32)
 key=None
 if eligible:
  from bfs_tail_archive import packed_width
  env={k:v for k,v in config['env'].items() if k not in ('MGBFS_CUDA_GRAPH_BATCHES','MGBFS_CALIBRATION_LAYERS')}
  shape={'n':config['n'],'width':packed_width(config['n'],config['n']-config['r']+1),'world':config['world'],'batch':config.get('batch',1),'env':env,'runtime':runtime,'binary':hashlib.sha256(binary.read_bytes()).hexdigest(),'resident_identity':getattr(session,'identity',None),'generation':getattr(session,'generation',0)}
  key=hashlib.sha256(json.dumps(shape,sort_keys=True).encode()).hexdigest()
  profiles=getattr(session,'graph_profiles',None)
  if profiles is None:profiles={};session.graph_profiles=profiles
  cached=profiles.get(key)
  if cached:
   return {'status':'REUSED_VERIFIED_SWEEP_GRAPH_PROFILE','graph_batches':cached['graph_batches'],'source_pair':cached['source_pair'],'source_identity':cached['identity'],'cache_hit':True,'target_calibration_executed':False,'policy':'same resident GPU group, byte-state geometry, buffers and runtime; source full-state parity evidence'}
 result=calibrator(config,source,root,runtime,deadline=deadline,cancelled=cancelled)
 if key is None:return result
 proof=result.get('identity') or next((s.get('configuration_identity') for s in result.get('samples',[]) if isinstance(s,dict) and s.get('configuration_identity')),None)
 if key and result.get('status')=='CALIBRATED' and proof and type(result.get('graph_batches')) is int and result['graph_batches']>=0 and result.get('samples') and all(isinstance(s,dict) and s.get('full_state_parity') is True for s in result['samples']):
  session.graph_profiles[key]={'graph_batches':result['graph_batches'],'source_pair':[config['n'],config['r']],'identity':proof}
 return result
