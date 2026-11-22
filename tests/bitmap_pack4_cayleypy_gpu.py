"""CayleyPy API with persistent native rank workers and codec switches."""
import json,os,tempfile
from collections import Counter
from pathlib import Path
from cayleypy import CayleyGraph,PermutationGroups
import multigpubfs
from multigpubfs import NativeOptions,persistent_graph_workers
from multigpubfs.graph_definition import from_cayleypy

root=Path(tempfile.mkdtemp(prefix='pack4-cayleypy-',dir='/workspace'))
os.environ['MGBFS_SEGMENT_PACK4']='1'
os.environ['MGBFS_SEGMENT_TEST_BATCH']='1'
os.environ['MGBFS_SEGMENT_ROWS']='3'
def layers(gens,start):
 front={tuple(start)};seen=set(front);out=[]
 while front:
  out.append(front)
  nxt={tuple(row[i] for i in g) for row in front for g in gens}-seen
  seen.update(nxt);front=nxt
 return out
def verify(result,expected,codec,complete):
 report=result.report;depth=len(report['layer_sizes'])-1
 assert result.bfs_completed is complete
 assert report['layer_sizes']==list(map(len,expected[:depth+1]))
 assert all(r['resident_codec']==codec for r in report['segment_memory_by_rank'])
 state=json.loads((result.output/'states.json').read_text())
 assert Counter(map(tuple,state['current']))==Counter(expected[depth])
 if depth:assert Counter(map(tuple,state['previous_small']))==Counter(expected[depth-1])
 if complete:assert depth==len(expected)-1
 return depth

multigpubfs.enable_native(recovery=False)
try:
 with persistent_graph_workers() as pool:
  ids=None
  graph=CayleyGraph(PermutationGroups.coxeter(5),device='cpu')
  gens=from_cayleypy(graph).action['generators']
  for i,(start,preset,cap,codec) in enumerate([
   ([0,1,2,3,4],'balanced',None,True),
   ([0,0,1,2,3],'max_capacity',None,True),
   ([0,1,2,3,4],'max_capacity',1,True),
   ([0,1,2,3,4],'balanced',None,False),
   ([0,1,2,3,4],'max_capacity',None,True)]):
   os.environ['MGBFS_SEGMENT_PACK4']='1' if codec else '0'
   if cap is None:os.environ.pop('MGBFS_SEGMENT_MAX_RANK_STATES',None)
   else:os.environ['MGBFS_SEGMENT_MAX_RANK_STATES']=str(cap)
   opts=NativeOptions(devices=(0,1),autotune=False,memory_backend='permutation_bitmap',
        memory_preset=preset,segment_rows=3,segment_strategy='atomic')
   result=graph.bfs(native_options=opts,output=root/f'case-{i}',start_states=start)
   expected=layers(gens,start)
   depth=verify(result,expected,'pack4-v1' if codec else 'unpacked-soa',cap is None)
   if cap is not None:assert result.report['reason']=='RESOURCE_10',result.report
   now=[w.process.pid for w in pool.workers]
   if ids is None:ids=now
   assert ids==now,(ids,now)
   print('PASS CayleyPy',i,preset,cap,'packed',codec,'depth',depth,'pids',ids,flush=True)
finally:multigpubfs.disable_native()
print('PACK4_CAYLEYPY_TWO_GPU_PASS',root,flush=True)
