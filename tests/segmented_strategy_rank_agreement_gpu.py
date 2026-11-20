"""Direct two-rank strategy disagreement fails collectively before any BFS layer."""
import json,os,subprocess,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.launch import run_graph
root=Path(tempfile.mkdtemp(prefix="strategy-rank-agreement-",dir="/workspace"))
native=os.environ["MGBFS_EXECUTABLE"]
graph=GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],list(range(4)))
definition=root/"graph.json";definition.write_text(graph.to_json())
base=run_graph(graph,root/"seed",devices=[0,1],autotune=False,shards=4,_batch=2,
               memory_backend="segmented24",segment_strategy="atomic",executable=native,_profile_layers=1)
assert base["layer_sizes"]==[1,3]
config=root/"seed"/"launch.json"
for case,strategies,missing,expected in (
 ("mismatch",("atomic","warp"),(False,False),"SEGMENT_STRATEGY_RANK_MISMATCH"),
 ("invalid",("bogus","bogus"),(False,False),"SEGMENT_STRATEGY_INVALID"),
 ("batch-one-missing",("batch","batch"),(False,True),"SEGMENT_STRATEGY_BATCH_NATIVE_ABI_MISSING"),
 ("batch-both-missing",("batch","batch"),(True,True),"SEGMENT_STRATEGY_BATCH_NATIVE_ABI_MISSING")):
 parent=root/case;parent.mkdir();procs=[]
 try:
  for rank,strategy in enumerate(strategies):
   env=dict(os.environ,MGBFS_SEGMENT_STRATEGY=strategy,MGBFS_EXPERIMENTAL_SEGMENTED24="1",RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE="2")
   env.pop("MGBFS_EXPERIMENTAL_BITMAP",None)
   env["MGBFS_SEGMENT_TEST_BATCH"]="1"
   if missing[rank]:env["MGBFS_SEGMENT_TEST_BATCH_ABI_MISSING"]="1"
   else:env.pop("MGBFS_SEGMENT_TEST_BATCH_ABI_MISSING",None)
   procs.append(subprocess.Popen([native,"graph-rank",str(definition),str(config),str(rank),
     str(parent/"id"),str(parent/f"rank-{rank}"),"0"],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True))
  for rank,proc in enumerate(procs):
   out,err=proc.communicate(timeout=75)
   (parent/f"rank-{rank}.log").write_text(out+"\n"+err)
   assert proc.returncode!=0 and expected in out+err,(case,rank,proc.returncode,out,err)
   assert not (parent/f"rank-{rank}"/"report.json").exists()
 finally:
  for proc in procs:
   if proc.poll() is None:proc.kill();proc.wait()
 print("PASS rank agreement",case,expected,flush=True)
print("STRATEGY_RANK_AGREEMENT_TWO_GPU_PASS",root,flush=True)
