"""Executable and two real CUDA libraries: reject unavailable strategy safely."""
import json,os,subprocess,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.launch import run_graph

native=os.environ["MGBFS_EXECUTABLE"]
old_lib=os.environ["MGBFS_OLD_CUDA_LIB_DIR"]
new_lib=os.environ["MGBFS_NEW_CUDA_LIB_DIR"]
tail="/workspace/nccl/lib:/usr/local/cuda/lib64"
old=dict(os.environ,LD_LIBRARY_PATH=old_lib+":"+tail)
new=dict(os.environ,LD_LIBRARY_PATH=new_lib+":"+tail)
for e in (old,new):
 e.pop("MGBFS_SEGMENT_TEST_BATCH",None)
 e.pop("MGBFS_SEGMENT_TEST_BATCH_ABI_MISSING",None)
def key(env):
 return json.loads(subprocess.run([native,"key-info"],env=env,check=True,capture_output=True,text=True).stdout)
assert key(old)["segmented_strategies_v1"]=={"atomic":True,"warp":True,"batch":False}
assert key(new)["segmented_strategies_v1"]=={"atomic":True,"warp":True,"batch":True}
gens=[]
for i in range(3):
 x=list(range(4));x[i],x[i+1]=x[i+1],x[i];gens.append(x)
graph=GraphDefinition.permutation(gens,list(range(4)))
expected=list(map(len,graph.exact_layers(100)))
out=Path(tempfile.mkdtemp(prefix="segmented-compat-",dir="/workspace"))
kwargs=dict(devices=[0,1],shards=4,autotune=False,executable=native,memory_backend="segmented24",
            segment_rows=128,_batch=4,_profile_layers=30)
for label,strategy,env in (("old-atomic","atomic",old),("new-batch","batch",new),
                           ("old-again","atomic",old)):
 r=run_graph(graph,out/label,segment_strategy=strategy,_native_env=env,**kwargs)
 assert r["status"]=="COMPLETE" and r["layer_sizes"]==expected,(label,r)
 print("PASS",label,flush=True)
try:
 run_graph(graph,out/"old-batch",segment_strategy="batch",_native_env=old,**kwargs)
except RuntimeError as ex:
 assert "NATIVE_SEGMENT_STRATEGY_UNAVAILABLE" in str(ex),str(ex)
 print("PASS old-batch explicit rejection",flush=True)
else:raise AssertionError("batch accepted by old CUDA library")
print("OLD_NATIVE_COMPAT_TWO_GPU_PASS",out,flush=True)
