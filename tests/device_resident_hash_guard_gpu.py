"""Verify unsupported resident hash graphs fail before execution on real ranks."""
import os,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
from multigpubfs.generic_session import persistent_graph_workers

root=Path(tempfile.mkdtemp(prefix="resident-hash-guard-",dir=os.environ.get("MGBFS_TEST_TMP_ROOT")))
def rot(k):return [(i+k)%6 for i in range(6)]
directed=GraphDefinition.permutation([rot(1),rot(4)],list(range(6)))
small=GraphDefinition.permutation([[1,0,2],[0,2,1]],[0,1,2])
def run(g,out,world):
 return run_graph(g,out,devices=list(range(world)),autotune=False,shards=2,_batch=2,segment_rows=2,
  memory_backend="segmented24",segment_strategy="atomic",execution_engine="device",
  executable=os.environ["MGBFS_EXECUTABLE"])
def all_text(out):
 texts=[]
 for f in out.rglob("*"):
  if f.is_file() and f.stat().st_size<2_000_000:
   try:texts.append(f.read_text(errors="replace"))
   except OSError:pass
 return "\n".join(texts)
with persistent_graph_workers() as pool:
 try:run(directed,root/"directed",2)
 except Exception as e:err=repr(e)
 else:raise AssertionError("directed graph unexpectedly accepted")
 evidence=err+"\n"+all_text(root/"directed")
 assert "RESIDENT_HASH_REQUIRES_INVERSE_CLOSED_PERMUTATION" in evidence,evidence[-10000:]
 assert not (root/"directed"/"report.json").exists()
 print("PASS directed rejected before resident BFS",err,flush=True)
with persistent_graph_workers():
 r=run(small,root/"healthy",2)
 assert r["status"]=="COMPLETE" and r["layer_sizes"]==[1,2,2,1],r
 print("PASS healthy new group",flush=True)
print("EVIDENCE",root,flush=True)
