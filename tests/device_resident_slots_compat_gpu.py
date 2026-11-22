"""Physical compatibility: new runtime must not ask an older CUDA library for multi-slot rings."""
import json, os, tempfile
from pathlib import Path
from multigpubfs import GraphDefinition, run_graph
from multigpubfs.generic_session import persistent_graph_workers

root=Path(tempfile.mkdtemp(prefix="resident-slots-compat-",dir=os.environ.get("MGBFS_TEST_TMP_ROOT")))
os.environ["LD_LIBRARY_PATH"]=os.environ["RESIDENT_OLD_LIB"]+":"+os.environ["RESIDENT_LD_TAIL"]
graph=GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],[0,1,2,3])
layers=graph.exact_layers(1000)
def run(out,backend):
 return run_graph(graph,out,devices=[0,1],autotune=False,shards=2,_batch=8,segment_rows=64,
  memory_backend=backend,segment_strategy="atomic",execution_engine="device",
  executable=os.environ["MGBFS_EXECUTABLE"])

for backend in ("permutation_bitmap","segmented24"):
 for slots in (2,4):
  os.environ["MGBFS_RESIDENT_MAILBOX_SLOTS"]=str(slots)
  out=root/f"{backend}-unsupported-{slots}"
  with persistent_graph_workers():
   try:run(out,backend)
   except RuntimeError as e:assert "NATIVE_DISTRIBUTED_RANK_FAILED" in str(e),e
   else:raise AssertionError("old library silently accepted unsupported ring capacity")
   logs="".join((out/f"rank-{r}.log").read_text() for r in (0,1))
   assert f"RESIDENT_MAILBOX_SLOTS_UNSUPPORTED requested={slots} max=1" in logs,logs
   assert not (out/"report.json").exists()
  print("PASS old library rejects slots",backend,slots,flush=True)
 os.environ.pop("MGBFS_RESIDENT_MAILBOX_SLOTS",None)
 if backend=="segmented24":
  os.environ["MGBFS_RESIDENT_HASH_PARENT_INDEX"]="1"
  out=root/"hash-parent-index-unsupported"
  with persistent_graph_workers():
   try:run(out,backend)
   except RuntimeError as e:assert "NATIVE_DISTRIBUTED_RANK_FAILED" in str(e),e
   else:raise AssertionError("old library silently accepted parent index")
   logs="".join((out/f"rank-{r}.log").read_text() for r in (0,1))
   assert "RESIDENT_HASH_PARENT_INDEX_UNSUPPORTED" in logs,logs
   assert not (out/"report.json").exists()
  print("PASS old library rejects parent index",flush=True)
  os.environ.pop("MGBFS_RESIDENT_HASH_PARENT_INDEX",None)
 with persistent_graph_workers():
  r=run(root/f"{backend}-default",backend)
  assert r["status"]=="COMPLETE" and r["layer_sizes"]==[len(x) for x in layers],r
  assert all(m["resident"]["slots_per_ring"]==1 for m in r["segment_memory_by_rank"])
  assert r["runtime_identity"]["cuda_library_sha256"]==os.environ["RESIDENT_OLD_SHA"]
 print("PASS old library keeps one slot",backend,flush=True)
(root/"acceptance.json").write_text(json.dumps({"old_cuda_library_sha256":os.environ["RESIDENT_OLD_SHA"],
 "world":2,"unsupported_slots":[2,4],"supported_slots":1,"layers":[len(x) for x in layers]},indent=2))
print("EVIDENCE",root,flush=True)
