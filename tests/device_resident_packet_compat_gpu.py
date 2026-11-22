"""Opt-in physical test: old resident CUDA library rejects batched packets."""
import json, os, tempfile
from pathlib import Path
from multigpubfs import GraphDefinition, run_graph
from multigpubfs.generic_session import persistent_graph_workers

root=Path(tempfile.mkdtemp(prefix="resident-packet-compat-",dir=os.environ.get("MGBFS_TEST_TMP_ROOT")))
os.environ["LD_LIBRARY_PATH"]=os.environ["RESIDENT_OLD_LIB"]+":"+os.environ["RESIDENT_LD_TAIL"]
graph=GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],[0,1,2,3])
layers=graph.exact_layers(1000)
def run(out):
 return run_graph(graph,out,devices=[0,1],autotune=False,shards=2,_batch=8,segment_rows=64,
  memory_backend="permutation_bitmap",segment_strategy="atomic",execution_engine="device",
  executable=os.environ["MGBFS_EXECUTABLE"])

os.environ["MGBFS_RESIDENT_PACKET_RECORDS"]="8"
with persistent_graph_workers():
 try:run(root/"unsupported")
 except RuntimeError as e:
  assert "NATIVE_DISTRIBUTED_RANK_FAILED" in str(e),e
 else:raise AssertionError("old library silently accepted 8-record packets")
 logs="".join((root/"unsupported"/f"rank-{r}.log").read_text() for r in (0,1))
 assert "RESIDENT_PACKET_UNSUPPORTED requested=128 max=16" in logs,logs
 assert not (root/"unsupported"/"report.json").exists()
 print("PASS old library rejects unsupported packet size before resident launch",flush=True)

os.environ["MGBFS_RESIDENT_PACKET_RECORDS"]="1"
with persistent_graph_workers():
 r=run(root/"supported")
 assert r["status"]=="COMPLETE" and r["layer_sizes"]==[len(x) for x in layers],r
 assert all(m["resident"]["packet_records"]==1 for m in r["segment_memory_by_rank"])
 assert r["runtime_identity"]["cuda_library_sha256"]==os.environ["RESIDENT_OLD_SHA"]
 print("PASS old library works with a one-record packet",flush=True)
(root/"acceptance.json").write_text(json.dumps({"old_cuda_library_sha256":os.environ["RESIDENT_OLD_SHA"],
 "world":2,"unsupported_records":8,"supported_records":1,"layers":[len(x) for x in layers]},indent=2))
print("EVIDENCE",root,flush=True)
