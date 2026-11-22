"""Opt-in physical test for legacy resident packet-identity compatibility."""
import json,os,tempfile
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
from multigpubfs.generic_session import persistent_graph_workers
root=Path(tempfile.mkdtemp(prefix="resident-payload-compat-",dir=os.environ.get("MGBFS_TEST_TMP_ROOT")))
os.environ["LD_LIBRARY_PATH"]=os.environ["RESIDENT_OLD_LIB"]+":"+os.environ["RESIDENT_LD_TAIL"]
os.environ["MGBFS_RESIDENT_PACKET_RECORDS"]="8"
os.environ["MGBFS_RESIDENT_LANES"]="8"
graph=GraphDefinition.permutation([[1,0,2,3],[0,2,1,3],[0,1,3,2]],[0,1,2,3])
layers=graph.exact_layers(1000)
def run(out):
 return run_graph(graph,out,devices=[0,1],autotune=False,shards=2,_batch=8,segment_rows=64,
  memory_backend="permutation_bitmap",segment_strategy="atomic",execution_engine="device",
  executable=os.environ["MGBFS_EXECUTABLE"])
for mode in ("carry","verify"):
 os.environ["MGBFS_RESIDENT_PAYLOAD_ID"]=mode
 out=root/f"unsupported-{mode}"
 with persistent_graph_workers():
  try:run(out)
  except RuntimeError as e:assert "NATIVE_DISTRIBUTED_RANK_FAILED" in str(e),e
  else:raise AssertionError("old library silently accepted unsupported payload identity")
  logs="".join((out/f"rank-{r}.log").read_text() for r in (0,1))
  assert "RESIDENT_PAYLOAD_ID_UNSUPPORTED" in logs,logs
  assert not (out/"report.json").exists()
 print("PASS old library rejects payload identity",mode,flush=True)
os.environ.pop("MGBFS_RESIDENT_PAYLOAD_ID",None)
with persistent_graph_workers():
 r=run(root/"supported")
 assert r["status"]=="COMPLETE" and r["layer_sizes"]==[len(x) for x in layers],r
 assert all(m["resident"]["lanes"]==8 and m["resident"]["payload_identity"]=="legacy" for m in r["segment_memory_by_rank"])
 assert r["runtime_identity"]["cuda_library_sha256"]==os.environ["RESIDENT_OLD_SHA"]
 print("PASS old library retains legacy payload",flush=True)
old_executable=os.environ.get("RESIDENT_OLD_EXECUTABLE")
if old_executable:
 os.environ["LD_LIBRARY_PATH"]=os.environ["RESIDENT_NEW_LIB"]+":"+os.environ["RESIDENT_LD_TAIL"]
 os.environ["MGBFS_EXECUTABLE"]=old_executable
 with persistent_graph_workers():
  old_result=run(root/"old-executable-new-library")
  assert old_result["status"]=="COMPLETE" and old_result["layer_sizes"]==[len(x) for x in layers],old_result
  assert old_result["runtime_identity"]["cuda_library_sha256"]==os.environ["RESIDENT_NEW_SHA"]
  assert all(m["resident"]["lanes"]==8 for m in old_result["segment_memory_by_rank"])
  print("PASS old executable with new library and legacy payload",flush=True)
(root/"acceptance.json").write_text(json.dumps({"old_cuda_library_sha256":os.environ["RESIDENT_OLD_SHA"],
 "world":2,"unsupported_modes":["carry","verify"],"supported_mode":"legacy","old_executable_checked":bool(old_executable),"layers":[len(x) for x in layers]},indent=2))
print("EVIDENCE",root,flush=True)
