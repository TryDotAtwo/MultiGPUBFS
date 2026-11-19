"""Physical CayleyPy/persistent configuration test on two CUDA devices."""
import json, tempfile
from pathlib import Path
from cayleypy import CayleyGraph, PermutationGroups
from multigpubfs.cayleypy_backend import NativeOptions, bfs
from multigpubfs.generic_session import persistent_graph_workers

def mahonian(n):
    sizes=[1]
    for degree in range(1,n+1):
        nxt=[0]*(len(sizes)+degree-1)
        for i,x in enumerate(sizes):
            for j in range(degree):nxt[i+j]+=x
        sizes=nxt
    return sizes

root=Path(tempfile.mkdtemp(prefix="segment-config-api-",dir="/workspace"))
graph=CayleyGraph(PermutationGroups.coxeter(6),device="cpu")
records=[]
with persistent_graph_workers():
    workers=None
    for mode in ("segmented","permutation_bitmap"):
        for segment_rows in (64,1<<32):
            options=NativeOptions(devices=(0,1),autotune=False,
                                  memory_backend=mode,segment_rows=segment_rows)
            result=bfs(graph,output=root/(mode+"-"+str(segment_rows)),native_options=options)
            assert result.bfs_completed and result.layer_sizes==mahonian(6),(mode,segment_rows,result.report)
            report=result.report
            geometry=report["segment_memory_by_rank"]
            assert len(geometry)==2
            assert all(x["configured_segment_rows"]==segment_rows for x in geometry),geometry
            assert all(1<=x["rows_per_bank"]<=segment_rows for x in geometry),geometry
            if workers is None:workers=report["rank_worker_pids"]
            assert report["rank_worker_pids"]==workers
            item={"backend":mode,"configured_segment_rows":segment_rows,
                  "actual_rows":[x["rows_per_bank"] for x in geometry],
                  "shards":report["plan"]["shards"],"bfs_seconds":report["bfs_seconds"],
                  "workers":workers}
            records.append(item);print(json.dumps(item),flush=True)
(root/"evidence.json").write_text(json.dumps(records,indent=2))
print("EVIDENCE",root,flush=True)
