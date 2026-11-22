"""Explicit physical A/B for the optional resident hash parent-extent index."""
import hashlib,json,math,os,tempfile,time
from pathlib import Path
from multigpubfs import GraphDefinition,run_graph
from multigpubfs.generic_session import persistent_graph_workers

world=int(os.environ.get("RESIDENT_AB_WORLD","4"))
n=int(os.environ.get("RESIDENT_AB_N","9"))
shards=int(os.environ.get("RESIDENT_AB_SHARDS","8"))
rows=int(os.environ.get("RESIDENT_AB_SEGMENT_ROWS","8192"))
order=os.environ.get("RESIDENT_AB_PARENT_ORDER","0,1,1,0").split(",")
assert all(v in ("0","1") for v in order)
gens=[]
for i in range(n-1):
    p=list(range(n));p[i],p[i+1]=p[i+1],p[i];gens.append(p)
graph=GraphDefinition.permutation(gens,list(range(n)))
sizes=[1]
for k in range(1,n+1):
    nxt=[0]*(len(sizes)+k-1)
    for d,count in enumerate(sizes):
        for j in range(k):nxt[d+j]+=count
    sizes=nxt
assert sum(sizes)==math.factorial(n)
root=Path(tempfile.mkdtemp(prefix=f"parent-index-s{n}-",dir=os.environ["MGBFS_TEST_TMP_ROOT"]))
exe=Path(os.environ["MGBFS_EXECUTABLE"]);lib=Path(os.environ["MGBFS_CUDA_LIB_FILE"])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
exe_sha,lib_sha=sha(exe),sha(lib)
print("IDENTITY",exe_sha,lib_sha,"root",root,flush=True)
records=[];pids=None;geometry=None
with persistent_graph_workers() as pool:
    for i,mode in enumerate(order):
        os.environ["MGBFS_RESIDENT_HASH_PARENT_INDEX"]=mode
        out=root/f"{i}-parent-index-{mode}"
        t=time.perf_counter()
        r=run_graph(graph,out,devices=list(range(world)),autotune=False,shards=shards,_batch=128,
            segment_rows=rows,memory_backend="segmented24",segment_strategy="atomic",
            execution_engine="device",executable=str(exe))
        wall=time.perf_counter()-t
        assert r["status"]=="COMPLETE" and r["layer_sizes"]==sizes,(mode,r)
        assert json.loads((out/"states.json").read_text())["current"]==[list(reversed(range(n)))]
        assert r["runtime_identity"]["cuda_library_sha256"]==lib_sha
        current=[w.process.pid for w in pool.workers]
        if pids is None:pids=current
        assert current==pids,(pids,current)
        ranks=r["segment_memory_by_rank"]
        assert all(m["resident"]["hash_parent_index_enabled"] is (mode=="1") for m in ranks)
        current_geometry=[(m["resident"]["extent_count"],m["resident_extent_pool_rows_per_bank"]) for m in ranks]
        if geometry is None:geometry=current_geometry
        assert current_geometry==geometry,(geometry,current_geometry)
        records.append({"backend":"segmented24","hash_parent_index_enabled":mode=="1","world":world,
            "graph":f"S{n} adjacent transpositions","vertices":sum(sizes),"layers":len(sizes),
            "layer_sizes":sizes,"generated_edges":sum(sizes)*(n-1),"segment_rows_requested":rows,
            "shards":shards,"batch":128,"bfs_seconds":r["bfs_seconds"],
            "setup_seconds":r["setup_seconds"],"launch_wall_seconds":r["launch_wall_seconds"],
            "observed_wall_seconds":wall,"rank_worker_pids":pids,"executable_sha256":exe_sha,
            "cuda_sha256":lib_sha,"rank_allocations":ranks,
            "exact_check":"all Mahonian layer counts and exact final state"})
        (root/"comparison.json").write_text(json.dumps(records,indent=2))
        print("PASS parent-index",mode,"bfs_s",r["bfs_seconds"],"wall_s",wall,flush=True)
print("EVIDENCE",root,flush=True)
