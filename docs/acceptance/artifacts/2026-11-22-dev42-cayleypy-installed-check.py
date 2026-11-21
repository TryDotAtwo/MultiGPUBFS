import json,math,os,time,traceback
from pathlib import Path
from cayleypy import CayleyGraph,PermutationGroups
import multigpubfs
from multigpubfs import NativeOptions,persistent_graph_workers

root=Path("/workspace/mgbfs-queue/dev42-installed-check")
root.mkdir(exist_ok=True)
os.environ["MGBFS_SEGMENT_ROWS"]="1048576"
os.environ["MGBFS_PROFILE_CACHE"]=str(root/"profiles")
import importlib.metadata
assert importlib.metadata.version("multigpubfs") == "0.2.0.dev42"
assert "/dev42site/" in multigpubfs.__file__
records=[]
def mahonian(n):
    sizes=[1]
    for k in range(2,n+1):
        q=[0]*(len(sizes)+k-1)
        for j,v in enumerate(sizes):
            for d in range(k): q[j+d]+=v
        sizes=q
    return sizes
previous=None
try:
    with persistent_graph_workers():
        for backend,strategy,ns in [("permutation_bitmap","atomic",(12,9)),("segmented24","warp",(10,9))]:
            multigpubfs.disable_native()
            multigpubfs.enable_native(NativeOptions(output_root=root,devices=tuple(range(8)),autotune=True,memory_backend=backend,memory_preset="balanced",segment_strategy=strategy),recovery=False)
            for n in ns:
                start=time.monotonic()
                result=CayleyGraph(PermutationGroups.coxeter(n),device="cpu").bfs()
                elapsed=time.monotonic()-start
                report=result.report
                assert result.bfs_completed and result.layer_sizes==mahonian(n)
                assert result.num_vertices()==math.factorial(n)
                pids=report["rank_worker_pids"]
                assert len(pids)==8 and len(set(pids))==8
                reused=previous==pids if previous else None
                if previous is not None: assert reused, (previous,pids)
                previous=pids
                ranks=report["segment_memory_by_rank"]
                assert len(ranks)==8 and all(r["segment_strategy"]==strategy for r in ranks)
                profile=report.get("autotune",{})
                if n==12: assert profile.get("status")=="MEASURED_EQUAL_PREFIX_GPU_PROFILE",profile
                entry={"backend":backend,"strategy":strategy,"n":n,"wall_seconds":elapsed,"bfs_seconds":report["bfs_seconds"],"diameter":result.diameter(),"states":result.num_vertices(),"layers_all_match_mahonian":True,"rank_pids":pids,"reused_prior_workers":reused,"profile_status":profile.get("status"),"measured_candidates":len(profile.get("pilots",[])),"shards":report["plan"]["shards"],"batch":report["plan"]["batch"],"runtime_identity":report.get("runtime_identity"),"output":str(result.output)}
                records.append(entry)
                (root/"results.json").write_text(json.dumps({"status":"RUNNING","cases":records},indent=2))
                print("CASE_PASS",json.dumps(entry),flush=True)
    (root/"results.json").write_text(json.dumps({"status":"PASS","cases":records},indent=2))
except BaseException:
    (root/"failure.txt").write_text(traceback.format_exc())
    (root/"results.json").write_text(json.dumps({"status":"FAIL","cases":records},indent=2))
    raise
finally:
    multigpubfs.disable_native()
