"""Both batch owners: a fatal rank must drain peers and the next pool works."""
import os
import tempfile
import time
from pathlib import Path
from multigpubfs import GraphDefinition
from multigpubfs.generic_session import persistent_graph_workers
from multigpubfs.launch import run_graph

root=Path(tempfile.mkdtemp(prefix="batch-rank-failure-",dir="/workspace"))
graph=GraphDefinition.permutation([[1,0,2,3,4],[0,2,1,3,4],[0,1,3,2,4],[0,1,2,4,3]],list(range(5)))
expected=[len(layer) for layer in graph.exact_layers(1000)]
env=dict(os.environ,MGBFS_SEGMENT_TEST_BATCH="1",MGBFS_SEGMENT_ROWS="3",
         MGBFS_SEGMENT_BATCH_TILE="3",MGBFS_GENERIC_GENERATOR="cuda",
         MGBFS_GENERIC_HISTORY="hash",MGBFS_GENERIC_TRANSPORT="full",MGBFS_GENERIC_SORT="none")
for backend in ("segmented24","permutation_bitmap"):
    bad=dict(env,MGBFS_SEGMENT_TEST_FATAL_RANK="1",MGBFS_SEGMENT_TEST_FATAL_AFTER_DEPTH="2")
    started=time.monotonic()
    pids=[]
    try:
        with persistent_graph_workers() as pool:
            try:
                run_graph(graph,root/(backend+"-failed"),devices=[0,1],autotune=False,
                    shards=4,executable=os.environ["MGBFS_EXECUTABLE"],memory_backend=backend,
                    segment_strategy="batch",_profile_layers=30,_native_env=bad)
            finally:
                pids=[w.process.pid for w in pool.workers]
    except RuntimeError as error:
        assert "NATIVE_DISTRIBUTED_RANK_FAILED" in str(error),repr(error)
    else:
        raise AssertionError("injected rank failure did not fail")
    assert time.monotonic()-started<60
    assert (root/(backend+"-failed")/"failure-evidence").is_dir()
    assert "SEGMENTED_TEST_ONE_RANK_FAILURE" in (root/(backend+"-failed")/"rank-1.log").read_text()
    for pid in pids: assert not Path("/proc",str(pid)).exists(),("worker still alive",pid)
    with persistent_graph_workers():
        report=run_graph(graph,root/(backend+"-fresh"),devices=[0,1],autotune=False,
            shards=4,executable=os.environ["MGBFS_EXECUTABLE"],memory_backend=backend,
            segment_strategy="batch",_profile_layers=30,_native_env=env)
        assert report["status"]=="COMPLETE" and report["layer_sizes"]==expected,report
    print("PASS",backend,"fatal rank cleanup and following fresh persistent job",flush=True)
print("BATCH_RANK_FAILURE_TWO_GPU_PASS",root,flush=True)
