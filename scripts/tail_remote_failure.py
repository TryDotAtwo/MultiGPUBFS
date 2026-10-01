"""Real capacity-stop gate; no unfinished layer may be labeled complete."""
import json
import sys
from pathlib import Path
sys.path.insert(0,'/root/tail-src/scripts')
from run_tail_bfs import run
runtime=json.loads(Path('/root/tail-build/runtime-env.json').read_text())
runtime['PATH']='/venv/main/bin:'+runtime['PATH']
runtime['LD_LIBRARY_PATH']='/venv/main/lib/python3.12/site-packages/nvidia/nccl/lib:'+runtime['LD_LIBRARY_PATH']
config=dict(n=7,r=4,world=2,batch=4,timeout_seconds=60,env=dict(
    MGBFS_PROFILE='DENSE',MGBFS_OWNER_BACKEND='CUCO_RANK',MGBFS_PRE_DEDUP='ON',
    MGBFS_CAPACITY_MODE='max_per_rank',MGBFS_BENCH_CAPACITY='8',MGBFS_FUTURE_CAPACITY='16',
    MGBFS_BUCKET_CAPACITY='8',MGBFS_BUCKETS='16',MGBFS_SHARDS='8',MGBFS_JOB_BUCKETS='2',
    MGBFS_LIBRARY_POOL_BYTES=str(64<<20),MGBFS_ARCHIVE_ROWS='4',MGBFS_ARCHIVE_SLOTS='256',
    NCCL_CUMEM_ENABLE='0'))
root=Path('/root/tail-capacity-failure')
try:
    run(config,Path('/root/tail-src'),root,runtime)
except RuntimeError:
    manifest=json.loads((root/'saved/manifest.json').read_text())
    assert manifest['status']=='INCOMPLETE'
    assert manifest['last_completed_layer']>=0
    assert manifest['last_completed_layer']<11
    assert sum(f['bytes'] for f in manifest['files'])<=1_000_000_000
    assert all(f['depth']<=manifest['last_completed_layer'] for f in manifest['files'])
    print(json.dumps(dict(status='PASS_INCOMPLETE_CAPACITY',
        last_completed_layer=manifest['last_completed_layer'],
        reason=manifest['stop_reason'],completed_states=sum(x['states'] for x in manifest['layers'])),indent=2))
else:
    raise RuntimeError('unexpected COMPLETE on intentionally undersized capacity')
