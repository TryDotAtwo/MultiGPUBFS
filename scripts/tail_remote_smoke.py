"""Run the real two-rank archived LRX7r4 pipeline for both cuco candidates."""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,'/root/tail-src/scripts')
from run_tail_bfs import run

env=json.loads(Path('/root/tail-build/runtime-env.json').read_text())
env['PATH']='/venv/main/bin:'+env['PATH']
nccl='/venv/main/lib/python3.12/site-packages/nvidia/nccl'
env['LD_LIBRARY_PATH']=nccl+'/lib:'+env['LD_LIBRARY_PATH']
for owner in ('CUCO_INDEXED','CUCO_RANK'):
    config=dict(n=7,r=4,world=2,batch=128,timeout_seconds=120,
        env=dict(MGBFS_PROFILE='DENSE',MGBFS_OWNER_BACKEND=owner,MGBFS_PRE_DEDUP='ON',
            MGBFS_CAPACITY_MODE='max_per_rank', MGBFS_BENCH_CAPACITY='256',
            MGBFS_FUTURE_CAPACITY='512',MGBFS_BUCKET_CAPACITY='256',
            MGBFS_BUCKETS='16',MGBFS_SHARDS='8',MGBFS_JOB_BUCKETS='2',
            MGBFS_LIBRARY_POOL_BYTES=str(64<<20),MGBFS_ARCHIVE_ROWS='128',
            MGBFS_ARCHIVE_SLOTS='64',NCCL_CUMEM_ENABLE='0'))
    result=run(config,Path('/root/tail-src'),Path('/root/tail-smoke-'+owner),env)
    print(owner,result,flush=True)
