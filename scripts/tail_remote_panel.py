"""Matched alternating panel; no claim about graphs too large for this rental."""
import json
import os
import statistics
import sys
import time
from pathlib import Path
sys.path.insert(0,'/root/tail-src/scripts')
from run_tail_bfs import run
from distributed_gpu_bench import run_group

source=Path('/root/tail-src')
root=Path('/root/tail-panel')
root.mkdir(exist_ok=False)
runtime=json.loads(Path('/root/tail-build/runtime-env.json').read_text())
runtime['PATH']='/venv/main/bin:'+runtime['PATH']
runtime['LD_LIBRARY_PATH']='/venv/main/lib/python3.12/site-packages/nvidia/nccl/lib:'+runtime['LD_LIBRARY_PATH']
results=[]
for n in (9,11):
    for repeat in range(3):
        owners=('CUCO_INDEXED','CUCO_RANK') if repeat%2==0 else ('CUCO_RANK','CUCO_INDEXED')
        for owner in owners:
            env=dict(MGBFS_PROFILE='DENSE',MGBFS_OWNER_BACKEND=owner,MGBFS_PRE_DEDUP='ON',
                MGBFS_BENCH_WORLD_SIZE='2',MGBFS_RANK_MAP='0,1',MGBFS_CAPACITY_MODE='max_per_rank',
                MGBFS_BENCH_CAPACITY='1000000',MGBFS_FUTURE_CAPACITY='2000000',
                MGBFS_BUCKET_CAPACITY='1000000',MGBFS_BUCKETS='256',MGBFS_SHARDS='8',
                MGBFS_JOB_BUCKETS='4',MGBFS_LIBRARY_POOL_BYTES=str(512<<20),
                MGBFS_ARCHIVE_ROWS='8192',MGBFS_ARCHIVE_SLOTS='256',NCCL_CUMEM_ENABLE='0',
                MGBFS_STATE_CODEC='permutation_u8',MGBFS_ARCHIVE_CODEC='permutation_u8',
                MGBFS_TRACE_DEPTHS='1',MGBFS_BENCH_WARMUP='0')
            modes=(False,True) if repeat%2==0 else (True,False)
            for archived in modes:
                label=f'n{n}-{owner}-{repeat}-archive{int(archived)}'
                case=root/label
                print('CASE '+label,flush=True)
                at=time.monotonic()
                if archived:
                    run(dict(n=n,r=4,world=2,batch=32768,timeout_seconds=180,env=env),source,case,runtime)
                    ranks=[json.loads((case/f'result/rank-{r}.json').read_text()) for r in range(2)]
                    seconds=max(r['search_complete_seconds'] for r in ranks)
                    layers=[sum(row) for row in zip(*(r['local_layer_sizes'] for r in ranks))]
                else:
                    case.mkdir()
                    command=['torchrun','--standalone','--nproc-per-node=2','--no-python',
                        str(source/'target/release/mgbfs'),'bench','--reference',f'lrx{n}r4','32768',
                        str(case/'bootstrap'),str(case/'unused-archive'),'{RANK_OUT}','--search-only']
                    result=run_group(command,case,'search',dict(os.environ,**runtime,**env),timeout=180)
                    if result['status']!='COMPLETE':
                        raise RuntimeError(label+' INCOMPLETE')
                    seconds=result['search_complete_seconds']
                    layers=result['layer_sizes']
                results.append(dict(n=n,owner=owner,archived=archived,repeat=repeat,
                    search_seconds=seconds,launcher_wall_seconds=time.monotonic()-at,
                    layer_sizes=layers,case=str(case)))
                (root/'results.json').write_text(json.dumps(results,indent=2))
summary=[]
for n in (9,11):
    for owner in ('CUCO_INDEXED','CUCO_RANK'):
        for archived in (False,True):
            rows=[row for row in results if (row['n'],row['owner'],row['archived'])==(n,owner,archived)]
            values=[row['search_seconds'] for row in rows]
            median=statistics.median(values)
            summary.append(dict(n=n,owner=owner,archived=archived,median_seconds=median,
                mad_seconds=statistics.median(abs(x-median) for x in values),repeats=len(values)))
(root/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
