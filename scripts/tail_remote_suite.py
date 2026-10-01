"""Finite real-GPU grid and matched LRX15r4 archived/search-only panel."""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path


def trace_layers(path,world=2):
    begins,ends={},{}
    for line in path.read_text(errors='replace').splitlines():
        match=re.search(r'MGBFS_DEPTH_(BEGIN|END) rank=(\d+) depth=(\d+) (.*)',line)
        if match:
            fields=dict(re.findall(r'(\w+)=([^\s]+)',match[4]))
            (begins if match[1]=='BEGIN' else ends)[int(match[2]),int(match[3])]=fields
    result=[]
    depth=0
    while all((rank,depth) in begins and (rank,depth) in ends for rank in range(world)):
        result.append(dict(depth=depth,states=sum(int(begins[r,depth]['count']) for r in range(world)),
            seconds=max(float(ends[r,depth]['seconds']) for r in range(world))))
        depth+=1
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--runtime-env',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--deadline-unix',type=float,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.source/'scripts'))
    from bfs_tail_archive import atomic_json
    from sweep_tail_bfs import execute,pairs
    from run_tail_bfs import run
    from distributed_gpu_bench import run_group
    args.root.mkdir(exist_ok=False)
    runtime=json.loads(args.runtime_env.read_text())
    runtime['PATH']='/venv/main/bin:'+runtime['PATH']
    runtime['LD_LIBRARY_PATH']='/root/tail-nccl/nvidia/nccl/lib:'+runtime['LD_LIBRARY_PATH']
    env=dict(MGBFS_PROFILE='DENSE',MGBFS_OWNER_BACKEND='CUCO_RANK',MGBFS_PRE_DEDUP='ON',
        MGBFS_BENCH_WORLD_SIZE='2',MGBFS_RANK_MAP='0,1',MGBFS_CAPACITY_MODE='max_per_rank',
        MGBFS_BENCH_CAPACITY='4000000',MGBFS_FUTURE_CAPACITY='8000000',
        MGBFS_BUCKET_CAPACITY='4000000',MGBFS_BUCKETS='256',MGBFS_SHARDS='8',
        MGBFS_JOB_BUCKETS='4',MGBFS_LIBRARY_POOL_BYTES=str(2<<30),
        MGBFS_ARCHIVE_ROWS='8192',MGBFS_ARCHIVE_SLOTS='256',NCCL_CUMEM_ENABLE='0',
        MGBFS_STATE_CODEC='permutation_u8',MGBFS_ARCHIVE_CODEC='permutation_u8',
        MGBFS_TRACE_DEPTHS='1',MGBFS_BENCH_WARMUP='0')
    panel=[]
    # Alternate modes and owners; both modes may hit identical resource limits.
    for repeat in range(3):
        for owner in (('CUCO_INDEXED','CUCO_RANK') if repeat%2==0 else ('CUCO_RANK','CUCO_INDEXED')):
            for archived in ((False,True) if repeat%2==0 else (True,False)):
                if args.deadline_unix-time.time()<180:break
                case=args.root/f'n15-{owner}-{repeat}-archive{int(archived)}'
                cfg=dict(n=15,r=4,world=2,batch=32768,timeout_seconds=90,env=dict(env,MGBFS_OWNER_BACKEND=owner))
                print('CASE '+case.name,flush=True)
                started=time.monotonic()
                if archived:
                    try:run(cfg,args.source,case,runtime)
                    except Exception as error:print('ARCHIVED_STOP '+type(error).__name__,flush=True)
                    manifest=json.loads((case/'saved/manifest.json').read_text())
                    status=manifest['status'];reason=manifest['stop_reason'];layers=manifest['layers']
                else:
                    case.mkdir()
                    command=['torchrun','--standalone','--nproc-per-node=2','--no-python',
                        str(args.source/'target/release/mgbfs'),'bench','--reference','lrx15r4','32768',
                        str(case/'bootstrap'),str(case/'unused'),'{RANK_OUT}','--search-only']
                    report=run_group(command,case,'search',dict(os.environ,**runtime,**cfg['env']),timeout=90)
                    status=report['status'];reason=report.get('failure_code',status)
                    layers=trace_layers(case/'search.log')
                panel.append(dict(owner=owner,repeat=repeat,archived=archived,status=status,
                    reason=reason,layers=layers,case=str(case),launcher_wall_seconds=time.monotonic()-started))
                atomic_json(args.root/'panel.json',panel)
    common=min((len(row['layers']) for row in panel),default=0)
    counts_match=bool(common) and all(
        [layer['states'] for layer in row['layers'][:common]]==
        [layer['states'] for layer in panel[0]['layers'][:common]] for row in panel)
    atomic_json(args.root/'panel-comparison.json',dict(
        scope='same completed prefix only unless every case is COMPLETE',
        all_cases_complete=len(panel)==12 and all(row['status']=='COMPLETE' for row in panel),
        cases=len(panel),common_completed_layers=common,counts_match=counts_match,
        paired_seconds=[dict(owner=row['owner'],repeat=row['repeat'],archived=row['archived'],
            layer_seconds=[layer['seconds'] for layer in row['layers'][:common]]) for row in panel]))
    base=dict(world=2,batch=32768,timeout_seconds=60,env=env,run_id='20261002-grid')
    def adaptive(config,source,case,runtime):
        order=math.factorial(config['n'])//math.factorial(config['r'])
        capacity=max(256,min(4_000_000,order))
        config['env'].update(MGBFS_BENCH_CAPACITY=str(capacity),MGBFS_FUTURE_CAPACITY=str(capacity*2),
            MGBFS_BUCKET_CAPACITY=str(capacity),MGBFS_LIBRARY_POOL_BYTES=str(
                64<<20 if order<65536 else 512<<20 if order<4_000_000 else 2<<30))
        return run(config,source,case,runtime)
    remaining=args.deadline_unix-time.time()-180
    if remaining>0:
        ledger=execute(base,args.source,args.root/'grid',runtime,pairs(2,15),remaining,adaptive)
        counts={status:sum(row['status']==status for row in ledger['cases'].values())
                for status in ('COMPLETE','INCOMPLETE')}
        atomic_json(args.root/'suite-summary.json',dict(grid=counts,pending=ledger['pending'],
            panel_cases=len(panel),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__=='__main__':main()
